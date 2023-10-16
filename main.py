import itertools
import os
import torch
from tqdm import tqdm
from torch.cuda import amp
from clip import clip
from clip.model import CLIPModel
from config import CFG
from dataset import edge_dataloaders, server_dataloader
from latent.model import LatentDiffusionModel
from parsejson import dataparse
from utils import AvgMeter, sample
from vae.model import VanillaVAE

if __name__ == '__main__':
    train_dict, _, test_dict = dataparse(CFG)
    clip_model = CLIPModel(CFG)
    vae = VanillaVAE(CFG.in_channels, CFG.latent_dim)
    transform = clip_model.preprocess
    triplet_train_loader, vae_train_loader, test_loader = \
        edge_dataloaders(CFG, train_dict, test_dict, transform)
    if CFG.stage == 'clip':
        clip_model = clip_model.to(CFG.device)
        params = [
            {"params": clip_model.model.visual.parameters(), "lr": CFG.image_encoder_lr},
            {"params": clip_model.model.transformer.parameters(), "lr": CFG.text_encoder_lr},
            {"params": itertools.chain(
                clip_model.image_projection.parameters(), clip_model.text_projection.parameters()
            ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
        ]

        clip_optimizer = torch.optim.Adam(params)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            clip_optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
        )
        if CFG.triplet:
            train_loader = triplet_train_loader
        else:
            train_loader = vae_train_loader

        num_batch = len(train_loader)

        clip_model.train()
        print('-'*30 + 'Training CLIP Model' + '-'*30)
        scaler = amp.GradScaler()

        for e in range(CFG.epochs):
            number_cls = train_loader.number_cls
            loss_meter = AvgMeter()
            pbar = tqdm(enumerate(train_loader), total=num_batch)
            for n_iter, (imgs, pids, captions) in pbar:
                imgs = imgs.to(CFG.device)
                txts = clip.tokenize(captions, truncate=True).to(CFG.device)
                clip_optimizer.zero_grad()
                with amp.autocast():
                    ploss, sloss, tloss = clip_model(imgs, txts, pids)
                    loss = ploss + CFG.weight_sc * sloss + CFG.weight_tri * tloss
                    scaler.scale(loss).backward()
                    scaler.step(clip_optimizer)
                    scaler.update()
                count = imgs.size(0)
                loss_meter.update(loss.item(), count)
                pbar.set_description("Epoch %d Loss: %.2f" % (e, loss_meter.avg))
            lr_scheduler.step(loss_meter.avg)

        print('-' * 30 + 'Saving Text Embedding' + '-' * 30)
        num_batch = len(vae_train_loader)
        pbar = tqdm(enumerate(vae_train_loader), total=num_batch)
        embeddings, texts, personids = [], [], []
        clip_model.eval()
        with torch.no_grad():
            txts = clip.tokenize(['']).to(CFG.device)
            _, uncond_embedding = clip_model.model.encode_text(txts)
            for n_iter, (imgs, pids, captions) in pbar:
                txts = clip.tokenize(captions, truncate=True).to(CFG.device)
                _, embeding = clip_model.model.encode_text(txts)
                embeddings.append(embeding.detach().cpu())
                texts += captions
                personids.append(pids)

        d = {
            'embedings': torch.cat(embeddings, dim=0),
            'personids': torch.cat(personids, dim=0),
            'texts': texts,
             'uncond_embedding': uncond_embedding.detach().cpu()
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(d, './checkpoints/embedings.pt')

    elif CFG.stage == 'vae':
        vae = vae.to(CFG.device)
        vae_optimizer = torch.optim.Adam(vae.parameters(), lr=CFG.vae_lr, betas=CFG.vae_betas,
                                         eps=CFG.vae_eps, weight_decay=0., amsgrad=False, maximize=False,
                                         foreach=None, capturable=False, differentiable=False, fused=False)
        print('-' * 30 + 'Training VAE Model' + '-' * 30)
        vae.train()
        num_batch = len(vae_train_loader)
        for e in range(CFG.epochs):
            loss_meter = AvgMeter()
            pbar = tqdm(enumerate(vae_train_loader), total=num_batch)
            for n_iter, (imgs, pids, captions) in pbar:
                imgs = imgs.to(CFG.device)
                results = vae(imgs)
                vae_loss = vae.loss_function(*results, M_N=1.0)['loss']
                vae_loss.backward()
                vae_optimizer.step()
                count = imgs.size(0)
                loss_meter.update(vae_loss.item(), count)
                pbar.set_description("Epoch %d Loss: %.2f" % (e, loss_meter.avg))
        print('-' * 30 + 'Saving VAE Model' + '-' * 30)
        state = {
            'net': vae.state_dict(),
            'epoch': e,
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state, './checkpoints/' + CFG.stage + '.pt')

    elif CFG.stage == 'latdiff':
        print('-' * 30 + 'Traning Latent Diffusion Model' + '-' * 30)
        vae_decoder = vae.decoder_input
        diffusion = LatentDiffusionModel(vae_decoder, CFG.scale, device=CFG.device,
                                         sampler_name=CFG.sampler_name, n_steps=CFG.steps)
        latent_train_loader, uncond_embedding = server_dataloader(CFG)
        optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=CFG.diff_lr)
        for e in range(CFG.epochs):
            loss_meter = AvgMeter()
            num_batch = len(latent_train_loader)
            pbar = tqdm(enumerate(latent_train_loader), total=num_batch)
            for n_iter, (embeddings, pids, captions) in pbar:
                embeddings = embeddings.to(CFG.device)
                optimizer.zero_grad()
                # Generate data by VAE decoder
                noise = sample(len(captions), CFG.latent_dim, CFG.device)
                x0 = diffusion.model.vae_decoder(noise)
                # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
                noise = torch.randn_like(x0)
                # Calculate loss
                loss = diffusion.model.loss(x0, embeddings, uncond_embedding, noise)
                # Compute gradients
                loss.backward()
                # Take an optimization step
                optimizer.step()

        diffusion.infer(dest_path='outputs',
                      prompt=opt.prompt,
                      batch_size=CFG.batch_size,
                      uncond_scale=CFG.scale)
    else:
        raise NotImplementedError('Select CLIP, VAE and Latent diffusion models for training.')
