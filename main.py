import itertools
import torch
from tqdm import tqdm
from torch.cuda import amp
from clip import clip
from config import CFG
from dataset import create_dataloader
from model import CLIPModel, VanillaVAE
from parsejson import dataparse
from utils import AvgMeter

if __name__ == '__main__':
    train_dict, _, test_dict = dataparse(CFG)
    clip_model = CLIPModel(CFG).float()
    clip_model = clip_model.to(CFG.device)

    params = [
        {"params": clip_model.model.visual.parameters(), "lr": CFG.image_encoder_lr},
        {"params": clip_model.model.transformer.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            clip_model.image_projection.parameters(), clip_model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]

    clip_optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        clip_optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    transform = clip_model.preprocess
    triplet_train_loader, plain_train_loader, test_loader = \
        create_dataloader(CFG, train_dict, test_dict, transform)

    if CFG.triplet:
        train_loader = triplet_train_loader
    else:
        train_loader = plain_train_loader

    num_batch = len(train_loader)

    clip_model.train()
    print('-'*30 + 'Training CLIP Model' + '-'*30)
    for e in range(2):
        number_cls = train_loader.number_cls
        loss_meter = AvgMeter()
        pbar = tqdm(enumerate(train_loader), total=num_batch)
        for n_iter, (imgs, pids, captions) in pbar:
            imgs = imgs.to(CFG.device)
            txts = clip.tokenize(captions, truncate=True).to(CFG.device)
            with amp.autocast():
                ploss, sloss, tloss = clip_model(imgs, txts, pids)
            loss = ploss + CFG.weight_sc * sloss + CFG.weight_tri * tloss
            clip_optimizer.zero_grad()
            loss.backward()
            clip_optimizer.step()
            count = imgs.size(0)
            loss_meter.update(loss.item(), count)
            pbar.set_description("Epoch %d Loss: %.2f" % (e, loss_meter.avg))
            break
        lr_scheduler.step(loss_meter.avg)

    vae = VanillaVAE(CFG.in_channels, CFG.latent_dim).to(CFG.device)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=CFG.vae_lr, betas=CFG.vae_betas,
                                     eps=CFG.vae_eps, weight_decay=0., amsgrad=False, maximize=False,
                                     foreach=None, capturable=False, differentiable=False, fused=False)
    print('-' * 30 + 'Training VAE Model' + '-' * 30)
    vae.train()
    num_batch = len(plain_train_loader)
    for e in range(CFG.epochs):
        loss_meter = AvgMeter()
        pbar = tqdm(enumerate(plain_train_loader), total=num_batch)
        for n_iter, (imgs, pids, captions) in pbar:
            imgs = imgs.to(CFG.device)
            results = vae(imgs)
            vae_loss = vae.loss_function(*results, M_N=1.0)['loss']
            vae_loss.backward()
            vae_optimizer.step()
            count = imgs.size(0)
            loss_meter.update(vae_loss.item(), count)
            pbar.set_description("Epoch %d Loss: %.2f" % (e, loss_meter.avg))
