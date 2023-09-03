import itertools
import torch
import clip
from tqdm import tqdm
from config import CFG
from torch.cuda import amp
from dataset import create_dataloader
from model import CLIPModel
from parsejson import dataparse

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

    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    transform = clip_model.preprocess
    triplet_train_loader, plain_train_loader, test_loader = \
        create_dataloader(CFG, train_dict, test_dict, transform)

    num_batch = len(plain_train_loader)

    for e in range(CFG.epochs):
        pbar = tqdm(enumerate(plain_train_loader))
        for n_iter, (imgs, pids, captions) in pbar:
            imgs = imgs.to(CFG.device)
            txts = clip.tokenize(captions).to(CFG.device)
            with amp.autocast():
                loss = clip_model(imgs, txts)
            pbar.set_description("Epoch %d Loss: %.2f [%d/%d]" % (e, loss, n_iter, num_batch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
