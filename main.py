import itertools
import torch
import clip
import os
from tqdm import tqdm
from torch.cuda import amp
from config import CFG
from dataset import create_dataloader
from model import CLIPModel
from parsejson import dataparse
from utils import AvgMeter

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"
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

    if CFG.triplet:
        train_loader = triplet_train_loader
    else:
        train_loader = plain_train_loader

    num_batch = len(train_loader)

    clip_model.train()
    for e in range(CFG.epochs):
        number_cls = train_loader.number_cls
        loss_meter = AvgMeter()
        pbar = tqdm(enumerate(train_loader), total=num_batch)
        for n_iter, (imgs, pids, captions) in pbar:
            imgs = imgs.to(CFG.device)
            txts = clip.tokenize(captions, truncate=True).to(CFG.device)
            with amp.autocast():
                ploss, sloss, tloss = clip_model(imgs, txts, pids)
            loss = ploss + CFG.weight_sc * sloss + CFG.weight_tri * tloss
            pbar.set_description("Epoch %d Loss: %.2f" % (e, loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count = imgs.size(0)
            loss_meter.update(loss.item(), count)
        lr_scheduler.step(loss_meter.avg)

