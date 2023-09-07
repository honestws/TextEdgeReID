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

    if CFG.triplet:
        train_loader = triplet_train_loader
    else:
        train_loader = plain_train_loader

    num_batch = len(train_loader)

    clip_model.train()
    for e in range(CFG.epochs):
        number_cls = train_loader.number_cls
        pbar = tqdm(enumerate(train_loader), total=num_batch)
        for n_iter, (imgs, pids, captions) in pbar:
            imgs = imgs.to(CFG.device)
            txts = clip.tokenize(captions, truncate=True).to(CFG.device)
            with amp.autocast():
                loss = clip_model(imgs, txts)
            pbar.set_description("Epoch %d Loss: %.2f" % (e, loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

