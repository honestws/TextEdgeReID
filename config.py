import torch


class CFG(object):
    dataset = 'RSTPReid' # 'CUHK-PEDES', 'ICFG-PDES', 'RSTPReid'
    proj_path = '/home/honestws/codes/TextEdgeReID/'
    input_size = [256, 128]
    triplet = True
    stride_size = [16, 16]
    batch_size = 16
    num_instances = 2
    num_workers = 4
    head_lr = 1e-3
    image_embedding = 768
    text_embedding = 768
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-4
    patience = 1
    factor = 0.8
    epochs = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temperature = 1.0
    projection_dim = 256
    dropout = 0.1