import torch
import os.path as osp
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sampler import RandomIdentitySampler
from PIL import Image

def collate_fn(batch):
    imgs, pids, captions = zip(*batch)
    _captions = []
    for c in captions:
        _captions += c
    return torch.cat(imgs, dim=0), torch.cat(pids, dim=0), _captions

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageTextDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, captions = self.dataset[index]
        img = read_image(img_path)
        pids = torch.tensor([pid] * len(captions), dtype=torch.int64)
        imgs = torch.cat([self.transform(img).unsqueeze(0) for _ in range(len(captions))], dim=0)

        return [imgs, pids, captions]


class CUHKPEDES(object):
    """
    The CUHK-PEDES dataset is a caption-annotated pedestrian dataset.
    It contains 40,206 images over 13,003 persons.
    Images are collected from five existing person re-identification datasets,
    CUHK03, Market-1501, SSM, VIPER, and CUHK01 while each image is annotated with 2 text descriptions
    by crowd-sourcing workers. Sentences incorporate rich details about person appearances, actions, poses.
    """

    def __init__(self, train_list, test_list):
        super(CUHKPEDES, self).__init__()
        self.train = self._process_dir(train_list, relabel=True)
        self.test = self._process_dir(test_list)

    def _process_dir(self, dictionary, relabel=False):
        dataset = []
        pid_container = set()
        for item in dictionary:
            pid_container.add(item['id'])
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if relabel: self.number_cls = len(pid_container)

        for item in dictionary:
            pid = item['id']
            if relabel: pid = pid2label[item['id']]
            dataset.append((item['file_path'], pid, item['captions']))
        return dataset


class ICFGPDES(object):
    """
    Compared with existing databases, ICFG-PEDES has three key advantages.
    First, its textual descriptions are identity-centric and fine-grained.
    Second, the images included in ICFG-PEDES are more challenging,
    containing more appearance variability due to the presence of complex backgrounds and variable illumination.
    Third, the scale of ICFG-PEDES is larger.
    """

    def __init__(self, train_list, test_list):
        super(ICFGPDES, self).__init__()
        self.train = self._process_dir(train_list, relabel=True)
        self.test = self._process_dir(test_list)

    def _process_dir(self, dictionary, relabel=False):
        dataset = []
        pid_container = set()
        for item in dictionary:
            pid_container.add(item['id'])
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if relabel: self.number_cls = len(pid_container)

        for item in dictionary:
            pid = item['id']
            if relabel: pid = pid2label[item['id']]
            dataset.append((item['file_path'], pid, item['captions']))
        return dataset


class RSTPReid(object):
    """
    RSTPReid contains 20505 images of 4,101 persons from 15 cameras.
    Each person has 5 corresponding images taken by different cameras
    with complex both indoor and outdoor scene transformations and backgrounds
    in various periods of time, which makes RSTPReid much more challenging and
    more adaptable to real scenarios. Each image is annotated with 2 textual descriptions.
    For data division, 3701 (index < 18505), 200 (18505 <= index < 19505) and 200 (index >= 19505)
    identities are utilized for training, validation and testing, respectively
    (Marked by item 'split' in the JSON file). Each sentence is no shorter than 23 words.
    """

    def __init__(self, train_list, test_list):
        super(RSTPReid, self).__init__()
        self.train = self._process_dir(train_list, relabel=True)
        self.test = self._process_dir(test_list)

    def _process_dir(self, dictionary, relabel=False):
        dataset = []
        pid_container = set()
        for item in dictionary:
            pid_container.add(item['id'])
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if relabel: self.number_cls = len(pid_container)

        for item in dictionary:
            pid = item['id']
            if relabel: pid = pid2label[item['id']]
            dataset.append((item['img_path'], pid, item['captions']))
        return dataset

__factory = {
    'CUHK-PEDES': CUHKPEDES,
    'ICFG-PDES': ICFGPDES,
    'RSTPReid': RSTPReid
}


def create_dataloader(CFG, train_list, test_list, triplet_transform):
    dataset = __factory[CFG.dataset](train_list, test_list)
    tri_train_set = ImageTextDataset(dataset.train, triplet_transform)
    triplet_train_loader = DataLoader(
        tri_train_set, batch_size=CFG.batch_size,
        sampler=RandomIdentitySampler(dataset.train, CFG.batch_size, CFG.num_instances),
        num_workers=CFG.num_workers, collate_fn=collate_fn
    )
    setattr(triplet_train_loader, 'number_cls', dataset.number_cls)
    plain_transform = T.Compose([
        T.Resize([256, 128], interpolation=3),
        T.RandomHorizontalFlip(),
        T.Pad(10),
        T.RandomCrop([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    plain_train_set = ImageTextDataset(dataset.train, plain_transform)
    plain_train_loader = DataLoader(plain_train_set, batch_size=CFG.batch_size,
                                    shuffle=True, num_workers=CFG.num_workers,
                                    collate_fn=collate_fn)
    setattr(plain_train_loader, 'number_cls', dataset.number_cls)
    val_set = ImageTextDataset(dataset.test, triplet_transform)
    test_loader = DataLoader(
        val_set, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, collate_fn=collate_fn
    )

    return triplet_train_loader, plain_train_loader, test_loader