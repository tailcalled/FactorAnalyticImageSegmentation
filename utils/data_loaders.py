from pathlib import Path

import torch
import torchvision

from utils.transforms import imagenet_transform, coco_transform


def get_imagenet_data(cfg):
    train_folder = Path(cfg.dataset_root) / "train"
    print(train_folder)
    return torchvision.datasets.ImageFolder(
        train_folder, transform=imagenet_transform["train"]
    )


def get_coco_data(cfg):
    annotations_path = (
        Path(cfg.dataset_root) / "annotations" / "instances_train2014.json"
    )
    train_path = Path(cfg.dataset_root) / "train2014"

    return torchvision.datasets.CocoDetection(
        root=train_path, annFile=annotations_path, transform=coco_transform["train"]
    )


def get_data(cfg, slice=5, train=True):
    if cfg.dataset == "coco":
        train = get_coco_data(cfg)
        return torch.utils.data.Subset(train, indices=range(0, len(train), slice))

    elif cfg.dataset == "imagenet":
        train = get_imagenet_data(cfg)
        return torch.utils.data.Subset(train, indices=range(0, len(train), slice))


def make_loader(dataset, batch_size, num_workers):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda xs: torch.stack([x[0] for x in xs]),
    )
