from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

from models.mean_color_model import MeanColorModelV2


def resolve_model_class(name):
    return {
        "mean_color_modelv2": MeanColorModelV2(),
    }[name]


def resolve_optimizer(cfg, params):
    if cfg.optimizer == "sgd":
        return SGD(
            params,
            lr=cfg.learning_rate,
            momentum=cfg.optimizer_momentum,
            weight_decay=cfg.optimizer_weight_decay,
        )
    elif cfg.optimizer == "adam":
        return Adam(
            params,
            lr=cfg.learning_rate,
            weight_decay=cfg.optimizer_weight_decay,
        )
    else:
        raise NotImplementedError


def resolve_lr_scheduler(cfg, optimizer):
    if cfg.lr_scheduler == "poly":
        return LambdaLR(
            optimizer,
            lambda ep: max(1e-6, (1 - ep / cfg.num_epochs) ** cfg.lr_scheduler_power),
        )
    else:
        raise NotImplementedError


def get_data(cfg, slice=5, train=True):
    annotations_path = (
        Path(cfg.dataset_root) / "annotations" / "instances_train2014.json"
    )
    train_path = Path(cfg.dataset_root) / "train2014"

    ## transformations
    transform = transforms.Compose(
        [
            transforms.RandomCrop(
                (cfg.aug_input_crop_size, cfg.aug_input_crop_size),
                pad_if_needed=True,
                padding_mode="symmetric",
            ),
            transforms.ToTensor(),
        ]
    )

    full_trainset = torchvision.datasets.CocoDetection(
        root=train_path, annFile=annotations_path, transform=transform
    )

    return torch.utils.data.Subset(
        full_trainset, indices=range(0, len(full_trainset), slice)
    )


def make_loader(dataset, batch_size, num_workers):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda xs: torch.stack([x[0] for x in xs]),
    )


def resolve_loss_fn(cfg):
    if cfg.loss_fn == "mse":
        return nn.MSELoss()


def make_setup(cfg):
    train = get_data(cfg)
    train_loader = make_loader(train, cfg.batch_size, cfg.num_workers)

    # model
    model = resolve_model_class(cfg.model_name)
    criterion = resolve_loss_fn(cfg)
    optimizer = resolve_optimizer(cfg, model.parameters())

    return model, train_loader, criterion, optimizer
