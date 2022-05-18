from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from models.deeplabv3p_color import DeepLabV3pColor

from models.mean_color_model import MeanColorModelV2
from utils.data_loaders import get_data, make_loader


def resolve_model_class(name):
    return {"mean_color_modelv2": MeanColorModelV2, "deeplabv3p": DeepLabV3pColor}[name]


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


def resolve_loss_fn(cfg):
    if cfg.loss_fn == "mse":
        return nn.MSELoss()


def make_setup(cfg):
    train = get_data(cfg, cfg.data_slice)
    train_loader = make_loader(train, cfg.batch_size, cfg.num_workers)

    # model
    model_cls = resolve_model_class(cfg.model_name)
    model = model_cls(cfg)

    criterion = resolve_loss_fn(cfg)
    optimizer = resolve_optimizer(cfg, model.parameters())

    return model, train_loader, criterion, optimizer
