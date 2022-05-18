from __future__ import annotations
from collections import namedtuple
from logging import Logger
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import torch
import torch.multiprocessing
from tqdm import tqdm
import wandb

torch.multiprocessing.set_sharing_strategy("file_system")

from utils.helpers import make_setup
from utils.vizualise import color_transform, whiten, save_result, update_debug
from utils.progress import AverageMeter, ProgressMeter
from utils.config import command_line_parser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SaveInfo = namedtuple("SaveInfo", ["dataset", "epoch", "i", "save_freq"])


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def train_log(loss, examples_seen, epoch):
    wandb.log({"epoch": epoch, "loss": loss}, step=examples_seen)
    print(f"Loss after " + str(examples_seen).zfill(5) + f" examples: {loss:.3f}")


def train_batch(images, model, optimizer, criterion, batch_size, save_info):
    images = images.to(device)

    images = images.to(device)
    if images.shape[1] == 1:
        images = torch.cat([images, images, images], axis=1)

    whitened = whiten(images)
    whitened_and_color_transformed = color_transform(whitened)
    output = model(whitened_and_color_transformed)
    output = output.to(device)
    if save_info.i % save_info.save_freq == 0:
        save_result(
            images,
            output,
            whitened_and_color_transformed,
            batch_size,
            output_dir=f"output/{save_info.dataset}/img_batches/",
            img_name=f"epoch{save_info.epoch}_batch{save_info.i}.png",
        )

    loss = criterion(output, images)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, output


def train(train_loader, model, criterion, optimizer, cfg):
    train_loss_history = []
    train_running_loss = 0.0
    for epoch in tqdm(range(cfg.epochs)):
        train_running_loss = 0.0
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4f")
        model = model.train()
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses],
            prefix=f"Epoch: [{epoch}]",
        )

        examples_seen = 0
        batch_ct = 0

        end = time.time()
        for i, images in tqdm(enumerate(train_loader)):

            data_time.update(time.time() - end)
            batch_size = images.shape[0]
            if i % cfg.print_freq == 0:
                progress.display(i)

            save_info = SaveInfo(cfg.dataset, epoch, i, cfg.save_freq)
            loss, output = train_batch(
                images, model, optimizer, criterion, batch_size, save_info
            )

            # if loss.detach().item() > (train_running_loss / (i + 1)) * 3:
            #     save_debug_images = update_debug(
            #         save_debug_images, images, output, loss, cfg, i
            #     )

            # measure and record loss
            losses.update(loss.item(), images.size(0))

            # log to wandb
            examples_seen += len(images)
            batch_ct += 1
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, examples_seen, epoch)

            batch_time.update(time.time() - end)

            train_running_loss += loss.detach().item()
            train_loss_history.append(loss.detach().item())

            if i % cfg.save_freq == 0:
                plt.plot(np.arange(len(train_loss_history)), train_loss_history)
                plt.xlabel("# of batches")
                plt.ylabel("loss")
                plt.savefig(f"output/progress.png")
                plt.close()

            end = time.time()

        print(f"Epoch: {epoch} | Loss: {train_running_loss}")

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "encoder_model": cfg.encoder_model_name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            True,
        )


def model_pipeline(cfg):
    hyperparams = dict(
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        dataset=cfg.dataset,
        architecture=cfg.model_name,
    )

    with wandb.init(
        "Factor Analytic Image Segmentation", entity="team23", config=hyperparams
    ):
        model, train_loader, criterion, optimizer = make_setup(cfg)
        count_parameters(model)
        model = model.to(device)
        wandb.watch(model, criterion, log="all", log_freq=cfg.log_freq)

        if not torch.cuda.is_available():
            print("WARNING! Training on CPU")

        ## show images
        # plot_example(train_loader, cfg.batch_size)
        train(train_loader, model, criterion, optimizer, cfg)

    return model


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def main():
    cfg = command_line_parser()
    model = model_pipeline(cfg)


if __name__ == "__main__":
    main()
