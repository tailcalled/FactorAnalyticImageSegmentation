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
import torchvision.transforms as transforms
from tqdm import tqdm
import wandb

from einops import rearrange, reduce, repeat

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


sample_subimage = transforms.RandomResizedCrop(224)


def train_batch(images, model, optimizer, criterion, batch_size, save_info):
    images = images.to(device)
    images_a = sample_subimage(images)
    images_b = sample_subimage(images)

    whitened_a = whiten(images_a)
    whitened_and_color_transformed_a = color_transform(whitened_a)
    whitened_b = whiten(images_b)
    whitened_and_color_transformed_b = color_transform(whitened_b)
    transformed_ab = torch.cat(
        [whitened_and_color_transformed_a, whitened_and_color_transformed_b], axis=0
    )
    original_ab = torch.cat([images_a, images_b], axis=0)
    output_color, output_loadings, output_error = model(transformed_ab)
    reduced_loadings = output_loadings[:images_a.shape[0]]
    resize = transforms.Resize(56)
    images_a = resize(images_a)
    if save_info.i % save_info.save_freq == 0:
        reduced_images = []
        connected_images = []
        for image in range(batch_size):
            reduced_loadings_image = rearrange(reduced_loadings[image], "loadings height width -> (height width) loadings")
            # a bit hacky way to check the correlations but meh
            connected_image = reduced_loadings_image[(reduced_loadings.shape[2] + 1) * reduced_loadings.shape[3] // 2]
            connected_image = torch.sum(connected_image * reduced_loadings_image, axis=1)
            connected_image = repeat(connected_image, "(height width) -> colors height width", colors=3, height=reduced_loadings.shape[2])
            connected_images.append(connected_image)
            U, S, V = torch.pca_lowrank(reduced_loadings_image, 3)
            reduced_loadings_image = rearrange(U, "(height width) colors -> colors height width", height=output_loadings.shape[2], width=output_loadings.shape[3])
            reduced_images.append(reduced_loadings_image)
        reduced_loadings = torch.stack(reduced_images)
        connected_images = torch.stack(connected_images)
        save_result(
            images_a,
            output_color[: images_a.shape[0]],
            resize(whitened_and_color_transformed_a),
            reduced_loadings,
            connected_images,
            batch_size,
            output_dir=f"output/{save_info.dataset}/img_batches/",
            img_name=f"epoch{save_info.epoch}_batch{save_info.i}.png",
        )
    output_color = rearrange(
        output_color,
        "(layer batch) color height width -> batch height width (layer color)",
        layer=2,
    )
    output_loadings = rearrange(
        output_loadings,
        "(layer batch) (color loadings) height width -> batch height width (layer color) loadings",
        layer=2,
        color=3,
    )
    output_error = torch.broadcast_to(
        output_error, (2 * images_a.shape[0], 3, images_a.shape[2], images_a.shape[3])
    )
    output_error = rearrange(
        output_error,
        "(layer batch) color height width -> batch height width (layer color)",
        layer=2,
        color=3,
    )
    original_ab = rearrange(
        original_ab,
        "(layer batch) color height width -> batch height width (layer color)",
        layer=2,
    )

    original_ab = resize(rearrange(original_ab, "b h w c -> b c h w"))
    original_ab = rearrange(original_ab, "b c h w -> b h w c")
    output_correlations = output_loadings @ output_loadings.transpose(3, 4)
    output_correlations = output_correlations + torch.diag_embed(output_error)
    output_precision = torch.inverse(output_correlations)
    logdet = torch.logdet(2 * 3.14 * output_precision)
    difference = output_color - original_ab

    per_pixel_loss = (
        0.5
        * torch.einsum("bhwc,bhwck,bhwk->bhw", difference, output_precision, difference)
        - logdet / 2
    )
    loss = torch.mean(per_pixel_loss)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


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
            loss = train_batch(
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
