from __future__ import annotations
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing
from tqdm import tqdm
import wandb

torch.multiprocessing.set_sharing_strategy("file_system")

print(f"torch version: {torch.__version__}")

from utils.helpers import make_setup
from utils.vizualise import whiten, save_result, update_debug
from utils.progress import AverageMeter, ProgressMeter
from utils.config import command_line_parser


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_log(loss, examples_seen, epoch):
    wandb.log({"epoch": epoch, "loss": loss}, step=examples_seen)
    print(f"Loss after " + str(examples_seen).zfill(5) + f" examples: {loss:.3f}")


def train_batch(images, model, optimizer, criterion, i, batch_size, save_freq):
    images = images.to(device)

    images = images.to(device)
    if images.shape[1] == 1:
        images = torch.cat([images, images, images], axis=1)

    res_images = whiten(images)
    output = model(res_images)
    output = output.to(device)
    if i % save_freq == 0:
        save_result(
            images,
            output,
            res_images,
            batch_size,
            output_dir=f"output/img_batches/batch{i}.png",
        )

    loss = criterion(output, images)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, output


def train(train_loader, model, criterion, optimizer, cfg):
    save_debug_images = "always"
    wandb.watch(model, criterion, log="all", log_freq=cfg.log_freq)
    train_loss_history = []

    for epoch in tqdm(range(cfg.epochs)):
        train_running_loss = 0.0
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4f")
        model = model.train()
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch),
        )

        examples_seen = 0
        batch_ct = 0

        end = time.time()
        for i, images in tqdm(enumerate(train_loader)):

            data_time.update(time.time() - end)

            ## training step
            if i % cfg.print_freq == 0:
                progress.display(i)

            loss, output = train_batch(
                images, model, optimizer, criterion, i, cfg.batch_size, cfg.save_freq
            )

            if loss.detach().item() > (train_running_loss / (i + 1)) * 3:
                save_debug_images = update_debug(
                    save_debug_images, images, output, loss, cfg, i
                )

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
        model = model.to(device)
        if not torch.cuda.is_available():
            print("WARNING! Training on CPU")

        ## show images
        # plot_example(train_loader, cfg.batch_size)
        train(train_loader, model, criterion, optimizer, cfg)

    return model


def main():
    cfg = command_line_parser()
    model = model_pipeline(cfg)


if __name__ == "__main__":
    main()
