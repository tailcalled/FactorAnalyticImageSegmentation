from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchtyping import TensorType

from einops import rearrange, reduce, repeat

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import wandb

## functions to show an image
def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def showable(img):
    print(img.cpu().numpy().shape)
    return np.transpose(img.cpu().numpy(), (1, 2, 0))


def plot_example(trainloader, batch_size):
    dataiter = iter(trainloader)
    image = dataiter.next()
    # image, labels = dataiter.next()
    print(image)
    print(image.shape)
    print(image[0, 0, 0, 0])

    blurred_img = transforms.GaussianBlur(15, 7)(image)
    whitened_img = image - blurred_img
    print(torch.cat([whitened_img, image], axis=1).shape)

    deblur_img = transforms.GaussianBlur(99, 10)(whitened_img)
    deblur_img_norm = transforms.GaussianBlur(99, 10)(torch.abs(whitened_img))
    deblur_img = deblur_img / deblur_img_norm

    whitened_img = (whitened_img + 1) / 2

    deblur_img = deblur_img + whitened_img
    deblur_img = (deblur_img - torch.min(deblur_img)) / (
        torch.max(deblur_img) - torch.min(deblur_img)
    )

    print(whitened_img[0].shape, blurred_img[0].shape, image[0].shape)

    ## show images
    # imshow(torchvision.utils.make_grid(torch.cat([deblur_img, whitened_img, image])))
    imshow(
        torchvision.utils.make_grid(
            torch.cat(
                [
                    torch.stack([whitened_img[i], blurred_img[i], image[i], image[i]])
                    for i in range(batch_size)
                ]
            )
        )
    )


def whiten(
    images: TensorType["batch_size", "channels", "H", "W"]
) -> TensorType["batch_size", "channels", "H", "W"]:
    blur = transforms.GaussianBlur(7, 7)
    return images - blur(images)


def color_transform(
    images: TensorType["batch_size", "channels", "H", "W"]
) -> TensorType["batch_size", "channels", "H", "W"]:
    r, g, b = torch.split(images, [1, 1, 1], dim=1)
    x = r + g + b
    y = r**2 + g**2 + b**2
    z = r * g + r * b + g * b
    imgs_color_transformed = torch.concat([x, y, z], dim=1)
    return (imgs_color_transformed + 3) / 6


def get_image_grid(image, output, whitened_img, cfg):
    out = output
    diff = image - out
    diff = diff + 0.5 if cfg.save_debug_imgs else diff / 2 + 0.5
    return torchvision.utils.make_grid(
        torch.cat(
            [
                torch.stack([diff[i], image[i], whitened_img[i], out[i]]).detach()
                for i in range(cfg.batch_size)
            ]
        )
    )


def save_result(
    img,
    output,
    whitened_and_color_transformed,
    pca_result,
    connected_images,
    batch_size,
    output_dir=None,
    img_name=None,
):

    if not output_dir:
        raise ValueError("output directory has not been set!")

    Path(output_dir).mkdir(parents=True, exist_ok=True) 
    img = img.to(device)
    output = output.to(device)
    diff = img - output
    whitened_and_color_transformed = whitened_and_color_transformed.to(device)
    diff = diff.to(device)
    pca_result = pca_result.to(device)
    print(
        (img + 0.5).shape, img.shape, whitened_and_color_transformed.shape, output.shape
    )
    print(pca_result.shape)
    lower = reduce(pca_result, "batch colors width height -> colors", "min")
    upper = reduce(pca_result, "batch colors width height -> colors", "max")
    lower = repeat(lower, "colors -> batch colors width height", batch=pca_result.shape[0], width=pca_result.shape[2], height=pca_result.shape[3])
    upper = repeat(upper, "colors -> batch colors width height", batch=pca_result.shape[0], width=pca_result.shape[2], height=pca_result.shape[3])
    pca_result = (pca_result - lower)/(upper-lower)
    print(pca_result.shape)

    lower = torch.min(connected_images)
    upper = torch.max(connected_images)
    connected_images = (connected_images - lower)/(upper-lower)

    images = showable(
        torchvision.utils.make_grid(
            torch.cat(
                [
                    torch.stack(
                        [
                            (diff + 0.5)[i],
                            img[i],
                            # whitened_img[i],
                            whitened_and_color_transformed[i],
                            output[i],
                            pca_result[i],
                            connected_images[i]
                        ]
                    ).detach()
                    for i in range(batch_size)
                ]
            ),
            nrow=12
        )
    )
    print(f"images: {images.shape}")
    images = plt.imshow(images)
    wandb.log(
        {
            "img": [
                wandb.Image(images, caption="blurred | orig | net_input | prediction")
            ]
        }
    )
    # plt.imshow(torchvision.utils.make_grid(showable(torch.cat([diff + 0.5, image, whitened_img, out]).detach()).cpu()))

    plt.savefig(Path(output_dir) / img_name)
    plt.close()


def update_debug(save_debug_images, images, output, loss, cfg, i):
    print("WARNING! Loss increased a lot")
    save_result(
        images,
        output,
        whiten(images),
        cfg.batch_size,
        output_dir="output/debug_img_batches",
        img_name=f"debug{i % cfg.storage}.png",
    )

    if save_debug_images == "always":
        if loss.detach().item() > 3:
            save_debug_images = cfg.storage // 2
    else:
        save_debug_images -= 1

    return save_debug_images
