from multiprocessing.sharedctypes import Value
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    res_img = image - blurred_img
    print(torch.cat([res_img, image], axis=1).shape)

    deblur_img = transforms.GaussianBlur(99, 10)(res_img)
    deblur_img_norm = transforms.GaussianBlur(99, 10)(torch.abs(res_img))
    deblur_img = deblur_img / deblur_img_norm

    res_img = (res_img + 1) / 2

    deblur_img = deblur_img + res_img
    deblur_img = (deblur_img - torch.min(deblur_img)) / (
        torch.max(deblur_img) - torch.min(deblur_img)
    )

    print(res_img[0].shape, blurred_img[0].shape, image[0].shape)

    ## show images
    # imshow(torchvision.utils.make_grid(torch.cat([deblur_img, res_img, image])))
    imshow(
        torchvision.utils.make_grid(
            torch.cat(
                [
                    torch.stack([res_img[i], blurred_img[i], image[i], image[i]])
                    for i in range(batch_size)
                ]
            )
        )
    )


def whiten(image):
    blur = transforms.GaussianBlur(7, 7)
    return (image - blur(image) + 1) / 2


def get_image_grid(image, output, res_img, cfg):
    out = output
    diff = image - out
    diff = diff + 0.5 if cfg.save_debug_imgs else diff / 2 + 0.5
    return torchvision.utils.make_grid(
        torch.cat(
            [
                torch.stack([diff[i], image[i], res_img[i], out[i]]).detach()
                for i in range(cfg.batch_size)
            ]
        )
    )

def save_preds(image, output, res_img, loss, i, cfg):

    img_grid = get_image_grid(
        output.type(torch.uint8),
        image.type(torch.uint8),
        res_img.type(torch.uint8),
        cfg,
    )
    output_dir = f"output/debug_imgbatches/debug{i % cfg.storage}.png"
    plt.imshow(showable(img_grid.type(torch.uint8)))
    plt.title(f"{i} {np.round(loss.detach().item(), 2)}")
    plt.savefig(output_dir)
    plt.close()


def show_progress(image, pred, i, cfg):
    res_img = whiten(image)
    diff = image - pred

    print((diff + 0.5).shape, image.shape, res_img.shape, pred.shape)
    img_grid = get_image_grid(image, pred, res_img, cfg)
    plt.savefig(f"output/imgbatches/batch{i}.png")
    plt.imshow(showable(img_grid))
    plt.close()


def save_result(img, output, res_img, batch_size, output_dir=None):
    if not output_dir:
        raise ValueError("output directory has not been set!")

    img = img.to(device)
    output = output.to(device)
    diff = img - output
    res_img = res_img.to(device)
    diff = diff.to(device)
    print((img + 0.5).shape, img.shape, res_img.shape, output.shape)
    plt.imshow(
        showable(
            torchvision.utils.make_grid(
                torch.cat(
                    [
                        torch.stack(
                            [(diff + 0.5)[i], img[i], res_img[i], output[i]]
                        ).detach()
                        for i in range(batch_size)
                    ]
                )
            )
        )
    )
    # plt.imshow(torchvision.utils.make_grid(showable(torch.cat([diff + 0.5, image, res_img, out]).detach()).cpu()))

    plt.savefig(output_dir)
    plt.close()

def update_debug(save_debug_images, images, output, loss, cfg, i):
    print("WARNING! Loss increased a lot")
    save_result(
        images,
        output,
        whiten(images),
        cfg.batch_size,
        output_dir=f"output/debug_img_batches/debug{i % cfg.storage}.png",
    )

    if save_debug_images == "always":
        if loss.detach().item() > 3:
            save_debug_images = cfg.storage // 2
    else:
        save_debug_images -= 1

    return save_debug_images
