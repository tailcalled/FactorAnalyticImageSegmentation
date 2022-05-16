import argparse
import os
import json


def expandpath(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def command_line_parser():
    parser = argparse.ArgumentParser(
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset_root", type=expandpath, required=True, help="Path to dataset"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        help="Name for your run to easier identify it.",
    )
    parser.add_argument(
        "--log_dir",
        type=expandpath,
        required=False,
        help="Place for artifacts and logs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint, which can also be an AWS link s3://...",
    )

    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of samples in a batch for training",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers in a batch for training",
    )

    parser.add_argument(
        "--batch_size_validation",
        type=int,
        default=8,
        help="Number of samples in a batch for validation",
    )

    parser.add_argument(
        "--aug_input_crop_size", type=int, default=100, help="Training crop size"
    )
    parser.add_argument(
        "--aug_geom_scale_min",
        type=float,
        default=1.0,
        help="Augmentation: lower bound of scale",
    )
    parser.add_argument(
        "--aug_geom_scale_max",
        type=float,
        default=1.0,
        help="Augmentation: upper bound of scale",
    )
    parser.add_argument(
        "--aug_geom_tilt_max_deg",
        type=float,
        default=0.0,
        help="Augmentation: maximum rotation degree",
    )
    parser.add_argument(
        "--aug_geom_wiggle_max_ratio",
        type=float,
        default=0.0,
        help="Augmentation: perspective warping level between 0 and 1",
    )
    parser.add_argument(
        "--aug_geom_reflect",
        type=str2bool,
        default=False,
        help="Augmentation: Random horizontal flips",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "adam"],
        help="Type of optimizer",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate at start of training",
    )

    parser.add_argument(
        "--optimizer_momentum", type=float, default=0.9, help="Optimizer momentum"
    )
    parser.add_argument(
        "--optimizer_weight_decay",
        type=float,
        default=0.0001,
        help="Optimizer weight decay",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="poly",
        choices=["poly"],
        help="Type of learning rate scheduler",
    )
    parser.add_argument(
        "--lr_scheduler_power", type=float, default=0.9, help="Poly learning rate power"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="COCO",
        choices=["COCO"],
        help="Dataset name",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="mean_color_modelv2",
        choices=["mean_color_modelv2"],
        help="model",
    )
    parser.add_argument(
        "--save_debug_imgs",
        type=str2bool,
        default=True,
        help="If we should store images for debuggin purposes",
    )
    parser.add_argument(
        "--storage", type=int, default=20, help="Number of debug images to store."
    )

    parser.add_argument(
        "--print_freq", type=int, default=10, help="How often to print progress bar"
    )

    parser.add_argument(
        "--log_freq", type=int, default=10, help="How often to log wandb"
    )

    parser.add_argument(
        "--save_freq", type=int, default=100, help="How often to log wandb"
    )

    parser.add_argument(
        "--loss_fn", type=str, default="mse", help="Which loss function to use"
    )

    cfg = parser.parse_args()

    print(json.dumps(cfg.__dict__, indent=4, sort_keys=True))

    return cfg
