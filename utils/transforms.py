import torchvision.transforms as transforms


imagenet_transform = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(270),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

coco_transform = {
    "train": transforms.Compose(
        [
            transforms.RandomCrop(
                (270),
                pad_if_needed=True,
                padding_mode="symmetric",
            ),
            transforms.ToTensor(),
        ]
    )
}
