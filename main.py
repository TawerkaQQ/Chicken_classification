import torch
import numpy as np
import torch.nn as nn
import torchvision
import imgaug.augmenters as iaa

from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from cli import parse_args
from train import train_loop


def load_model(model_name: str, num_classes: int, pretrained: bool = False):
    models = {
        'resnet18': torchvision.models.resnet18,
        'resnet34': torchvision.models.resnet34,
        'resnet50': torchvision.models.resnet50,
        'resnet101': torchvision.models.resnet101,
        'resnet152': torchvision.models.resnet152,
    }
    if pretrained:
        model = models[model_name](weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    else:
        model = models[model_name](num_classes=num_classes)
    return model


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'

    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)

    # log command line args
    with open(output_dir / 'args.txt', 'w') as f:
        f.write(args.__str__())

    device = torch.device(args.device)

    # albumentations aug
    seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1.0))),
        iaa.Sometimes(0.5, iaa.LinearContrast((0.75, 1.5))),
        iaa.AdditiveGaussianNoise(loc=0,
                                  scale=(0.0, 0.05 * 255),
                                  per_channel=0.5),
        iaa.ChangeColorTemperature((3000, 16000),
                                   from_colorspace='RGB'),
        iaa.GammaContrast((0.5, 1.75)),
        iaa.Sometimes(0.5, iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2),
                               "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)))], random_order=True)

    train_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        np.asarray,
        seq.augment_image,
        np.copy,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(args.train_dataset, train_transforms)

    val_dataset = datasets.ImageFolder(args.val_dataset, val_transforms)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    model = load_model(args.model, len(train_dataset.classes))
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_summary_writer = SummaryWriter(log_dir=log_dir / 'train')
    val_summary_writer = SummaryWriter(log_dir=log_dir / 'val')

    start_epoch = 0
    best_accuracy = 0

    if args.resume_checkpoint:
        print(f'Resume training from {args.resume_checkpoint}')
        checkpoint = torch.load(args.resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['accuracy']

    print(f'Start epoch {start_epoch}')
    train_loop(model,
               start_epoch,
               args.epochs,
               optimizer,
               device,
               criterion,
               train_summary_writer,
               val_summary_writer,
               train_dataloader,
               val_dataloader,
               checkpoint_dir,
               best_acc=best_accuracy)


if __name__ == '__main__':
    main()
