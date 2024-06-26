import argparse

from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50',
                                 'resnet101', 'resnet152'],
                        default='resnet50')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'])
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--train-dataset', type=Path, required=True)
    parser.add_argument('--val-dataset', type=Path, required=True)
    parser.add_argument('--test-dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--resume-checkpoint', type=Path)
    return parser.parse_args()
