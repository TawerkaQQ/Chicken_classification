import cv2
import torch
import pathlib
import os
import torchvision

from torchvision import datasets, transforms
from tqdm import tqdm

train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]
)


def apply_model(model, path_to_data: str):
    acc_model = 0
    all_photo = len(test_loader.dataset)

    with torch.no_grad():

        for image in tqdm(os.listdir(path_to_data)):
            inputs = cv2.imread(f'{path_to_data}/{image}')
            inputs = cv2.resize(inputs, (224, 224))
            inputs = train_transforms(inputs)

            outputs_latest_model = model(inputs.unsqueeze(0))
            _, predicted_latest_model = torch.max(outputs_latest_model, 1)

            if predicted_latest_model == 0:
                print('man')
            else:
                print('woman')

            acc_model += (predicted_latest_model == 0).sum().item()

    acc_model = acc_model / all_photo

    print(f'Accuracy for the latest model: {acc_model:.4f}')

    return None


def validate_and_load_model(model_path: pathlib.Path):
    device = torch.get_default_device()
    model = torchvision.models.resnet18(num_classes=2)
    checkpoints = torch.load(model_path)

    new_dict = {}
    for key, value in checkpoints['model_state_dict'].items():
        new_dict[key.replace('module.', "")] = value

    model.load_state_dict(new_dict)
    model.to(device)
    model.eval()

    return model


# def test_model(path_to_model: str, images_dir: str, outpur_dir: str) -> None:
#     image_names = os.listdir(images_dir)
#     model = validate_and_load_model(path_to_model)
#
#     for img_name in tqdm(image_names, desc="Testing images"):
#         if img_name.endswith((".png", ".jpg", ".jpeg")):
#             collected_img = apply_model()

#     return None


if __name__ == "__main__":
    test_dataset = datasets.ImageFolder("data/test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader.dataset.transform = train_transforms

    model = validate_and_load_model('./checkpoints/latest_model.pt')
    best_model = validate_and_load_model(
        './checkpoints/clf_model.pt'
        )

    apply_model(model, best_model, test_loader)
