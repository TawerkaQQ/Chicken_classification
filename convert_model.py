"""
Convert model to .ptc or .onnx
"""

import torch
import torchvision
import typing
import onnx
import onnxslim

def convert_model_to_torchscript(path_to_weights: str):

    model = torchvision.models.resnet18(num_classes= 2)

    checkpoints = torch.load(path_to_weights)

    new_dict = {}
    for key, value in checkpoints['model_state_dict'].items():
        new_dict[key.replace('module.', "")] = value

    model.load_state_dict(new_dict)
    model.eval()

    example_input = torch.randn((1, 3, 224, 224), dtype=torch.float32)

    traced_cpu = torch.jit.trace(model, example_input)
    torch.jit.save(traced_cpu, "torchscript_model.ptc")


def convert_model_to_onnx(path_to_weights: str) -> None:

    model = torchvision.models.resnet18(num_classes= 2)

    checkpoints = torch.load(path_to_weights)

    new_dict = {}
    for key, value in checkpoints['model_state_dict'].items():
        new_dict[key.replace('module.', "")] = value

    model.load_state_dict(new_dict)
    model.eval()

    example_input = torch.randn((1, 3, 224, 224), dtype=torch.float32)

    f = 'model.onnx'
    torch.onnx.export(
        model,
        (example_input,),
        f,
        verbose=False,
        input_names=["tensor_1_3_224_224"],
        dynamo=True
)

    model_onnx = onnx.load(f)

    model_onnx = onnxslim.slim(model_onnx)

    onnx.save(model_onnx, f)

if __name__ == '__main__':

    convert_model_to_onnx('./checkpoints/clf_model.pt')
