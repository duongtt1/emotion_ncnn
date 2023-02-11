import torch 
import torchvision 
from torchvision.models.mobilenetv2 import mobilenet_v2
import onnx 


if __name__ == "__main__":
    model = mobilenet_v2(pretrained=True)

    img = torch.zeros(1, 3, 320, 320)  # image size(1,3,320,192) iDetection
    path = "checkpoints/mobilenet_v2.onnx"
    torch.onnx.export(model, img, path, verbose=False, opset_version=12,
                            # input_names=['images'],
                        #   output_names=['classes', 'boxes'] if y is None else ['output'],
                          dynamic_axes= None)

    model_onnx = onnx.load(path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model