import io
import torch
import torch.onnx
from models.resnet_cifar import ResNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test():
    model = ResNet

    pthfile = r'/home/wxz/HRank-master/result/resnet_56/trafic/pruned_checkpoint/resnet_56_cov12.pt'
    loaded_model = torch.load(pthfile, map_location='cpu')
    # try:
    #   loaded_model.eval()
    # except AttributeError as error:
    #   print(error)

    model.load_state_dict(loaded_model['state_dict'])
    # model = model.to(device)

    # data type nchw
    dummy_input1 = torch.randn(1, 3, 112, 112)
    # dummy_input2 = torch.randn(1, 3, 64, 64)
    # dummy_input3 = torch.randn(1, 3, 64, 64)
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    # torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
    torch.onnx.export(model, dummy_input1, "resnet56.onnx", verbose=True, input_names=input_names,
                      output_names=output_names)


if __name__ == "__main__":
    test()