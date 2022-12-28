import torch
import torch.nn
import onnx

model = Network()
model = torch.load('/home/wxz/HRank-master/result/resnet_56/trafic/pruned_checkpoint/resnet_56_cov12.pt')
model.eval()

input_names = ['input']
output_names = ['output']

x = torch.randn(1, 3, 112, 112, requires_grad=True)

torch.onnx.export(model, x, 'resnet_56_cov12.onnx', input_names=input_names, output_names=output_names, verbose='True')
