import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from data import imagenet
from models import *
# from utils import progress_bar
from mask import *
import utils

from ttttt import my_Data_Set, Load_Image_Information

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument(
    '--data_dir',
    type=str,
    default='/dataset/data/data',#原来是./data
    help='dataset path')
parser.add_argument(
    '--dataset',
    type=str,
    default='trafic',#cifar10
    choices=('cifar10', 'imagenet'),
    help='dataset')
parser.add_argument(
    '--lr',
    default=0.01,
    type=float,
    help='initial learning rate')
parser.add_argument(
    '--lr_decay_step',
    default='5,10',
    type=str,
    help='learning rate decay step')
parser.add_argument(
    '--resume',
    type=str,
    default='/code/HRank-master/pretrained_models/resnet_56.pt',
    help='load the model from the specified checkpoint')
parser.add_argument(
    '--resume_mask',
    type=str,
    default=None,
    help='mask loading')
parser.add_argument(
    '--gpu',
    type=str,
    default='2',
    help='Select gpu to use')
parser.add_argument(
    '--job_dir',
    type=str,
    default='/code/HRank-master/result/resnet_56/trafic/',
    help='The directory where the summaries will be stored.')
parser.add_argument(
    '--epochs',
    type=int,
    default=15,
    help='The num of epochs to train.')
parser.add_argument(
    '--train_batch_size',
    type=int,
    default=128,
    help='Batch size for training.')
parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=100,
    help='Batch size for validation.')
parser.add_argument(
    '--start_cov',
    type=int,
    default=0,
    help='The num of conv to start prune')
parser.add_argument(
    '--compress_rate',
    type=str,
    default='[0.1]+[0.60]*35+[0.0]*2+[0.6]*6+[0.4]*3+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]',
    help='compress rate of each conv')
parser.add_argument(
    '--arch',
    type=str,
    default='resnet_56',
    choices=('resnet_50', 'vgg_16_bn', 'resnet_56', 'resnet_110', 'densenet_40', 'googlenet'),
    help='The architecture to prune')

args = parser.parse_args()

args.dataset = 'trafic'
# args.dataset='cifar10'


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if len(args.gpu) == 1:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
lr_decay_step = list(map(int, args.lr_decay_step.split(',')))

ckpt = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
utils.print_params(vars(args), print_logger.info)

# Data
print_logger.info('==> Preparing data..')

if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=2)
elif args.dataset == 'imagenet':
    data_tmp = imagenet.Data(args)
    trainloader = data_tmp.loader_train
    testloader = data_tmp.loader_test
elif args.dataset == 'trafic':
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(128),
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
    }
    # 生成Pytorch所需的DataLoader数据输入格式
    train_dataset = my_Data_Set('/dataset/data/train.txt',
                                transform=data_transforms['train'], loader=Load_Image_Information)
    test_dataset = my_Data_Set('/dataset/data/train.txt',
                               transform=data_transforms['val'], loader=Load_Image_Information)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
else:
    assert 1 == 0

if args.compress_rate:
    import re

    cprate_str = args.compress_rate
    cprate_str_list = cprate_str.split('+')
    pat_cprate = re.compile(r'\d+\.\d*')
    pat_num = re.compile(r'\*\d+')
    cprate = []
    for x in cprate_str_list:
        num = 1
        find_num = re.findall(pat_num, x)
        if find_num:
            assert len(find_num) == 1
            num = int(find_num[0].replace('*', ''))
        find_cprate = re.findall(pat_cprate, x)
        assert len(find_cprate) == 1
        cprate += [float(find_cprate[0])] * num

    compress_rate = cprate

# Model
device_ids = list(map(int, args.gpu.split(',')))
print_logger.info('==> Building model..')
net = eval(args.arch)(compress_rate=compress_rate, num_classes=89)
net = net.to(device)

if len(args.gpu) > 1 and torch.cuda.is_available():
    device_id = []
    for i in range((len(args.gpu) + 1) // 2):
        device_id.append(i)
    net = torch.nn.DataParallel(net, device_ids=device_id)

cudnn.benchmark = True
print(net)

if len(args.gpu) > 1:
    m = eval('mask_' + args.arch)(model=net, compress_rate=net.module.compress_rate, job_dir=args.job_dir,
                                  device=device)
else:
    m = eval('mask_' + args.arch)(model=net, compress_rate=net.compress_rate, job_dir=args.job_dir, device=device)

criterion = nn.CrossEntropyLoss()

dummy_input = torch.randn(1, 3, 112, 112, dtype=torch.float).to(device)
torch.onnx.export(net,
                  dummy_input,
                  'lsw.onnx',
                  verbose=True,
                  input_names=['data'],  # ['data']
                  output_names=['scores'],
                  opset_version=11)


# print(dummy_input)

# Training
def train(epoch, cov_id, optimizer, scheduler, pruning=True):
    print_logger.info('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        with torch.cuda.device(device):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            if pruning:
                m.grad_mask(cov_id)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx,len(trainloader),
            #              'Cov: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (cov_id, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch, cov_id, optimizer, scheduler):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    global best_acc
    net.eval()
    num_iterations = len(testloader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

        print_logger.info(
            'Epoch[{0}]({1}/{2}): '
            'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                epoch, batch_idx, num_iterations, top1=top1, top5=top5))

    if top1.avg > best_acc:
        print_logger.info('Saving to ' + args.arch + '_cov' + str(cov_id) + '.pt')
        state = {
            'state_dict': net.state_dict(),
            'best_prec1': top1.avg,
            'epoch': epoch,
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir(args.job_dir + '/pruned_checkpoint'):
            os.mkdir(args.job_dir + '/pruned_checkpoint')
        best_acc = top1.avg
        torch.save(state, args.job_dir + '/pruned_checkpoint/' + args.arch + '_cov' + str(cov_id) + '.pt')

    print_logger.info("=>Best accuracy {:.3f}".format(best_acc))


if len(args.gpu) > 1:
    convcfg = net.module.covcfg
else:
    convcfg = net.covcfg

param_per_cov_dic = {
    'vgg_16_bn': 4,
    'densenet_40': 3,
    'googlenet': 28,
    'resnet_50': 3,
    'resnet_56': 3,
    'resnet_110': 3
}

if len(args.gpu) > 1:
    print_logger.info('compress rate: ' + str(net.module.compress_rate))
else:
    print_logger.info('compress rate: ' + str(net.compress_rate))

for cov_id in range(args.start_cov, len(convcfg)):
    # Load pruned_checkpoint
    print_logger.info("cov-id: %d ====> Resuming from pruned_checkpoint..." % (cov_id))

    m.layer_mask(cov_id + 1, resume=args.resume_mask, param_per_cov=param_per_cov_dic[args.arch], arch=args.arch)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

    if cov_id == 0:

        pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        if args.arch == 'resnet_50':
            tmp_ckpt = pruned_checkpoint
        else:
            tmp_ckpt = pruned_checkpoint['state_dict']

        if len(args.gpu) > 1:
            for k, v in tmp_ckpt.items():
                new_state_dict['module.' + k.replace('module.', '')] = v
        else:
            for k, v in tmp_ckpt.items():
                new_state_dict[k.replace('module.', '')] = v

        # ================================================
        # new_state_dict.pop('fc.weight')
        # new_state_dict.pop('fc.bias')
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if (k in model_dict and 'fc' not in k)}
        # 更新权重
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        # =========================================
        # net.load_state_dict(new_state_dict)#
    else:
        if args.arch == 'resnet_50':
            skip_list = [1, 5, 8, 11, 15, 18, 21, 24, 28, 31, 34, 37, 40, 43, 47, 50, 53]
            if cov_id + 1 not in skip_list:
                continue
            else:
                pruned_checkpoint = torch.load(
                    args.job_dir + "/pruned_checkpoint/" + args.arch + "_cov" + str(
                        skip_list[skip_list.index(cov_id + 1) - 1]) + '.pt')
                net.load_state_dict(pruned_checkpoint['state_dict'])
        else:
            if len(args.gpu) == 1:
                pruned_checkpoint = torch.load(
                    args.job_dir + "/pruned_checkpoint/" + args.arch + "_cov" + str(cov_id) + '.pt',
                    map_location='cuda:' + args.gpu)
            else:
                pruned_checkpoint = torch.load(
                    args.job_dir + "/pruned_checkpoint/" + args.arch + "_cov" + str(cov_id) + '.pt')
            net.load_state_dict(pruned_checkpoint['state_dict'])

    best_acc = 0.
    for epoch in range(0, args.epochs):
        train(epoch, cov_id + 1, optimizer, scheduler)
        scheduler.step()
        test(epoch, cov_id + 1, optimizer, scheduler)
