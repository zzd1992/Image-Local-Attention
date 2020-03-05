import torch
from torch import nn
import random
import math
import torchvision
import torchvision.transforms as transforms
import os, argparse, copy, time
from function import LocalAttention, TorchLocalAttention

parse = argparse.ArgumentParser()
parse.add_argument('-dataset', default='/home/DATASET/CIFAR', help="dir for CIFAR")
parse.add_argument('-mode', default='torch', choices=['torch', 'our'])
parse.add_argument('-sig', default=0.1, type=float)
parse.add_argument('-batchsize', default=64, type=int)
parse.add_argument('-epoch', default=40, type=int)
parse.add_argument('-lr', default=0.001, type=float)
args = parse.parse_args()
print(args)
if not os.path.isdir("log"): os.mkdir("log")
    
    
def train(net, loader, opt):
    net.train()
    for num, batch in enumerate(loader):
        x = batch[0].cuda()
        x_noise = x + args.sig * torch.randn(x.size(), device=x.device)
        opt.zero_grad()
        preds = net(x_noise)
        loss = ((preds - x) ** 2).mean()
        loss.backward()
        opt.step()
        if num == 0:
            error = loss.item()
        else:
            error = 0.95 * error + 0.05 * loss.item()

    return error


def evaluate(net, loader):
    net.eval()
    error = 0.0
    with torch.no_grad():
        for num, batch in enumerate(loader):
            x = batch[0].cuda()
            x_noise = x + args.sig * torch.randn(x.size(), device=x.device)
            preds = net(x_noise)
            loss = ((preds - x) ** 2).mean()
            error += loss.item()
        error /= num + 1
        psnr = 10.0 * math.log(1.0 / error) / math.log(10)
        
    return error, psnr


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if args.mode == 'torch':
            self.main = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      TorchLocalAttention(64, 64, 5, 5),
                                      nn.ReLU(True),
                                      TorchLocalAttention(64, 64, 5, 5),
                                      nn.ReLU(True),
                                      TorchLocalAttention(64, 64, 5, 5),
                                      nn.ReLU(True),
                                      nn.Conv2d(64, 3, 3, padding=1, bias=False)
                                      )
        else:
            self.main = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      LocalAttention(64, 64, 5, 5),
                                      nn.ReLU(True),
                                      LocalAttention(64, 64, 5, 5),
                                      nn.ReLU(True),
                                      LocalAttention(64, 64, 5, 5),
                                      nn.ReLU(True),
                                      nn.Conv2d(64, 3, 3, padding=1, bias=False)
                                      )
        
    def forward(self, x):
        out = self.main(x)        
        return out + x

    
torch.backends.cudnn.benchmark = True

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
transform_test = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root=args.dataset, train=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root=args.dataset, train=False, transform=transform_test)  
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=3)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=3)

net = Net()
net.cuda()
params = net.parameters()
opt = torch.optim.Adam(params, args.lr)

epoch = args.epoch
for i in range(epoch):
    if i + 1 == int(epoch * 0.6):
        for param_group in opt.param_groups: param_group['lr'] /= 10
    t = time.time()
    error = train(net, trainloader, opt)
    txt = "Epoch {}:\t{:.5f}\t{:.1f}s".format(i, error, time.time()-t)
    print(txt)
    error, psnr = evaluate(net, testloader)
    txt = "{:.5f}\t{:.3f}".format(error, psnr)
    print(txt)

