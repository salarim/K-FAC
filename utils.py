import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(100, 100)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(model, dataloader, optimizer, criterion, epoch, task_id):
    losses = AverageMeter()
    top1 = AverageMeter()

    for idx, (images, targets) in enumerate(dataloader):
        model.train()

        bsz = targets.shape[0]
        images, targets = images.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss, loss_lst = criterion(outputs, targets, task_id)

        loss.backward()
        optimizer.step()
  
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # if idx % 100 == 0:
        #     print(idx, ["{:.3f}".format(x) for x in loss_lst])

        if idx == len(dataloader) - 1:
            print('Train: [{0}][{1}/{2}]\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, idx + 1, len(dataloader), loss=losses, top1=top1))
        
    return top1, losses


def validate(model, dataloader, criterion, log=True):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            bsz = targets.shape[0]
            images, targets = images.cuda(), targets.cuda()

            output = model(images)
            loss = criterion(output, targets)

            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            top1.update(acc1[0], bsz)

            if log and idx == len(dataloader) - 1:
                print('Test: [{0}/{1}]\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        idx + 1, len(dataloader), loss=losses, top1=top1))
                
    return top1, losses


def _get_dataset_MNIST(permutation, get_train=True, batch_size=64, root='./data'):
    trans_perm = transforms.Compose([transforms.ToTensor(),
              transforms.Lambda(lambda x: x.view(-1)[permutation].view(1, 28, 28))])
    
    dataset = datasets.MNIST(root=root, train=get_train, transform=trans_perm, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                         num_workers=4, pin_memory=True)
    return loader


def get_datasets(dataset_name="pMNIST", random_seed=42, task_number=10,
                 batch_size_train=64, batch_size_test=100):
    
    if dataset_name == "pMNIST":
        
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)

        np.random.seed(random_seed)
        permutations = [
            np.random.permutation(28 * 28) for
            _ in range(task_number)
        ]

        train_datasets = [
            _get_dataset_MNIST(p, True, batch_size_train, root) for p in permutations
        ]
        test_datasets = [
            _get_dataset_MNIST(p, False, batch_size_test, root) for p in permutations
        ]
    return train_datasets, test_datasets


def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()

    return x


def _extract_patches(x, kernel_size, stride, padding):
    """
    :param x: The input feature maps.  (batch_size, in_c, h, w)
    :param kernel_size: the kernel size of the conv filter (tuple of two elements)
    :param stride: the stride of conv operation  (tuple of two elements)
    :param padding: number of paddings. be a tuple of two elements
    :return: (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x


class ComputeCovA:

    @classmethod
    def compute_cov_a(cls, a, layer):
        return cls.__call__(a, layer)

    @classmethod
    def __call__(cls, a, layer):
        if isinstance(layer, nn.Linear):
            cov_a = cls.linear(a, layer)
        elif isinstance(layer, nn.Conv2d):
            cov_a = cls.conv2d(a, layer)
        else:
            # FIXME(CW): for extension to other layers.
            # raise NotImplementedError
            cov_a = None

        return cov_a

    @staticmethod
    def conv2d(a, layer):
        batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a/spatial_size
        # FIXME(CW): do we need to divide the output feature map's size?
        return a.t() @ (a / batch_size)

    @staticmethod
    def linear(a, layer):
        # a: batch_size * in_dim
        batch_size = a.size(0)
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return a.t() @ (a / batch_size)


class ComputeCovG:

    @classmethod
    def compute_cov_g(cls, g, layer, batch_averaged=False):
        """
        :param g: gradient
        :param layer: the corresponding layer
        :param batch_averaged: if the gradient is already averaged with the batch size?
        :return:
        """
        # batch_size = g.size(0)
        return cls.__call__(g, layer, batch_averaged)

    @classmethod
    def __call__(cls, g, layer, batch_averaged):
        if isinstance(layer, nn.Conv2d):
            cov_g = cls.conv2d(g, layer, batch_averaged)
        elif isinstance(layer, nn.Linear):
            cov_g = cls.linear(g, layer, batch_averaged)
        else:
            cov_g = None

        return cov_g

    @staticmethod
    def conv2d(g, layer, batch_averaged):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]
        g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))

        if batch_averaged:
            g = g * batch_size
        g = g * spatial_size
        cov_g = g.t() @ (g / g.size(0))

        return cov_g

    @staticmethod
    def linear(g, layer, batch_averaged):
        # g: batch_size * out_dim
        batch_size = g.size(0)

        if batch_averaged:
            cov_g = g.t() @ (g * batch_size)
        else:
            cov_g = g.t() @ (g / batch_size)
        return cov_g
