import os
import pickle
import random
import matplotlib.pyplot as plt

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


def get_incorrect_indexes(output, target):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        incorrect = pred.ne(target.view(1, -1).expand_as(pred))

        return incorrect.nonzero(as_tuple=True)[1]


def visualize_incorrect_samples(dataset, model):
    with torch.no_grad():
        index = 0
        for idx, (images, targets) in enumerate(dataset):
            images, targets = images.cuda(), targets.cuda() 
            output = model(images)
            incorrect_images = images[get_incorrect_indexes(output, targets)]
            for i in range(incorrect_images.shape[0]):
                plt.imshow( incorrect_images[i].permute(1, 2, 0).cpu(), cmap='gray')
                plt.savefig('incorrect-outputs/{:d}'.format(index))
                index += 1
                plt.close()


def get_min_max_grad(model):
    max_grad = float('-inf')
    min_grad = float('inf')

    for module in model.children():
        classname = module.__class__.__name__
        if classname in {'Linear', 'Conv2d'}:
            grad = module.weight.grad.data
            if module.bias is not None:
                grad = torch.cat([grad, module.bias.grad.unsqueeze(1)], dim=1)
            
            if torch.max(grad) > max_grad:
                max_grad = torch.max(grad)
            if torch.min(grad) < min_grad:
                min_grad = torch.min(grad)
        
    return min_grad, max_grad


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

        if idx % 10 == 0:
            print(idx, ["{:.3f}".format(x) for x in loss_lst])
            # min_grad, max_grad = get_min_max_grad(model)
            # print('MIN GRAD: {:.3f}\tMAX GRAD: {:.3f}'.format(min_grad, max_grad))

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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def _get_dataset_permuted_MNIST(permutation, get_train=True, root='./data'):
    trans_perm = transforms.Compose([transforms.ToTensor(),
              transforms.Lambda(lambda x: x.view(-1)[permutation].view(1, 28, 28))])
    
    dataset = datasets.MNIST(root=root, train=get_train, transform=trans_perm, download=True)
    return dataset


def _get_dataset_rotated_MNIST(degree, get_train=True, root='./data'):
    trans_perm = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomRotation(degrees=(degree,degree))])

    dataset = datasets.MNIST(root=root, train=get_train, transform=trans_perm, download=True)
    return dataset


def get_datasets(dataset_name="pMNIST", task_number=10,
                 batch_size_train=64, batch_size_test=100, include_prev=False, saved_data=None, seed=1234):
    
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)
        
    if dataset_name == "pMNIST":
        _get_dataset = _get_dataset_permuted_MNIST
        if saved_data is None:
            saved_data = [
                np.random.permutation(28 * 28) for
                _ in range(task_number)
            ]

            with open('perms/{:s}-{:d}.pkl'.format(dataset_name, seed), 'wb') as output:
                pickle.dump(saved_data, output, pickle.HIGHEST_PROTOCOL)
    elif dataset_name == "rMNIST":
        _get_dataset = _get_dataset_rotated_MNIST
        if saved_data is None:
            degs = np.linspace(0, 360, task_number+1)
            saved_data = [x + np.random.rand() * (y-x) for x,y in zip(degs, degs[1:])]

            with open('perms/{:s}-{:d}.pkl'.format(dataset_name, seed), 'wb') as output:
                pickle.dump(saved_data, output, pickle.HIGHEST_PROTOCOL)

    train_datasets = [
        _get_dataset(p, True, root) for p in saved_data
    ]
    test_datasets = [
        _get_dataset(p, False, root) for p in saved_data
    ]

    if include_prev:
        new_train_datasets = []
        new_test_datasets = []

        for i in range(len(train_datasets)):
            new_train_datasets.append(torch.utils.data.ConcatDataset(train_datasets[:i+1]))
            new_test_datasets.append(torch.utils.data.ConcatDataset(test_datasets[:i+1]))

        train_datasets = new_train_datasets
        test_datasets = new_test_datasets
    
    train_loaders, test_loaders = [], []
    for train_dataset in train_datasets:
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True,
                                        num_workers=4, pin_memory=True)
        train_loaders.append(loader)

    for test_dataset in test_datasets:
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True,
                                        num_workers=4, pin_memory=True)
        test_loaders.append(loader)
            
    return train_loaders, test_loaders


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
