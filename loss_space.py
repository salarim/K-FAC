
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from utils import set_seed
from utils import Net, train, validate, get_datasets


class LossVisualizer:

    def __init__(self, model, loss_func):
        self.model = model
        self.loss_func = loss_func
        self.weights = {}
        max_dis = 100
        steps = 50 + 1

        self.coefs = np.meshgrid(np.linspace(-max_dis, max_dis, steps), np.linspace(-max_dis, max_dis, steps))
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.weights[name] = param.data

    def get_rand_dirs(self):
        rand_dirs_lst = []

        for i in range(2):
            dir_len = 0.0
            rand_dirs = {}

            for name, weight in self.weights.items():
                rand_dir = torch.randn(weight.shape).to(weight.device)
                dir_len += torch.sum(rand_dir**2)
                rand_dirs[name] = rand_dir

            dir_len = dir_len**0.5
            for name, rand_dir in rand_dirs.items():
                rand_dir /= dir_len

            rand_dirs_lst.append(rand_dirs)

        return rand_dirs_lst

    def get_losses(self, rand_dirs_lst):
        losses = np.zeros(self.coefs[0].shape)

        for i in range(self.coefs[0].shape[0]):
            for j in range(self.coefs[0].shape[1]):
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                param.data = self.weights[name] + \
                                            self.coefs[0][i,j] * rand_dirs_lst[0][name] + \
                                            self.coefs[1][i,j] * rand_dirs_lst[1][name]

                    losses[i,j] = self.loss_func(self.model)
                    print("{:.3f}\t{:.3f}\t{:.3f}".format(self.coefs[0][i,j], self.coefs[1][i,j], losses[i,j]))

        return losses

    def visualize(self, losses):
        x,y = self.coefs
        z = losses[:-1, :-1]
        z_min, z_max = -np.abs(z).max(), np.abs(z).max()

        fig, ax = plt.subplots()

        c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=0.0, vmax=z_max)
        ax.set_title('pcolormesh')
        # set the limits of the plot to the limits of the data
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        fig.colorbar(c, ax=ax)

        plt.savefig('heatmap')
        plt.close()

if __name__ == '__main__':
    tasks_nb = 1
    seed = 1234

    # set_seed(seed)
    train_datasets, test_datasets = get_datasets(task_number=tasks_nb,
                                                  batch_size_train=128,
                                                  batch_size_test=4096)
    test_criterion = torch.nn.CrossEntropyLoss()

    def loss_func(model):
        return validate(model, test_datasets[0], test_criterion)[1].avg

    def train_criterion(outputs, targets, task_id):
        loss = test_criterion(outputs, targets)
        return loss, [loss]

    model = Net().cuda()
    optimizer = optim.SGD(model.parameters(),
                        lr=0.01,
                        momentum=0.9,
                        weight_decay=1e-4)

    train(model, train_datasets[0], optimizer, train_criterion, 1, 1)
    print("original loss:", loss_func(model))
    loss_vis = LossVisualizer(model, loss_func)

    rand_dirs_lst = loss_vis.get_rand_dirs()
    losses = loss_vis.get_losses(rand_dirs_lst)
    loss_vis.visualize(losses)


    