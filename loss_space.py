
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from utils import set_seed
from utils import Net, train, validate, get_datasets


class LossVisualizer:

    def __init__(self, max_dis, steps):
        self.model = None
        self.loss_func = None
        self.weights = {}
        self.losses_dict = {}

        self.coefs = np.meshgrid(np.linspace(-max_dis, max_dis, steps), np.linspace(-max_dis, max_dis, steps))

    def set_model(self, model):
        self.model = deepcopy(model)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.weights[name] = param.data

    def set_loss_func(self, loss_func):
        self.loss_func = loss_func

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

    def add_losses(self, rand_dirs_lst, filename):
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

        self.losses_dict[filename] = losses

    def visualize(self):
        max_loss = 0.0
        for filename, losses in self.losses_dict.items():
            max_loss = max(max_loss, np.abs(losses[:-1, :-1]).max())
        x,y = self.coefs

        for filename, losses in self.losses_dict.items():
            z = losses[:-1, :-1]

            fig, ax = plt.subplots()

            c = ax.pcolormesh(x, y, z, cmap='OrRd', vmin=0.0, vmax=max_loss)
            # set the limits of the plot to the limits of the data
            ax.axis([x.min(), x.max(), y.min(), y.max()])
            ax.set_xlabel("coef 1")
            ax.set_ylabel("coef 2")

            fig.colorbar(c, ax=ax)

            plt.savefig(filename)
            plt.close()


def get_vis_loss_func(dataset, criterion):
    def loss_func(model):
        return validate(model, dataset, criterion)[1].avg

    return loss_func


if __name__ == '__main__':
    tasks_nb = 2
    max_dis = 10
    steps = 10 + 1
    seed = 1234

    # set_seed(seed)
    train_datasets, test_datasets = get_datasets(task_number=tasks_nb,
                                                  batch_size_train=128,
                                                  batch_size_test=4096)
    test_criterion = torch.nn.CrossEntropyLoss()

    def train_criterion(outputs, targets, task_id):
        loss = test_criterion(outputs, targets)
        return loss, [loss]

    model = Net().cuda()
    optimizer = optim.SGD(model.parameters(),
                        lr=0.01,
                        momentum=0.9,
                        weight_decay=1e-4)

    train(model, train_datasets[0], optimizer, train_criterion, 1, 1)

    loss_func = get_vis_loss_func(test_datasets[0], test_criterion)
    print("original loss: {:.3f}".format(loss_func(model)))
    loss_vis = LossVisualizer(max_dis, steps)
    loss_vis.set_model(model)
    loss_vis.set_loss_func(loss_func)

    rand_dirs_lst = loss_vis.get_rand_dirs()
    loss_vis.add_losses(rand_dirs_lst, 'heatmap')
    loss_vis.visualize()


    