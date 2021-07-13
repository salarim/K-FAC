import pickle
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from utils import set_seed, Net
from utils import Net, train, validate, get_datasets

from main import KFAC


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
            rand_dirs = {}
            rand_norms = {}

            for name, weight in self.weights.items():
                rand_dir = torch.randn(weight.shape).to(weight.device)
                if len(weight.shape) == 1:
                    rand_norm = weight.unsqueeze(dim=1)
                if len(weight.shape) == 2:
                    rand_norm = torch.norm(rand_dir, dim=1).unsqueeze(dim=1)

                rand_dirs[name] = rand_dir
                rand_norms[name] = rand_norm
            
            rand_norms_layer = {}
            for name, norm in rand_norms.items():
                layer, tp = name.split('.')
                if layer not in rand_norms_layer:
                    rand_norms_layer[layer] = norm**2
                else:
                    rand_norms_layer[layer] += norm**2

            for name, rand_dir in rand_dirs.items():
                layer, tp = name.split('.')
                norm = torch.sqrt(rand_norms_layer[layer])
                if tp == 'weight':
                    rand_dirs[name] = rand_dirs[name] / norm
                elif tp == 'bias':
                    rand_dirs[name] = rand_dirs[name] / norm.squeeze()

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
            z = losses #[:-1, :-1]

            fig, ax = plt.subplots()

            # c = ax.pcolormesh(x, y, z, cmap='OrRd', vmin=0.0, vmax=max_loss)
            c = ax.contourf(x, y, z, 20, vmin=0.0, vmax=max_loss)
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


def get_vis_loss_func_for_kfac(kfac, lmbd):
    def loss_func(model):
        return lmbd * kfac.get_taylor_approximation(model)

    return loss_func


def visualize_loss_space(datasets, models_dict, kfacs, max_dis, steps, lmbd, loss_vis_mod):
    criterion = torch.nn.CrossEntropyLoss()
    loss_vis = LossVisualizer(max_dis, steps)
    loss_vis_kfac = LossVisualizer(max_dis, steps)
    loss_vis.set_model(next(iter(models_dict.values())))
    rand_dirs_lst = loss_vis.get_rand_dirs()

    for model_name, model in models_dict.items():
        task_id, model_id = [int(x) for x in model_name.split('-')]

        for dataset_id, dataset in enumerate(datasets):
            if (task_id == 0 or (task_id+1) % loss_vis_mod == 0) and \
                (dataset_id == 0 or (dataset_id+1) % loss_vis_mod == 0):
                print('MODEL LOSS VISUALIZATION Source: {:d} Dest: {:d}'.format(task_id+1, dataset_id+1))

                file_name = 'heatmaps/s{:d}_d{:d}'.format(task_id+1, dataset_id+1)
                loss_func = get_vis_loss_func(dataset, criterion)
                loss_vis.set_loss_func(loss_func)
                loss_vis.set_model(model)

                loss_vis.add_losses(rand_dirs_lst, file_name)

    for model_name, model in models_dict.items():
        task_id, model_id = [int(x) for x in model_name.split('-')]

        for kfac_id, kfac in enumerate(kfacs):
            if (task_id == 0 or (task_id+1) % loss_vis_mod == 0) and \
                (kfac_id == 0 or (kfac_id+1) % loss_vis_mod == 0):
                print('KFAC LOSS VISUALIZATION Source: {:d} Dest: {:d}'.format(task_id+1, kfac_id+1))

                file_name = 'heatmaps/kfac_s{:d}_d{:d}'.format(task_id+1, kfac_id+1)
                loss_vis_kfac.set_loss_func(get_vis_loss_func_for_kfac(kfacs[kfac_id][-1], lmbd))
                loss_vis_kfac.set_model(model)

                loss_vis_kfac.add_losses(rand_dirs_lst, file_name)

    loss_vis.visualize()
    loss_vis_kfac.visualize()


def load_files(tasks_nb):
    with open('perms/perms.pkl', 'rb') as input:
        perms = pickle.load(input)
    train_datasets, test_datasets = get_datasets(task_number=tasks_nb,
                                                batch_size_train=128,
                                                batch_size_test=4096,
                                                permutations=perms)

    kfacs = []
    all_models = {}

    for i in range(tasks_nb):
        model_name = '{:d}-0'.format(i)
        model = Net().cuda()
        model.load_state_dict(torch.load('models/{:s}.pt'.format(model_name)))
        all_models[model_name] = model

        with open('kfacs/{:d}_weights.pkl'.format(i), 'rb') as input:
            weights = pickle.load(input)
        with open('kfacs/{:d}_maa.pkl'.format(i), 'rb') as input:
            m_aa = pickle.load(input)
        with open('kfacs/{:d}_mgg.pkl'.format(i), 'rb') as input:
            m_gg = pickle.load(input)
        
        kfac = KFAC(model, train_datasets[i], False)
        kfac.weights = weights
        kfac.m_aa = m_aa
        kfac.m_gg = m_gg
        kfacs.append([kfac])

    return train_datasets, test_datasets, kfacs, all_models


if __name__ == '__main__':
    tasks_nb = 50
    lmbd = 10**4
    max_dis = 1
    steps = 10 + 1
    loss_vis_mod = 10
    seed = 1234

    set_seed(seed)

    train_datasets, test_datasets, kfacs, all_models = load_files(tasks_nb)

    visualize_loss_space(test_datasets, all_models, kfacs, max_dis, steps, lmbd, loss_vis_mod)

    # loss_vis_kfac = LossVisualizer(max_dis, steps)
    # loss_vis_kfac.set_loss_func(get_vis_loss_func_for_kfac(kfacs[9][-1], lmbd))
    # loss_vis_kfac.set_model(all_models['0-0'])
    # rand_dirs_lst = loss_vis_kfac.get_rand_dirs()

    # loss_vis_kfac.add_losses(rand_dirs_lst, 'heatmap')
    # loss_vis_kfac.visualize()