import os
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from utils import set_seed
from utils import Net, train,  validate, get_datasets
from utils import ComputeCovA, ComputeCovG


class KFAC:

    def __init__(self, model, dataloader, batch_averaged=False):
        self.model = model
        self.dataloader = dataloader
        self.batch_averaged = batch_averaged

        self.known_modules = {'Linear', 'Conv2d'}
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()

        self.hook_handlers = []
        self.modules = []
        self.module_to_id = {}

        self.weights = []
        self.avg_loss = 0.0

        self._prepare_model()

        self.m_aa = [0.0 for i in range(len(self.modules))]
        self.m_gg = [0.0 for i in range(len(self.modules))]
        self.grads = []

    def _prepare_model(self):
        count = 0
        for module in self.model.modules():
            classname = module.__class__.__name__
            
            if classname in self.known_modules:
                self.modules.append(module)
                handler1 = module.register_forward_pre_hook(self._save_input)
                handler2 = module.register_backward_hook(self._save_grad_output)
                self.hook_handlers.append(handler1)
                self.hook_handlers.append(handler2)

                weights = module.weight.data
                if module.bias is not None:
                    weights = torch.cat([weights, module.bias.data.unsqueeze(1)], dim=1)
                self.weights.append(weights)
                self.module_to_id[module] = count

                count += 1

    def _save_input(self, module, input):
        if module.__class__.__name__ not in self.known_modules:
            return
        
        aa = self.CovAHandler(input[0].data, module)
        # Initialize buffers
        module_id = self.module_to_id[module]
        self.m_aa[module_id] += input[0].size(0) * aa

    def _save_grad_output(self, module, grad_input, grad_output):
        if module.__class__.__name__ not in self.known_modules:
            return
        
        # Accumulate statistics for Fisher matrices
        gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
        # Initialize buffers
        module_id = self.module_to_id[module]
        self.m_gg[module_id] += grad_output[0].size(0) * gg
    
    def _save_grad_weight(self, batch_size):
        for module_id, module in enumerate(self.modules):
            if module.__class__.__name__ not in self.known_modules:
                continue
        
            grad = module.weight.grad.data
            if module.bias is not None:
                grad = torch.cat([grad, module.bias.grad.unsqueeze(1)], dim=1)
            grad = batch_size * grad
            if module_id > len(self.grads)-1:
                self.grads.append(grad)
            else:
                self.grads[module_id] += grad

    def update_stats(self):
        self.data_size = 0
        criterion = torch.nn.CrossEntropyLoss()

        for idx, (images, targets) in enumerate(self.dataloader):
            batch_size = images.size(0)
            images, targets = images.cuda(), targets.cuda()

            self.model.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, targets)
            loss.backward()

            self._save_grad_weight(batch_size)
            self.avg_loss += batch_size * loss.item()

            self.data_size += batch_size

        for module_id in range(len(self.modules)):
            self.m_aa[module_id] /= self.data_size
            self.m_gg[module_id] /= self.data_size
            self.grads[module_id] /= self.data_size
        self.avg_loss /= self.data_size
        
        for handler in self.hook_handlers:
            handler.remove()

    def get_taylor_second_order_element(self, model):
        res = 0.0
        module_id = 0
        for module in model.modules():
            # print(module, self.modules, module_id)
            classname = module.__class__.__name__

            if classname in self.known_modules:
                module_weight = module.weight
                if module.bias is not None:
                    module_weight = torch.cat([module_weight, module.bias.unsqueeze(1)], dim=1)

                res += ((module_weight - self.weights[module_id]).view(1,-1) @ \
                        (self.m_gg[module_id] @ \
                        (module_weight - self.weights[module_id]) @ \
                        self.m_aa[module_id]).view(-1,1)).squeeze()
                module_id += 1
        
        return 0.5 * res

    def get_taylor_first_order_element(self, model):
        res = 0.0
        module_id = 0
        for module in model.modules():
            classname = module.__class__.__name__
            
            if classname in self.known_modules:
                module_weight = module.weight
                if module.bias is not None:
                    module_weight = torch.cat([module_weight, module.bias.unsqueeze(1)], dim=1)
                
                res += (module_weight - self.weights[module_id]).view(-1).dot(
                    self.grads[module_id].view(-1)
                )
            
                module_id += 1

        return res

    def get_taylor_approximation(self, model):
        return self.get_taylor_second_order_element(model)

    def visualize_attr(self, path, kfac_id, attr):
        im_dir = os.path.join(path, attr + '_' + str(kfac_id))
        if not os.path.exists(im_dir):
            os.mkdir(im_dir)

        for module_id in range(len(self.modules)):
            if attr == 'gg':
                arr = self.m_gg[module_id]
            elif attr == 'aa':
                arr = self.m_aa[module_id]

            arr = arr.cpu().numpy()
            arr_min, arr_max = np.min(arr), np.max(arr)
            
            eigvals = np.linalg.eigvals(arr)
            # arr = arr - arr_min
            # arr = arr / (arr_max - arr_min)
            print("{} {}\tmin: {:.3f}\tmax: {:.3f}\tmin_eigval: {:.3e}\tmax_eigval: {:.3e}".format(attr, module_id, np.min(arr), np.max(arr), np.min(eigvals), np.max(eigvals)))
            im_path = os.path.join(im_dir, 'm_' + str(module_id) + '.jpg')
            plt.imsave(im_path, arr, cmap='Greys')

    def get_weights_distance(self, model):
        dis = 0.0
        module_id = 0
        for module in model.modules():
            classname = module.__class__.__name__

            if classname in self.known_modules:
                module_weight = module.weight
                if module.bias is not None:
                    module_weight = torch.cat([module_weight, module.bias.unsqueeze(1)], dim=1)

                dis += torch.norm(module_weight - self.weights[module_id])
                module_id += 1

        return dis

def create_loss_function(kfacs, model, accumulate_last_kfac, lmbd):
    cross_entorpy = torch.nn.CrossEntropyLoss()

    def get_loss(outputs, targets, task_id):
        loss_lst = []
        loss = 0.0

        if len(kfacs) > 0:
            if accumulate_last_kfac:
                kfacs_in_use = [kfacs[-1]]
            else:
                kfacs_in_use = kfacs

            for task_kfacs in kfacs_in_use:
                closest_kfac = None
                closest_kfac_dis = float('inf')

                for model_id, model_kfac in enumerate(task_kfacs):
                    dis = model_kfac.get_weights_distance(model)
                    if dis <  closest_kfac_dis:
                        closest_kfac_dis = dis
                        closest_kfac = model_kfac

                task_kfac_loss = closest_kfac.get_taylor_approximation(model)
                
                task_kfac_loss *= lmbd
                loss_lst.append(task_kfac_loss)
                loss += task_kfac_loss

        last_loss = cross_entorpy(outputs, targets)
        loss_lst.append(last_loss)
        loss += last_loss

        return loss, loss_lst
    
    return get_loss


def main():
    EPOCHS = 1
    tasks_nb = 50
    models_nb_per_task = 1
    accumulate_last_kfac = False
    lmbd = 10**4
    seed = 1234

    set_seed(seed)
    train_datasets, test_datasets = get_datasets(task_number=tasks_nb,
                                                  batch_size_train=128,
                                                  batch_size_test=4096)
    
    models = [Net().cuda() for i in range(models_nb_per_task)]
    optimizers = [optim.SGD(model.parameters(),
                            lr=0.01,
                            momentum=0.9,
                            weight_decay=1e-4) for model in models]

    kfacs = []
    train_criterion = [create_loss_function(kfacs, model, accumulate_last_kfac, lmbd) for model in models]
    test_criterion = torch.nn.CrossEntropyLoss()
    val_accs = [[0.0]*tasks_nb for _ in range(tasks_nb)]

    for task_id in range(tasks_nb):
        task_kfacs = []

        for model_id, model in enumerate(models):
            print('Task {} Model {}:'.format(task_id+1, model_id+1))

            for epoch in range(1, EPOCHS+1):
                train(model, train_datasets[task_id], optimizers[model_id], train_criterion[model_id], epoch, task_id+1)

            for test_task_id in  range(tasks_nb):
                print('Test model {} on task {}'.format(model_id+1, test_task_id+1), flush=True)
                val_acc = validate(model, test_datasets[test_task_id], test_criterion)[0].avg.item()

                prev_acc = val_accs[task_id][test_task_id] * model_id
                val_accs[task_id][test_task_id] = (prev_acc + val_acc) / (model_id+1)

            task_kfacs.append(KFAC(model, train_datasets[task_id]))
            task_kfacs[-1].update_stats()
        
        kfacs.append(task_kfacs)

        if accumulate_last_kfac and len(kfacs) > 1:
            for model_kfac_id in range(len(kfacs[-1])):
                for module_id in range(len(kfacs[-1][model_kfac_id].modules)):
                    kfacs[-1][model_kfac_id].m_aa[module_id] += kfacs[-2][model_kfac_id].m_aa[module_id]
                    kfacs[-1][model_kfac_id].m_gg[module_id] += kfacs[-2][model_kfac_id].m_gg[module_id]

        kfacs[-1][-1].visualize_attr('images/', task_id, 'gg')
        kfacs[-1][-1].visualize_attr('images/', task_id, 'aa')

        print('#'*60, 'Avg acc: {:.2f}'.format(np.sum(val_accs[task_id])/(task_id+1)))


if __name__=='__main__':
    main()
