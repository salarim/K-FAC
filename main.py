from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim

from utils import Net, train,  validate, get_datasets
from utils import ComputeCovA, ComputeCovG 


class KFAC:

    def __init__(self, model, dataloader, batch_averaged=True):
        self.model = model
        self.dataloader = dataloader
        self.batch_averaged = batch_averaged

        self.known_modules = {'Linear', 'Conv2d'}
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.m_aa, self.m_gg = {}, {}
        self.weights = {}
        self.modules = []
        self.hook_handlers = []
        self.grads = {}
        self.avg_loss = 0.0

        self._prepare_model()

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
                self.weights[module] = weights

                count += 1

    def _save_input(self, module, input):
        aa = self.CovAHandler(input[0].data, module)
        # Initialize buffers
        if self.data_size == 0:
            self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
        self.m_aa[module] += input[0].size(0) * aa

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
        # Initialize buffers
        if self.data_size == 0:
            self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
        self.m_gg[module] += grad_output[0].size(0) * gg
    
    def _save_grad_weight(self, batch_size):
        for module in self.modules:
            grad = module.weight.grad.data
            if module.bias is not None:
                grad = torch.cat([grad, module.bias.grad.unsqueeze(1)], dim=1)
            grad = batch_size * grad
            if module not in self.grads:
                self.grads[module] = grad
            else:
                self.grads[module] += grad

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

        for module in self.modules:
            self.m_aa[module] /= self.data_size
            self.m_gg[module] /= self.data_size
            self.grads[module] /= self.data_size
        self.avg_loss /= self.data_size
        
        for handler in self.hook_handlers:
            handler.remove()

    def get_taylor_second_order_element(self):
        res = 0.0
        for module in self.modules:
            module_weight = module.weight
            if module.bias is not None:
                module_weight = torch.cat([module_weight, module.bias.unsqueeze(1)], dim=1)

            res += ((module_weight - self.weights[module]).view(1,-1) @ \
                    (self.m_gg[module] @ \
                     (module_weight - self.weights[module]) @ \
                     self.m_aa[module]).view(-1,1)).squeeze()
        return 0.5 * res

    def get_taylor_first_order_element(self):
        res = 0.0
        for module in self.modules:
            module_weight = module.weight
            if module.bias is not None:
                module_weight = torch.cat([module_weight, module.bias.unsqueeze(1)], dim=1)
            
            res += (module_weight - self.weights[module]).view(-1).dot(
                self.grads[module].view(-1)
            )

        return res

    def get_taylor_approximation(self):
        return self.avg_loss +\
               self.get_taylor_second_order_element()


def main():
    EPOCHS = 10
    tasks_nb = 10

    train_datasets, test_datasets = get_datasets(random_seed=1,
                                                  task_number=50,
                                                  batch_size_train=128,
                                                  batch_size_test=1024)
    
    model = Net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                            lr=0.01,
                            momentum=0.9,
                            weight_decay=1e-4)
    model, criterion = model.cuda(), criterion.cuda()

    models = []
    kfacs = []
    model_task_loss = np.ones((tasks_nb,tasks_nb))

    for task_id in range(tasks_nb):
        print('Task {}:'.format(task_id+1))

        for epoch in range(1, EPOCHS+1):
            train(model, train_datasets[task_id], optimizer, criterion, epoch, kfacs)

            if epoch == EPOCHS:
                for test_task_id in  range(task_id+1):
                    print('Test on task {}'.format(test_task_id+1))
                    validate(model, test_datasets[test_task_id], criterion)

        models.append(deepcopy(model))
        kfacs.append(KFAC(model, train_datasets[task_id]))
        kfacs[-1].update_stats()

        for prev_task in range(task_id):
            true_loss = validate(model, train_datasets[prev_task], criterion, log=False)[1].avg
            approx_loss = kfacs[prev_task].avg_loss +\
                        kfacs[prev_task].get_taylor_second_order_element()
            model_task_loss[task_id, prev_task] = true_loss - approx_loss
            print('Cur_task {} Prev_task {} '
                'True {:.4f} Approx {:.4f} Approx+1st {:.4f}'.format(task_id+1,
                                                    prev_task+1,
                                                    true_loss,
                                                    approx_loss,
                                                    approx_loss+kfacs[prev_task].get_taylor_first_order_element()))


if __name__=='__main__':
    main()
