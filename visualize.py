import re
import numpy as np
import matplotlib.pyplot as plt


def get_lmbd_config():
    input_files = ['acc_false_e_1_tnb_50_mnb_1/acc_false_e_1_tnb_50_mnb_1_lmbd_1e2.txt',
                    'acc_false_e_1_tnb_50_mnb_1/acc_false_e_1_tnb_50_mnb_1_lmbd_1e3.txt',
                    'acc_false_e_1_tnb_50_mnb_1/acc_false_e_1_tnb_50_mnb_1_lmbd_1e4.txt',
                    'acc_false_e_1_tnb_50_mnb_1/acc_false_e_1_tnb_50_mnb_1_lmbd_4e4.txt',
                    'acc_false_e_1_tnb_50_mnb_1/acc_false_e_1_tnb_50_mnb_1_lmbd_5e4.txt',
                    'acc_false_e_1_tnb_50_mnb_1/acc_false_e_1_tnb_50_mnb_1_lmbd_tid*1e2.txt',
                    'acc_false_e_1_tnb_50_mnb_1/acc_false_e_1_tnb_50_mnb_1_lmbd_tid*1e3.txt']
    input_files = ['outputs/'+x for x in input_files]

    legends = ['1e2',
                '1e3',
                '1e4',
                '4e4',
                '5e4',
                'task_id*1e2',
                'task_id*1e3']
    model_nbs = [1]*7

    return input_files, legends, model_nbs


def get_models_config():
    input_files = ['acc_false_e_1_tnb_50_mnb_1/acc_false_e_1_tnb_50_mnb_1_lmbd_1e4.txt',
                    'acc_false_e_1_tnb_50_mnb_5/acc_false_e_1_tnb_50_mnb_5_lmbd_1e4.txt',
                    'acc_true_e_1_tnb_50_mnb_1/acc_true_e_1_tnb_50_mnb_1_lmbd_1e2.txt',
                    'acc_true_e_1_tnb_50_mnb_5/acc_true_e_1_tnb_50_mnb_5_lmbd_1e2.txt',
                    'acc_false_e_1_tnb_50_mnb_1/acc_false_e_1_tnb_50_mnb_1_lmbd_1_ewc.txt',
                    'finetune_e_1.txt',
                    'multitask_e_1.txt']
    input_files = ['outputs/'+x for x in input_files]

    legends = ['1 model per task',
                '5 models per task',
                '1 model for all previous tasks',
                '5 models for all previous tasks',
                'EWC',
                'Finetune',
                'Multi-task']
    model_nbs = [1,
                 5,
                 1,
                 5,
                 1,
                 1,
                 1]
    
    return input_files, legends, model_nbs


def read_file(filename, task_nb, model_nb):
    train_accs = np.zeros((task_nb, model_nb))
    test_accs = 10.0 * np.ones((task_nb, model_nb, task_nb))

    model_id, task_id, test_task_id = -1, -1, -1
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            m = re.match(r"^Task (\d+) Model (\d+):$", line)
            if m:
                task_id = int(m.groups()[0]) - 1
                model_id = int(m.groups()[1]) - 1
                continue

            m = re.match(r"^Train: .+\(([0-9]+[.]?[0-9]+)\)$", line)
            if m:
                train_accs[task_id][model_id] = float(m.groups()[0])
                continue

            m = re.match(r"^Test model \d+ on task (\d+)$", line)
            if m:
                test_task_id = int(m.groups()[0]) - 1
                continue
            
            m = re.match(r"^Test: .+ \(([0-9]+[.]?[0-9]+)\)$", line)
            if m:
                acc = float(m.groups()[0])
                test_accs[task_id][model_id][test_task_id] = acc
                continue

    return train_accs, test_accs
  

def plot_avg_test_acc(filename, test_accs_dict):
    legends = []
    for legend, test_accs in test_accs_dict.items():
        task_nb, model_nb, _ = test_accs.shape

        avg_test_accs = np.zeros(task_nb)
        std_test_accs = np.zeros(task_nb)
        for i in range(task_nb):
            avg_test_accs[i] = np.mean(test_accs[i,:,0:i+1])
            std_test_accs[i] = np.mean(np.std(test_accs[i,:,0:i+1], axis=0))

        plt.errorbar(range(1,task_nb+1), avg_test_accs, yerr=std_test_accs, fmt='-')
        legends.append(legend)

    plt.legend(legends)

    plt.ylim(0,100)
    plt.xlabel("Number of observed tasks")
    plt.ylabel("Average test accuracy (%)")
    # plt.title("Average test accuracy of a kfac-approximate continual learner")
    plt.savefig(filename)
    plt.close()


def plot_bwt(filename, test_accs_dict):
    legends = []
    for legend, test_accs in test_accs_dict.items():
        task_nb, model_nb, _ = test_accs.shape
        avg_test_accs = test_accs.mean(axis=1)

        bwt = np.zeros(task_nb)
        for i in range(1,task_nb):
            for j in range(i):
                bwt[i] += (avg_test_accs[i,j] - avg_test_accs[j,j])
            bwt[i] /= i

        plt.plot(range(2,task_nb+1), bwt[1:])
        legends.append(legend)

    plt.legend(legends)

    # plt.ylim(0,100)
    plt.xlabel("Number of observed tasks")
    plt.ylabel("Average test accuracy (%)")
    # plt.title("Backward transfer of a kfac-approximate continual learner")
    plt.savefig(filename)
    plt.close()


def plot_fwt(filename, test_accs_dict, initial_accs):
    legends = []
    for legend, test_accs in test_accs_dict.items():
        task_nb, model_nb, _ = test_accs.shape
        avg_test_accs = test_accs.mean(axis=1)

        fwt = np.zeros(task_nb)
        for i in range(1,task_nb):
            for j in range(1,i+1):
                fwt[i] += (avg_test_accs[j-1,j] - initial_accs[0, j])
            fwt[i] /= i

        # if legend == '1 model for all previous tasks':
        plt.plot(range(2,task_nb+1), fwt[1:])
        legends.append(legend)

    plt.legend(legends)

    # plt.ylim(0,100)
    plt.xlabel("Number of observed tasks")
    plt.ylabel("Average test accuracy (%)")
    # plt.title("Forward transfer of a kfac-approximate continual learner")
    plt.savefig(filename)
    plt.close()


def plot_per_task_acc(filename, test_accs):
    legends = []
    task_nb, _, _ = test_accs.shape
    test_accs = test_accs.mean(axis=1)
    
    for task_id in range(task_nb):
        if task_id == 0 or (task_id+1) % 5 == 0:
            plt.plot(range(task_id+1,task_nb+1), test_accs[task_id:, task_id])
            legends.append('Task {:d}'.format(task_id+1))

    plt.legend(legends)

    plt.ylim(0,100)
    plt.xlabel("Number of observed tasks")
    plt.ylabel("Average test accuracy (%)")
    # plt.title("Average test accuracy of a kfac-approximate continual learner")
    plt.savefig(filename)
    plt.close()


def main():
    prefixes = ['lambda_', 'models_']
    config_func = [get_lmbd_config, get_models_config]

    prefix = prefixes[1]
    input_files, legends, model_nbs = config_func[1]()
    task_nb = 50
    ###############################################################################################

    initial_accs = read_file('outputs/initialization_test_accuracy.txt', task_nb, 1)[1][task_nb-1]

    test_accs_dict = {}
    for i, input_file in enumerate(input_files):
        train_accs, test_accs = read_file(input_file, task_nb, model_nbs[i])
        test_accs_dict[legends[i]] = test_accs

    plot_avg_test_acc(prefix + 'avg_acc',
                     test_accs_dict)

    plot_bwt(prefix + 'bwt', test_accs_dict)

    plot_fwt(prefix + 'fwt', test_accs_dict, initial_accs)

    # plot_per_task_acc('acc_per_task_' + 'acc_false_e_1_tnb_50_mnb_1_lmbd_1e4', test_accs_dict['1e4'])

if __name__ == '__main__':
    main()