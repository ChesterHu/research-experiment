import os
import matplotlib.pyplot as plt
import numpy as np

def read_values(fname):
    with open(fname) as file:
        return np.array(file.read().split(','), dtype = np.float)

def plot_function_values(values, label, xlabel='Iteration', ylabel = 'Function value'):
    x_axis = [i+1 for i in range(len(values))]
    plt.plot(x_axis, values, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

if __name__ == "__main__":
    file_name_fun_val = 'fun_values_rand.txt'
    file_name_fun_val_accel = 'fun_values_rand_accel.txt'

    file_name_cand_record = 'candidates_records.txt'

    full_path = os.path.realpath(__file__)
    dir_name = os.path.dirname(full_path)
    fig_name_fun_val = f'{dir_name}/../figures/function_values.png'
    fig_name_cand_record = f'{dir_name}/../figures/candidates_record.png'

    values = read_values(f'{dir_name}/../output/{file_name_fun_val}')
    plot_function_values(values, 'non-accelerated method')
    values = read_values(f'{dir_name}/../output/{file_name_fun_val_accel}')
    plot_function_values(values, 'accelerated method')

    plt.legend()
    plt.savefig(fig_name_fun_val)
    plt.close()
    
    values = read_values(f'{dir_name}/../output/{file_name_cand_record}')
    plot_function_values(values, '', xlabel='Itreration', ylabel='Number of candidates')
    plt.savefig(fig_name_cand_record)