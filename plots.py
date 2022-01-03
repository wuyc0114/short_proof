import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def plot_cor(n, d, methods=['optimal', 'Bayes AMP', 'Prox-linear', '1 step prox-linear', 'TAF', 'Gradient descent']):
    fig, ax = plt.subplots()
    x = np.arange(11)
    dir = os.getcwd() + '/results/'
    for method in methods:
        if method == 'optimal':
            df = pd.read_csv(dir + f'optimal_n={n}_d={d}_lr=0.1_num_exp=50.csv', header=None)
            y = df.mean(axis=0)
            ax.plot(x, y, marker='o', label='Optimal (theory)', ls = '--', markerfacecolor="None", markeredgecolor='red', color='r')

        if method == 'Bayes AMP':
            df = pd.read_csv(dir + f'AMP_n={n}_d={d}_lr=0.1_num_exp=50.csv', header=None)
            y = df.mean(axis=0)
            ax.plot(x, y, marker='|', label=method, markerfacecolor="None", markeredgecolor='k', color='k', markersize=14)

        if method == 'Prox-linear':
            df = pd.read_csv(dir + f'prox_linear_n={n}_d={d}_lr=0.1_num_exp=50.csv', header=None)
            y = df.mean(axis=0)
            ax.plot(x, y, marker='^', label=method, markerfacecolor="None", markeredgecolor='b', color='b')

        if method == '1 step prox-linear':
            df = pd.read_csv(dir + f'prox_linear_GD_n={n}_d={d}_lr=0.1_num_exp=50.csv', header=None)
            y = df.mean(axis=0)
            ax.plot(x, y, marker='s', label=method+ r'$, \xi$='+f'0.1', markerfacecolor="None", markeredgecolor='darkgreen', color='darkgreen', lw=2)

        if method == '1 step prox-linear':
            df = pd.read_csv(dir + f'prox_linear_GD_n={n}_d={d}_lr=0.05_num_exp=50.csv', header=None)
            y = df.mean(axis=0)
            ax.plot(x, y, marker='s', label=method+ r'$, \xi$='+f'0.05', ls = '--', markerfacecolor="None", markeredgecolor='darkgreen', color='darkgreen')

        if method == 'TAF':
            df = pd.read_csv(dir + f'TAF_n={n}_d={d}_lr=0.1_num_exp=50.csv', header=None)
            y = df.mean(axis=0)
            ax.plot(x, y, marker='p', label=method, markerfacecolor="None", markeredgecolor='m', color='m')

        if method == 'Gradient descent':
            df = pd.read_csv(dir + f'GD_n={n}_d={d}_lr=2_num_exp=50.csv', header=None)
            y = df.mean(axis=0)
            ax.plot(x, y, marker='v', label=method + r'$, \eta$='+f'2', markerfacecolor="None", markeredgecolor='dimgrey', color='dimgrey', lw = 2)

        if method == 'Gradient descent':
            df = pd.read_csv(dir + f'GD_n={n}_d={d}_lr=0.5_num_exp=50.csv', header=None)
            y = df.mean(axis=0)
            ax.plot(x, y, marker='v', label=method + r'$, \eta$='+f'0.5', markerfacecolor="None", markeredgecolor='dimgrey', color='dimgrey', ls = '--')


    plt.legend(loc='upper left', fontsize=8)
    ax.set_xlabel(r'$t$', fontsize = 12)
    ax.set_ylabel(r'$\frac{|\langle \mathbf{\theta}, \mathbf{\theta}_t\rangle|}{\Vert\mathbf{\theta}\Vert_2 \Vert\mathbf{\theta}_t\Vert_2}$', fontsize=9)
    plt.savefig(os.getcwd() + f'/figures/n={n}_d={d}.pdf')


def plot_step(n, d):
    fig, ax = plt.subplots()
    dir = os.getcwd() + '/results/'
    df = pd.read_csv(dir + f'step_optimal_n={n}_d={d}_num_exp=50.csv')
    ax.plot(df.iloc[:44, 0], df.iloc[:44, 1], label='Gradient descent', color='k', marker='o', markersize=3,
            markerfacecolor="None", markeredgecolor='k')
    ax.plot(df.iloc[:44, 0], df.iloc[:44, 2], label='1 step prox-linear', color='b', marker='s', markersize=3,
            markerfacecolor="None", markeredgecolor='b')
    if n == 1000:
        ax.axhline(y=0.99997331394118, color='r', linestyle='--')
        plt.text(-1.4, 0.988, 'Optimal (theory)', fontsize=10)
        plt.text(-2.2, 0.957, '1 step prox-linear', fontsize=10)
        plt.text(-2.2, 0.883, 'Gradient descent', fontsize=10)
    if n == 600:
        ax.axhline(y=0.974408018373367, color='r', linestyle='--')
        plt.text(-1.4, 0.93, 'Optimal (theory)', fontsize=10)
        plt.text(-2, 0.775, '1 step prox-linear', fontsize=10)
        plt.text(-2, 0.63, 'Gradient descent', fontsize=10)

    plt.xlabel(r'$\log \xi$ ( $ \log \eta )$')
    plt.ylabel('Correlation')
    plt.show()
    plt.savefig(os.getcwd() + f'/figures/step_n={n}_d={d}.pdf')
    return




