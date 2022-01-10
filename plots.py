import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import matplotlib


def plot_cor(n, d, methods=['optimal', 'Bayes AMP', 'Prox-linear', '1 step prox-linear', 'TAF', 'Gradient descent']):
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$t$', fontsize=16)
    ax.set_ylabel(
        r'$\frac{|\langle \mathbf{\theta}, \mathbf{\theta}^t\rangle|}{\Vert\mathbf{\theta}\Vert_2 \Vert\mathbf{\theta}^t\Vert_2}$',
        fontsize=16)
    colors = [matplotlib.cm.viridis(25 * x) for x in range(3, 20)]
    x = np.arange(11)
    dir = os.getcwd() + '/results/'
    for method in methods:
        if method == 'optimal':
            df = pd.read_csv(dir + f'optimal_n={n}_d={d}_lr=0.1_num_exp=50.csv', header=None)
            y = df.mean(axis=0)
            ax.plot(x, y, marker='o', label='Optimal (theory)', ls = '--', markerfacecolor="None", markeredgecolor='red', color='r', lw=2, markersize=7)

        if method == 'Bayes AMP':
            df = pd.read_csv(dir + f'AMP_n={n}_d={d}_lr=0.1_num_exp=50.csv', header=None)
            y = df.mean(axis=0)
            ax.plot(x, y, marker='|', label=method, markerfacecolor="None", markersize=14, markeredgecolor=colors[0], color=colors[0], lw=2)

        if method == 'Prox-linear':
            df = pd.read_csv(dir + f'prox_linear_n={n}_d={d}_lr=0.1_num_exp=50.csv', header=None)
            y = df.mean(axis=0)
            ax.plot(x, y, marker='^', label=method, markerfacecolor="None", markeredgecolor='k', color='k', lw=2, markersize=7)

        if method == '1 step prox-linear':
            df = pd.read_csv(dir + f'prox_linear_GD_n={n}_d={d}_lr=0.1_num_exp=50.csv', header=None)
            y = df.mean(axis=0)
            ax.plot(x, y, marker='s', label=method+ r'$, \xi$='+f'0.1', markerfacecolor="None", lw=2, markeredgecolor=colors[1], color=colors[1], markersize=7)

        if method == '1 step prox-linear':
            df = pd.read_csv(dir + f'prox_linear_GD_n={n}_d={d}_lr=0.05_num_exp=50.csv', header=None)
            y = df.mean(axis=0)
            ax.plot(x, y, marker='s', label=method+ r'$, \xi$='+f'0.05', ls = '--', markerfacecolor="None", markeredgecolor=colors[2], color=colors[2], lw=2, markersize=7)

        if method == 'TAF':
            df = pd.read_csv(dir + f'TAF_n={n}_d={d}_lr=0.1_num_exp=50.csv', header=None)
            y = df.mean(axis=0)
            ax.plot(x, y, marker='p', label=method, markerfacecolor="None", markeredgecolor=colors[3], color=colors[3], lw=2, markersize=9)

        if method == 'Gradient descent':
            df = pd.read_csv(dir + f'GD_n={n}_d={d}_lr=2_num_exp=50.csv', header=None)
            y = df.mean(axis=0)
            ax.plot(x, y, marker='v', label=method + r'$, \eta$='+f'2', markerfacecolor="None", lw=2, markeredgecolor=colors[4], color=colors[4], markersize=7)

        if method == 'Gradient descent':
            df = pd.read_csv(dir + f'GD_n={n}_d={d}_lr=0.5_num_exp=50.csv', header=None)
            y = df.mean(axis=0)
            ax.plot(x, y, marker='v', label=method + r'$, \eta$='+f'0.5', markerfacecolor="None", ls='--', markeredgecolor=colors[5], color=colors[5], lw=2, markersize=7)

    #plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
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
        plt.text(-1.53, 0.988, 'Optimal (theory)', fontsize=14)
        plt.text(-2.2, 0.957, '1 step prox-linear', fontsize=14)
        plt.text(-2.2, 0.888, 'Gradient descent', fontsize=14)
    if n == 600:
        ax.axhline(y=0.974408018373367, color='r', linestyle='--')
        plt.text(-1.53, 0.93, 'Optimal (theory)', fontsize=14)
        plt.text(-2, 0.775, '1 step prox-linear', fontsize=14)
        plt.text(-2, 0.63, 'Gradient descent', fontsize=14)

    plt.xlabel(r'$\log \xi$ ( $ \log \eta )$',  fontsize=14)
    plt.ylabel('Correlation',  fontsize=14)
    plt.show()
    plt.savefig(os.getcwd() + f'/figures/step_n={n}_d={d}.pdf')
    return




