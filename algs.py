# Numerical experiments for solving noiseless phase retrieval

import numpy as np
from scipy import optimize
import cvxpy as cp
import pandas as pd
import os
import time
from importlib import reload

class Simulation:
    def __init__(self, n=600, d=400, iter=10, delta_tilde=1.001, noise_level=0, N=100000, num_exp=50, lr1=0.1, lr2=0.1,
                 verbose=True, save=True):
        self.n = n
        self.d = d
        self.delta = n / d
        self.lr1 = lr1
        self.lr2 = lr2
        self.num_exp = num_exp
        self.delta_tilde = delta_tilde
        self.iter = iter
        self.noise_level = noise_level
        self.verbose = verbose
        self.save = save
        self.G = np.random.normal(size=N)
        self.W = np.random.normal(size=N)
        self.Y = self.G**2 + np.random.normal(scale=self.noise_level, size=N)
        self.get_lambda_bar()
        self.get_lambda_star()
        self.result_AMP = []
        self.result_GD = []
        self.result_prox_linear = []
        self.result_prox_linear_GD = []
        self.result_TAF = []
        self.result_optimal = []
        self.time_AMP = np.zeros(self.num_exp)
        self.time_GD = np.zeros(self.num_exp)
        self.time_prox_linear = np.zeros(self.num_exp)
        self.time_prox_linear_GD = np.zeros(self.num_exp)
        self.time_TAF = np.zeros(self.num_exp)

    def step(self):
        loglr_vec = np.arange(-3, 3.01, 0.05)
        self.step_results = pd.DataFrame(index=loglr_vec, columns=['Gradient descent', '1 step gradient descent'])
        for loglr in loglr_vec:
            print(f'Running experiment with log lr = {loglr}')
            self.lr = np.exp(loglr)
            self.result_GD = []
            self.result_prox_linear_GD = []
            for t in range(self.num_exp):
                self.X_GD = np.zeros((self.d, self.iter + 1))
                self.X_prox_linear_GD = np.zeros((self.d, self.iter + 1))
                self.cor_GD = np.zeros(self.iter + 1)
                self.cor_prox_linear_GD = np.zeros(self.iter + 1)
                self.get_data()
                self.gradient_descent()
                self.prox_linear(exact=False, GD=True)
                self.result_GD.append(self.cor_GD)
                self.result_prox_linear_GD.append(self.cor_prox_linear_GD)
            self.step_results.loc[loglr, 'Gradient descent'] = np.mean(self.result_GD)
            self.step_results.loc[loglr, '1 step gradient descent'] = np.mean(self.result_prox_linear_GD)
        if self.save:
            dir = os.getcwd() + '/results'
            self.step_results.to_csv(
                dir + f'/step_optimal_n={self.n}_d={self.d}_num_exp={self.num_exp}.csv')






    def run(self):
        for t in range(self.num_exp):
            if self.verbose:
                print(f'Running experiment {t}...')
            self.X_AMP = np.zeros((self.d, self.iter + 1))
            self.U_AMP = np.zeros((self.n, self.iter + 1))
            self.X_GD = np.zeros((self.d, self.iter + 1))
            self.X_prox_linear = np.zeros((self.d, self.iter + 1))
            self.X_prox_linear_GD = np.zeros((self.d, self.iter + 1))
            self.X_TAF = np.zeros((self.d, self.iter + 1))
            self.cor_AMP = np.zeros(self.iter + 1)
            self.cor_GD = np.zeros(self.iter + 1)
            self.cor_prox_linear = np.zeros(self.iter + 1)
            self.cor_prox_linear_GD = np.zeros(self.iter + 1)
            self.cor_TAF = np.zeros(self.iter + 1)
            self.cor_optimal = np.zeros(self.iter + 1)
            self.SE_mu_X = np.zeros(self.iter + 1)
            self.SE_sigma2_X = np.zeros(self.iter + 1)
            self.SE_mu_U = np.zeros(self.iter + 1)
            self.SE_sigma2_U = np.zeros(self.iter + 1)
            self.rho = np.zeros(self.iter + 1)
            self.get_data()
            self.get_a()
            self.get_SE()
            t1 = time.time()
            self.AMP()
            self.time_AMP[t] = time.time() - t1

            self.lr = self.lr1
            t1 = time.time()
            self.gradient_descent()
            self.time_GD[t] = time.time() - t1

            self.lr = self.lr2
            t1 = time.time()
            self.prox_linear(exact=True, GD=False)
            self.time_prox_linear[t] = time.time() - t1
            t1 = time.time()
            self.prox_linear(exact=False, GD=True)
            self.time_prox_linear_GD[t] = time.time() - t1
            t1 = time.time()
            self.TAF()
            self.time_TAF[t] = time.time() - t1
            self.result_AMP.append(self.cor_AMP)
            self.result_GD.append(self.cor_GD)
            self.result_prox_linear.append(self.cor_prox_linear)
            self.result_prox_linear_GD.append(self.cor_prox_linear_GD)
            self.result_TAF.append(self.cor_TAF)
            self.result_optimal.append(self.cor_optimal)


        if self.save:
            dir = os.getcwd() + '/results'
            pd.DataFrame(self.result_AMP).to_csv(
                dir + f'/AMP_n={self.n}_d={self.d}_lr={self.lr}_num_exp={self.num_exp}.csv', header=None, index=None)
            pd.DataFrame(self.result_GD).to_csv(
                dir + f'/GD_n={self.n}_d={self.d}_lr={self.lr}_num_exp={self.num_exp}.csv', header=None, index=None)
            pd.DataFrame(self.result_prox_linear).to_csv(
                dir + f'/prox_linear_n={self.n}_d={self.d}_lr={self.lr}_num_exp={self.num_exp}.csv', header=None,
                index=None)
            pd.DataFrame(self.result_prox_linear_GD).to_csv(
                dir + f'/prox_linear_GD_n={self.n}_d={self.d}_lr={self.lr}_num_exp={self.num_exp}.csv', header=None,
                index=None)
            pd.DataFrame(self.result_TAF).to_csv(
                dir + f'/TAF_n={self.n}_d={self.d}_lr={self.lr}_num_exp={self.num_exp}.csv', header=None, index=None)
            pd.DataFrame(self.result_optimal).to_csv(
                dir + f'/optimal_n={self.n}_d={self.d}_lr={self.lr}_num_exp={self.num_exp}.csv', header=None, index=None)


    def get_data(self):
        self.A = np.random.normal(size=(self.n, self.d)) / np.sqrt(self.d)
        self.x = np.random.normal(size=self.d)
        self.y = (self.A @ self.x)**2 + np.random.normal(size=self.n, scale=self.noise_level)
        self.get_xs()
        self.L = 2 * np.linalg.norm(self.A, ord=2)**2 / self.delta
        return

    #-----Spectral Initialization----------------------------------------
    def Gamma(self, y):
        # Eq.(137) in [MM19]  https://arxiv.org/pdf/1708.05932.pdf
        return (y - 1) / (y + np.sqrt(self.delta_tilde) - 1)

    def get_xs(self):
        self.Zs = np.diag(self.Gamma(self.y))
        D = self.A.T @ self.Zs @ self.A
        w, v = np.linalg.eigh(D)
        self.xs =  v[:, np.argmax(w)]
        return

    # -----Bayes AMP-----------------------------------------------------
    # http://proceedings.mlr.press/v130/mondelli21a.html

    def get_a(self):
        enum = 1 / self.delta - np.mean(self.Gamma(self.Y) ** 2 / (self.lambda_star - self.Gamma(self.Y)) ** 2)
        denom = 1 / self.delta + np.mean(
            self.Gamma(self.Y) ** 2 * (self.G ** 2 - 1) / (self.lambda_star - self.Gamma(self.Y)) ** 2)
        self.a = np.sqrt(enum / denom)
        return


    def phi(self, l):
        return l * np.mean(self.Gamma(self.Y) * self.G**2 / (l - self.Gamma(self.Y)))

    def psi(self, l):
        return l / self.delta + l * np.mean(self.Gamma(self.Y) / (l - self.Gamma(self.Y)))

    def get_lambda_bar(self):
        self.lambda_bar = optimize.fminbound(self.psi, 1, 100)
        return

    def zeta(self, l):
        return self.psi(max(l, self.lambda_bar))

    def get_lambda_star(self):
        self.lambda_star = optimize.brentq(lambda x: self.zeta(x) - self.phi(x), 1, 100)
        return

    def h(self, u, y, sigma2, mu):
        return (sigma2 + mu**2) / sigma2 * np.sqrt(np.abs(y)) * np.tanh(
            mu / sigma2 * u * np.sqrt(np.abs(y))) - mu / sigma2 * u

    def h_derivative(self, u, y, sigma2, mu):
        return (sigma2 + mu**2) / sigma2 * np.sqrt(np.abs(y)) * (
                    1 - np.tanh(mu / sigma2 * u * np.sqrt(np.abs(y)))**2) * mu / sigma2 * np.sqrt(
            np.abs(y)) - mu / sigma2

    def get_SE(self):
        self.SE_mu_X[0] = self.a / np.sqrt(self.delta)
        self.SE_sigma2_X[0] = (1 - self.a**2) / self.delta
        self.rho[0] = self.SE_mu_X[0] / (self.SE_mu_X[0] ** 2 + self.SE_sigma2_X[0])
        self.cor_optimal[0] = np.abs(self.SE_mu_X[0]) / np.sqrt(self.SE_sigma2_X[0] + self.SE_mu_X[0]**2)
        for t in range(1, self.iter + 1):
            self.SE_mu_U[t - 1] = self.SE_mu_X[t - 1]**2 / (
                        self.SE_mu_X[t - 1]**2 + self.SE_sigma2_X[t - 1]) / np.sqrt(self.delta)
            self.SE_sigma2_U[t - 1] = self.SE_mu_X[t - 1]**2 * self.SE_sigma2_X[t - 1] / (
                        self.SE_mu_X[t - 1]**2 + self.SE_sigma2_X[t - 1])**2 / self.delta
            self.SE_mu_X[t] = np.sqrt(self.delta) * np.mean(
                self.G * self.h(self.SE_mu_U[t - 1] * self.G + np.sqrt(self.SE_sigma2_U[t - 1]) * self.W, self.Y,
                                self.SE_sigma2_U[t - 1], self.SE_mu_U[t - 1])) - \
                              self.SE_mu_U[t - 1] * np.sqrt(self.delta) * np.mean(
                self.h_derivative(self.SE_mu_U[t - 1] * self.G + np.sqrt(self.SE_sigma2_U[t - 1]) * self.W, self.Y,
                                  self.SE_sigma2_U[t - 1], self.SE_mu_U[t - 1]))
            self.SE_sigma2_X[t] = np.mean(
                self.h(self.SE_mu_U[t - 1] * self.G + np.sqrt(self.SE_sigma2_U[t - 1]) * self.W, self.Y,
                       self.SE_sigma2_U[t - 1], self.SE_mu_U[t - 1])**2)
            self.rho[t] = self.SE_mu_X[t] / (self.SE_mu_X[t]**2 + self.SE_sigma2_X[t])
            self.cor_optimal[t] = np.abs(self.SE_mu_X[t]) / np.sqrt((self.SE_mu_X[t]**2 + self.SE_sigma2_X[t]))
        return

    @staticmethod
    def get_cor(a, b):
        return np.dot(a, b) / np.sqrt(np.sum(a**2) * np.sum(b**2))

    def AMP(self):
        if self.verbose:
            print('Running AMP...')
        self.X_AMP[:, 0] = np.sqrt(self.d / self.delta) * self.xs
        self.cor_AMP[0] = np.abs(self.get_cor(self.x, self.X_AMP[:, 0]))
        b = self.rho[0] / self.delta
        self.U_AMP[:, 0] = self.rho[0] * self.A @ self.X_AMP[:, 0] / np.sqrt(self.delta) - b * np.sqrt(
            self.delta) / self.lambda_star * self.Zs @ self.A @ self.X_AMP[:, 0]
        for t in range(1, self.iter + 1):
            c = np.mean(self.h_derivative(self.U_AMP[:, t - 1], self.y, self.SE_sigma2_U[t - 1], self.SE_mu_U[t - 1]))
            b = self.rho[t] / self.delta
            self.X_AMP[:, t] = self.A.T @ self.h(self.U_AMP[:, t - 1], self.y, self.SE_sigma2_U[t - 1],
                                             self.SE_mu_U[t - 1]) / np.sqrt(self.delta) - c * self.rho[t - 1] * self.X_AMP[:, t - 1]
            self.U_AMP[:, t] = self.A @ self.X_AMP[:, t] * self.rho[t] / np.sqrt(self.delta) - b * \
                           self.h(self.U_AMP[:, t - 1], self.y, self.SE_sigma2_U[t - 1], self.SE_mu_U[t - 1])
            self.cor_AMP[t] = np.abs(self.get_cor(self.x, self.X_AMP[:, t]))
            if self.verbose:
                print(f'Iteration = {t}, loss = {np.mean((self.y - (self.A @ self.X_AMP[:, t] / np.sqrt(np.sum(self.X_AMP[:, t]**2) / self.d))**2)**2)}')
        return

    # -----Gradient descent-----------------------------------------------------

    def gradient_descent(self):
        if self.verbose:
            print('Running gradient descent...')
        self.X_GD[:, 0] = self.xs * np.sqrt(self.d)
        self.cor_GD[0] = np.abs(self.get_cor(self.x, self.X_GD[:, 0]))
        for t in range(1, self.iter + 1):
            g = -4 * self.A.T @ ((self.y - (self.A @ self.X_GD[:, t - 1])**2) * (self.A @ self.X_GD[:, t - 1])) / self.n
            self.X_GD[:, t] = self.X_GD[:, t - 1] - self.lr * g
            if self.verbose:
                print(f'Iteration = {t}, loss = {np.mean((self.y - (self.A @ self.X_GD[:, t]) ** 2) ** 2)}')
            self.cor_GD[t] = np.abs(self.get_cor(self.x, self.X_GD[:, t]))
        return

    # -----prox-linear algorithm-----------------------------------------------------
    # https://arxiv.org/pdf/1705.02356.pdf

    def prox_linear(self, exact=True, GD=True):
        if self.verbose:
            print('Running prox-linear algorithm...')
        if exact:
            self.X_prox_linear[:, 0] = self.xs * np.sqrt(self.d)
            self.cor_prox_linear[0] = np.abs(self.get_cor(self.x, self.X_prox_linear[:, 0]))
        if GD:
            self.X_prox_linear_GD[:, 0] = self.xs * np.sqrt(self.d)
            self.cor_prox_linear_GD[0] = np.abs(self.get_cor(self.x, self.X_prox_linear_GD[:, 0]))
        for t in range(1, self.iter + 1):
            if exact:
                x0 = self.X_prox_linear[:, t - 1]
                x = cp.Variable(self.d)
                objective = cp.Minimize(
                    self.L / 2 * cp.sum_squares(x - x0) + cp.norm(
                        (self.A @ x0) ** 2 + 2 * cp.multiply((self.A @ x0), (self.A @ (x - x0))) - self.y,
                        1) / self.delta)
                prob = cp.Problem(objective)
                prob.solve()
                self.X_prox_linear[:, t] = x.value
                self.cor_prox_linear[t] = np.abs(self.get_cor(self.x, self.X_prox_linear[:, t]))

            if GD:
                x0 = self.X_prox_linear_GD[:, t - 1]
                self.X_prox_linear_GD[:, t] = x0 - self.lr * self.prox_linear_gd(x0)
                self.cor_prox_linear_GD[t] = np.abs(self.get_cor(self.x, self.X_prox_linear_GD[:, t]))

            if self.verbose:
                if exact:
                    print(f'Iteration = {t}, loss = {np.mean((self.y - (self.A @ self.X_prox_linear[:, t])**2)**2)}')
                if GD:
                    print(f'Iteration = {t}, (1 step GD) loss = {np.mean((self.y - (self.A @ self.X_prox_linear_GD[:, t])**2)**2)}')
        return

    def prox_linear_gd(self, x0):
        y = np.sign((self.A @ x0)**2 - self.y)
        return 2 / self.delta * self.A.T @ (y * (self.A @ x0))

    # -----Truncated Amplitude Flow-----------------------------------------------------
    # https://arxiv.org/pdf/1605.08285.pdf

    def TAF(self):
        if self.verbose:
            print('Running Truncated Amplitude Flow...')
        gamma = 0.7
        alpha = 0.6
        psi = np.sqrt(self.y)
        self.X_TAF[:, 0] = self.xs * np.sqrt(self.d)
        self.cor_TAF[0] = np.abs(self.get_cor(self.x, self.X_TAF[:, 0]))
        for t in range(1, self.iter + 1):
            Gamma = (np.abs(self.A @ self.X_TAF[:, t - 1]) >= psi / (1 + gamma)) * 1
            self.X_TAF[:, t] = self.X_TAF[:, t - 1] - alpha / self.delta * self.A.T @ (
                        Gamma * (self.A @ self.X_TAF[:, t - 1] - psi * np.sign(self.A @ self.X_TAF[:, t - 1])))
            self.cor_TAF[t] = np.abs(self.get_cor(self.x, self.X_TAF[:, t]))
            if self.verbose:
                print(f'Iteration = {t}, loss = {np.mean((self.y - (self.A @ self.X_TAF[:, t]) ** 2) ** 2)}')

        return

#    if __name__ == '__main__':



