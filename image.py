import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from algs import Simulation
import random
from importlib import reload


class Image_Simulation(Simulation):
    def __init__(self, verbose=True, save=True, iter=10, colored=False, col_idx=None):
        Simulation.__init__(self, verbose=verbose, save=save, delta_tilde=2, lr1=1, lr2=0.05, iter=iter)
        self.colored = colored
        if not self.colored:
            self.image = mpimg.imread('cat2.jpeg')
            self.image = self.rgb2gray(self.image)
            self.n = 12000
        else:
            self.image = mpimg.imread('cat.jpeg')
            self.image = self.image[:, :, col_idx]
            self.n = 12000
        self.mean = np.mean(self.image)
        self.std = np.std(self.image)
        self.image = (self.image - np.mean(self.image)) / np.std(self.image)
        self.shape = self.image.shape
        self.x = np.array(self.image).ravel()
        self.d = len(self.x)

        self.delta = self.n / self.d
        self.SE_mu_X = np.zeros(self.iter + 1)
        self.SE_sigma2_X = np.zeros(self.iter + 1)
        self.SE_mu_U = np.zeros(self.iter + 1)
        self.SE_sigma2_U = np.zeros(self.iter + 1)
        self.cor_AMP = np.zeros(self.iter + 1)
        self.cor_GD = np.zeros(self.iter + 1)
        self.cor_prox_linear = np.zeros(self.iter + 1)
        self.cor_prox_linear_GD = np.zeros(self.iter + 1)
        self.cor_TAF = np.zeros(self.iter + 1)
        self.cor_optimal = np.zeros(self.iter + 1)
        self.rho = np.zeros(self.iter + 1)
        self.get_lambda_bar()
        self.get_lambda_star()
        self.get_sensing_vector()
        self.get_a()
        self.get_SE()
        self.X_AMP = np.zeros((self.d, self.iter + 1))
        self.U_AMP = np.zeros((self.n, self.iter + 1))
        self.X_GD = np.zeros((self.d, self.iter + 1))
        self.X_prox_linear = np.zeros((self.d, self.iter + 1))
        self.X_prox_linear_GD = np.zeros((self.d, self.iter + 1))
        self.X_TAF = np.zeros((self.d, self.iter + 1))

    def run(self):
        self.AMP()
        self.lr=self.lr1
        self.gradient_descent()
        self.lr = self.lr2
        self.prox_linear(exact=False, GD=True)
        self.TAF()
        return

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def show_image(self, x, to_file=None):
        img = x.reshape(self.shape)
        plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=img.min(), vmax=img.max())
        plt.show()
        plt.axis('off')
        if to_file is not None:
            plt.savefig(to_file)
        return


    def get_sensing_vector(self):
        self.A = np.random.normal(size=(self.n, self.d)) / np.sqrt(self.d)
        self.y = (self.A @ self.x)**2
        self.Zs = np.diag(self.Gamma(self.y))
        D = self.A.T @ self.Zs @ self.A
        _, self.xs = self.power_method(D)
        self.L, _ = self.power_method(self.A.T @ self.A)
        self.L = self.L * 2 / self.delta
        return

    def power_method(self, X, iter=1000):
        n = X.shape[0]
        X = X + 5 * np.eye(n)
        x0 = np.random.normal(size=n)
        x0 = self.normalize(x0)
        for i in range(iter):
            if self.verbose:
                print(f'Running the {i}-th iteration for the power method')
            x0 = self.normalize(X @ x0)

        return np.sqrt(np.sum((X @ x0)**2) / np.sum(x0**2)) - 10, x0

    @staticmethod
    def normalize(v):
        return v / np.sqrt(np.sum(v**2))


def rescale(img):
    img[:, :, 0] = (img[:, :, 0] - img[:, :, 0].min()) / (img[:, :, 0].max() - img[:, :, 0].min())
    img[:, :, 1] = (img[:, :, 1] - img[:, :, 1].min()) / (img[:, :, 1].max() - img[:, :, 1].min())
    img[:, :, 2] = (img[:, :, 2] - img[:, :, 2].min()) / (img[:, :, 2].max() - img[:, :, 2].min())
    return img



if __name__ == '__main__':
    random.seed(220109)
    dir = os.getcwd() + '/figures/'
    sim0 = Image_Simulation(colored=True, col_idx=0, verbose=False, iter=20)
    sim1 = Image_Simulation(colored=True, col_idx=1, verbose=False, iter=20)
    sim2 = Image_Simulation(colored=True, col_idx=2, verbose=False, iter=20)
    sim0.run()
    sim1.run()
    sim2.run()

    for iter in [2, 4, 8, 16]:
        img = np.zeros(list(sim0.shape) + [3])
        img[:, :, 0] = (sim0.normalize(sim0.x) * np.sqrt(sim0.d) * sim0.std + sim0.mean).reshape(
            sim0.shape)
        img[:, :, 1] = (sim1.normalize(sim1.x) * np.sqrt(sim1.d) * sim1.std + sim1.mean).reshape(
            sim1.shape)
        img[:, :, 2] = (sim2.normalize(sim2.x) * np.sqrt(sim2.d) * sim2.std + sim2.mean).reshape(
            sim2.shape)
        plt.imshow(rescale(img))
        plt.show()
        plt.axis('off')
        plt.savefig(dir + 'cat.pdf')

        img[:, :, 0] = -(sim0.normalize(sim0.X_AMP[:, iter]) * np.sqrt(sim0.d) * sim0.std + sim0.mean).reshape(
            sim0.shape)
        img[:, :, 1] = -(sim1.normalize(sim1.X_AMP[:, iter]) * np.sqrt(sim1.d) * sim1.std + sim1.mean).reshape(
            sim1.shape)
        img[:, :, 2] = -(sim2.normalize(sim2.X_AMP[:, iter]) * np.sqrt(sim2.d) * sim2.std + sim2.mean).reshape(
            sim2.shape)
        plt.imshow(rescale(img))
        plt.show()
        plt.axis('off')
        plt.savefig(dir + f'AMP_iter={iter}.pdf')

        img[:, :, 0] = -(sim0.normalize(sim0.X_GD[:, iter]) * np.sqrt(sim0.d) * sim0.std + sim0.mean).reshape(sim0.shape)
        img[:, :, 1] = -(sim1.normalize(sim1.X_GD[:, iter]) * np.sqrt(sim1.d) * sim1.std + sim1.mean).reshape(sim1.shape)
        img[:, :, 2] = (sim2.normalize(sim2.X_GD[:, iter]) * np.sqrt(sim2.d) * sim2.std + sim2.mean).reshape(sim2.shape)
        plt.imshow(rescale(-img))
        plt.show()
        plt.axis('off')
        plt.savefig(dir + f'GD_iter={iter}.pdf')

        img[:, :, 0] = -(sim0.normalize(sim0.X_prox_linear_GD[:, iter]) * np.sqrt(sim0.d) * sim0.std + sim0.mean).reshape(sim0.shape)
        img[:, :, 1] = (sim1.normalize(sim1.X_prox_linear_GD[:, iter]) * np.sqrt(sim1.d) * sim1.std + sim1.mean).reshape(sim1.shape)
        img[:, :, 2] = -(sim2.normalize(sim2.X_prox_linear_GD[:, iter]) * np.sqrt(sim2.d) * sim2.std + sim2.mean).reshape(sim2.shape)
        plt.imshow(rescale(-img))
        plt.show()
        plt.axis('off')
        plt.savefig(dir + f'prox_linear_GD_iter={iter}.pdf')

        img[:, :, 0] = (sim0.normalize(sim0.X_TAF[:, iter]) * np.sqrt(sim0.d) * sim0.std + sim0.mean).reshape(sim0.shape)
        img[:, :, 1] = -(sim1.normalize(sim1.X_TAF[:, iter]) * np.sqrt(sim1.d) * sim1.std + sim1.mean).reshape(sim1.shape)
        img[:, :, 2] = -(sim2.normalize(sim2.X_TAF[:, iter]) * np.sqrt(sim2.d) * sim2.std + sim2.mean).reshape(sim2.shape)
        plt.imshow(rescale(-img))
        plt.show()
        plt.axis('off')
        plt.savefig(dir + f'TAF_iter={iter}.pdf')


