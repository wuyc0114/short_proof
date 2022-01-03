import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from algs import Simulation
from importlib import reload


class Image_Simulation(Simulation):
    def __init__(self, verbose=True, save=True):
        Simulation.__init__(self, verbose=verbose, save=save, delta_tilde=2, lr1=1, lr2=0.05)
        self.image = mpimg.imread('cat2.jpeg')
        self.image = self.rgb2gray(self.image)
        self.image = (self.image - np.mean(self.image)) / np.std(self.image)
        self.shape = self.image.shape
        self.x = np.array(self.image).ravel()
        self.d = len(self.x)
        self.n = 12000
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
        self.gradient_descent()
        self.prox_linear(exact=False, GD=True)
        self.TAF()
        return

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def show_image(self, x):
        img = x.reshape(self.shape)
        plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=img.min(), vmax=img.max())
        plt.show()
        plt.axis('off')
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
        print(self.get_cor(x0, X @ x0))

        return np.sqrt(np.sum((X @ x0)**2) / np.sum(x0**2)) - 10, x0

    @staticmethod
    def normalize(v):
        return v / np.sqrt(np.sum(v**2))



