# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:42:49 2018

@author: zhaiweichen
"""

import math
import numpy as np
from scipy import linalg

class CausalCalculator:
    def __init__(self, X, Y_cause):
        """data for test whether the time series Y_cause causes X
        :param X: dim(X) = (N, dm) = (length of time series, dimension of X)
        :param Y_cause:
        """
        self.X = X
        self.Y = Y_cause
        self.X_t = X.T
        self.Y_t = Y_cause.T

    def calcSigmaHat(self, sigma, eta):
        return sigma + eta * np.identity(sigma.shape[0])

    def calcGrangerCausality(self, k, m, eta_xt=0.00001, eta_yt= 0.00001, eta_xtkm=0.00001):
        """
        
        :param k:
        :param m:
        :param eta_xt:
        :param eta_yt:
        :param eta_xtkm:
        :return:
        """
        N = self.X.shape[0]
        dim_X = self.X.shape[1]
        dim_Y = self.Y.shape[1]


        x_t = []
        y_t = []
        x_tk_m = []
        y_tk_m = []

        for t in range(k + m - 1, N):
            x_t.append(self.X[t])
            y_t.append(self.Y[t])

            cut_x = self.X[t - k - m + 1: t - k + 1]
            x_tk_m.append(np.ravel(cut_x[::-1]))         # reverse the array and make the array 1d array
            cut_y = self.Y[t - k - m + 1: t - k + 1]
            y_tk_m.append(np.ravel(cut_y[::-1]))         # reverse the array and make the array 1d array

        x_t = (np.array(x_t)).T
        y_t = (np.array(y_t)).T
        x_tk_m = (np.array(x_tk_m)).T
        y_tk_m = (np.array(y_tk_m)).T

        dim_x_t = x_t.shape[0]
        dim_y_t = y_t.shape[0]
        dim_x_tk_m = x_tk_m.shape[0]

        x = np.r_[x_t, y_t]
        y = np.r_[x_tk_m, y_tk_m]

        sigma = np.cov(m=x, y=y, rowvar=True)   # row of x and y represents a variable, and each column a single observation
        """ 
        sigma = ( sigma_xt_xt   sigma_xt_yt     sigma_xt_xtkm       sigma_xt_ytkm ) 
                ( sigma_yt_xt   sigma_yt_yt     sigma_yt_xtkm       sigma_yt_ytkm )
                ( sigma_xtkm_xt sigma_xtkm_yt   sigma_xtkm_xtkm     sigma_xtkm_ytkm )
                ( sigma_ytkm_xt sigma_ytkm_yt   sigma_ytkm_xtkm     sigma_ytkm_ytkm )
        """

        yt_start_idx = dim_x_t
        xtkm_start_idx = dim_x_t + dim_y_t
        ytkm_start_idx = dim_x_t + dim_y_t + dim_x_tk_m

        sigma_xt_xt = sigma[    0 : yt_start_idx, 0              : yt_start_idx]
        sigma_xt_xtkm = sigma[  0 : yt_start_idx, xtkm_start_idx : ytkm_start_idx]
        sigma_xt_ytkm = sigma[  0 : yt_start_idx, ytkm_start_idx :]

        sigma_xtkm_xt = sigma[  xtkm_start_idx : ytkm_start_idx, 0 : yt_start_idx]
        sigma_xtkm_xtkm = sigma[xtkm_start_idx : ytkm_start_idx, xtkm_start_idx : ytkm_start_idx]
        sigma_xtkm_ytkm = sigma[xtkm_start_idx : ytkm_start_idx, ytkm_start_idx : ]

        sigma_ytkm_xt = sigma[  ytkm_start_idx:, 0              : yt_start_idx]
        sigma_ytkm_xtkm = sigma[ytkm_start_idx:, xtkm_start_idx : ytkm_start_idx]
        sigma_ytkm_ytkm = sigma[ytkm_start_idx:, ytkm_start_idx : ]


        sigma_tilde_ytkm_xt_xtkm = sigma_ytkm_xt\
                                 - np.dot(np.dot(2 * sigma_ytkm_xtkm, np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))), sigma_xtkm_xt)\
                                 + np.dot(np.dot(np.dot(np.dot(sigma_ytkm_xtkm, np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))),sigma_xtkm_xtkm), np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))), sigma_xtkm_xt)

        sigma_tilde_xt_xt_xtkm = sigma_xt_xt \
                                   - np.dot(np.dot(2 * sigma_xt_xtkm, np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))), sigma_xtkm_xt) \
                                   + np.dot(np.dot(np.dot(np.dot(sigma_xt_xtkm, np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))),sigma_xtkm_xtkm), np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))), sigma_xtkm_xt)

        sigma_tilde_xt_ytkm_xtkm = sigma_xt_ytkm \
                                   - np.dot(np.dot(2 * sigma_xt_xtkm, np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))), sigma_xtkm_ytkm) \
                                   + np.dot(np.dot(np.dot(np.dot(sigma_xt_xtkm, np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))),sigma_xtkm_xtkm), np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))), sigma_xtkm_ytkm)

        sigma_tilde_ytkm_ytkm_xtkm = sigma_ytkm_ytkm \
                                   - np.dot(np.dot(2 * sigma_ytkm_xtkm, np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))), sigma_xtkm_ytkm) \
                                   + np.dot(np.dot(np.dot(np.dot(sigma_ytkm_xtkm, np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))),sigma_xtkm_xtkm), np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))), sigma_xtkm_ytkm)

        A = np.dot(np.dot(sigma_tilde_ytkm_xt_xtkm, np.linalg.inv(sigma_tilde_xt_xt_xtkm + eta_xt * np.identity(sigma_tilde_xt_xt_xtkm.shape[0]))), sigma_tilde_xt_ytkm_xtkm)
        B = sigma_tilde_ytkm_ytkm_xtkm + eta_yt * np.identity(sigma_tilde_ytkm_ytkm_xtkm.shape[0])

        eigenvalues = np.real(linalg.eig(a=A, b=B)[0])
        eigenvalue = np.max(eigenvalues)
        if eigenvalue > 1.0:
            eigenvalue = 0.9999
        Gyx = 0.5 * math.log(1 / (1 - eigenvalue), 2)

        return Gyx
    
if __name__ =='__main__':
    x = np.random.rand(6,1)
    y = np.random.rand(6,1)
    c = CausalCalculator(y,x)
    print(c.calcGrangerCausality(2,1))
