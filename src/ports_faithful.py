"""
Line-by-line Python ports of the two MATLAB reference implementations released by
Fathy et al. at https://github.com/YasminFathy/AMDRIoT :

  baseline/dual_prediction.m         -> lms_dual_prediction_matlab()
      (their implementation of Santini & Roemer [13], LMS dual prediction)
  myapproach_AMDR/AMDR_dual_prediction.m -> amdr_dual_prediction_matlab()
      (AM-DR, Fathy et al. [25])

These functions reproduce the MATLAB scripts exactly, including their quirks,
so that the port can be checked against the numbers the authors report. MATLAB
indices are 1-based; every array here is 0-based, and each MATLAB line is quoted
next to its translation.
"""
import numpy as np


def lms_dual_prediction_matlab(x, N=5, mu=1e-5, emax=0.5):
    x = np.asarray(x, float)
    L = len(x)
    w = np.zeros((L + 1, N))          # w = zeros(len_x, N)  (+1 row: MATLAB grows w(k+1,:))
    y = np.zeros(L)                   # y = zeros(len_x,1)
    x_e = np.zeros(L)                 # x_e = zeros(len_x,1)
    tx = np.zeros(L, bool)            # e_max = zeros(len_x,1)
    tx[:N] = True                     # e_max(1:N) = 1
    y[:N] = x[:N]                     # y(1:N) = x(1:N)
    x_e[:N] = x[:N]                   # x_e(1:N) = x(1:N)
    e = np.zeros(L)                   # e = zeros(len_x,1)
    e[:N] = 1.0                       # e(1:N) = ones(N,1)
    for k in range(N, L):             # for k = N+1 : len_x   (0-based k = MATLAB k-1)
        # y(k) = sum(fliplr(w(k,:)') .* x_e(k-N:k-1));
        # fliplr of a COLUMN vector is a no-op, so w_i multiplies x_e(k-N+i-1):
        y[k] = np.dot(w[k], x_e[k - N:k])
        e[k] = x[k] - y[k]            # e(k) = x(k) - y(k)
        if np.all(np.abs(e[k - N + 1:k + 1]) < emax):   # all(abs(e(k-N+1:k)) < emax)
            x_e[k] = y[k]
            w[k + 1] = w[k]
        else:
            tx[k] = True
            x_e[k] = x[k]
            # w(k+1,i) = w(k,i) + (mu * x_e(k-N+i-1) * e(k)),  i = 1..N
            w[k + 1] = w[k] + mu * x_e[k - N:k] * e[k]
    return {"tx": tx, "x_e": x_e, "y": y, "e": e}


def amdr_dual_prediction_matlab(x, w_s=10, w_f=5, alpha=1e-7, emax=0.5):
    x = np.asarray(x, float)
    L = len(x)
    lam = np.zeros(L + 1)             # lambda = zeros(len_x,1)
    y = np.zeros(L)
    e = np.zeros(L)
    tx = np.zeros(L, bool)
    N = w_s                           # N = w_s
    x_e = np.zeros(L)
    x_e[:N] = x[:N]                   # x_e(1:N) = x(1:N)
    tx[:N] = True                     # e_max(1:N) = 1
    e[:N] = 1.0                       # e(1:N) = ones(N,1)
    start = 0                         # start = 1  (MATLAB) -> 0-based 0
    maf = np.zeros(L)
    mas = np.zeros(L)
    for k in range(w_f - 1, N):       # for k = w_f : N
        maf[k] = x[k - w_f + 1:k + 1].mean()   # mean(x(k-w_f+1:k))  (unused later)
    for k in range(N, L):             # for k = N+1 : len_x
        maf[k] = x_e[k - w_f + 1:k].mean()     # mean(x_e(k-w_f+1:k-1))  -> w_f-1 samples
        mas[k] = x_e[start:k].mean()           # mean(x_e(start:k-1))
        y[k] = lam[k] * maf[k] + (1 - lam[k]) * mas[k]
        e[k] = x[k] - y[k]
        wdiff = maf[k] - mas[k]
        if np.all(np.abs(e[k - w_f + 1:k + 1]) < emax):   # all(abs(e(k-w_f+1:k)) < emax)
            x_e[k] = y[k]
            lam[k + 1] = lam[k]
        else:
            tx[k] = True
            x_e[k] = x[k]
            lam[k + 1] = lam[k] + alpha * e[k] * wdiff
            start = k                 # start = k
    return {"tx": tx, "x_e": x_e, "y": y, "e": e, "lam": lam[:L]}
