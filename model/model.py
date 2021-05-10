from __future__ import division, print_function
import numpy as np
from scipy.integrate import odeint
from scipy.stats import lognorm
import multiprocessing as mp
from numba import jit

np.random.seed(1)
noise = lognorm.rvs(0.3, size=100000)

hifss, ppargss = 0.025, 0.005
deg_hif = 0.17
deg_cebp = 0.028
deg_pparg = np.log(2)/1.4
k3 = deg_hif * hifss
deg_lip = 0.005

k4 = 44.
k_r = 15.

Kc, Kh, k2, k1, Ki = np.array([0.00209, 0.000237, 1.390, 0.000155, 0.00306])
syn_hif = deg_hif * hifss

@jit
def ppar_model(ds, t, R, e0=1, over=None, kd=None, timing=0):

    x0, x1, x2, x3 = ds[0], ds[1], ds[2], ds[3]
    Eov = over * hifss if t > timing and over else 0
    Einh = 1 + kd if t > timing and kd else 1

    # equation 5
    Kinh = Kc * (1 + (x2 + Eov)/Ki)

    # equation 1
    dx0 = e0 * k1 * (R + k_r * x1**3/(Kinh**3+x1**3)) - deg_pparg * x0
    # equation 2
    dx1 = k2 * x0 - deg_cebp * x1  # CEBPA
    # equation 3
    dx2 = Einh * k3 * x0/(Kh + x0) - deg_hif * x2  # hif
    # equation 4
    dx3 = k4 * (x2 + Eov) * x0**3/(Kh**3 + x0**3) - deg_lip * x3
    return [dx0, dx1, dx2, dx3]


def run_model(rosi, trials=300, over=None, kd=None, timing=0, time=None):

    assert -1 <= kd <= 0

    yint = [0, 0, 0, 0]
    if time is None:
        time = np.linspace(0, 144, 50)
    d = []
    for _n in range(trials):
        e0 = noise[_n]
        y0 = odeint(ppar_model, yint, time, args=(rosi, e0, over, kd, timing))
        d.append(y0[-1, :])
    return np.mean(np.vstack(d), axis=0)


def run_model_sc(rosi, trials=300, over=None, kd=None, timing=0, time=None):
    yint = [0, 0, 0, 0]
    if time is None:
        time = np.linspace(0, 144, 50)
    d = []
    for _n in range(trials):
        e0 = noise[_n]
        y0 = odeint(ppar_model, yint, time, args=(rosi, e0, over, kd, timing))
        d.append(y0)
    return np.dstack(d)

