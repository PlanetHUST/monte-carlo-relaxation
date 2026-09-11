from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import dblquad, quad

# 常数定义
NUM_POINTS = int(3e2)  # 网格点数
J_MIN = 3e-4  # j的最小值
OUTPUT_PATH = Path("data/runtime/newdata.csv")


def f(s, a):
    E = 1 / (2 * a)
    x = s * E
    return np.where(x >= 0, x ** (1 / 4), np.exp(x))


def Gamma_0(a):
    I, _ = quad(f, -np.inf, 1, args=(a,), epsabs=1e-6, epsrel=1e-6)
    return I


def Gamma_ijk(i, j, k, a, e):
    def integrand(s, y):
        r = (e * y + 1) * a
        return (
            f(s, a)
            * (1 - y**2) ** -0.5
            * (1 + y * e) ** i
            * (2 * (2 * a / r - 1)) ** (-k)
            * ((2 * a / r - s) / (2 * a / r - 1)) ** j
        )

    I, _ = dblquad(
        integrand,
        -1,
        1,
        lambda y: 1,
        lambda y: 2 / (1 + y),
        epsabs=1e-6,
        epsrel=1e-6,
    )
    return 2 ** (1 + k - i) / np.pi * I


def DJ(a, j):
    e = np.sqrt(1 - j**2)
    return (
        (5 - 3 * j**2) / 12 * Gamma_0(a)
        - j**2 * Gamma_ijk(1, 1, 1, a, e)
        + Gamma_ijk(3, 1, 0, a, e)
        - 1 / 3 * Gamma_ijk(3, 3, 0, a, e)
    )


def compute_point(args):
    x, j = args
    return DJ(x, j)


def main():
    # 生成参数网格
    x_vals = np.linspace(1, 1e4, NUM_POINTS)
    j_vals = np.linspace(J_MIN, 1 - J_MIN, NUM_POINTS)
    param_grid = list(product(x_vals, j_vals))

    # 并行计算
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(compute_point, param_grid)

    # 重塑结果并保存
    result = np.array(results).reshape(NUM_POINTS, NUM_POINTS)
    pd.DataFrame(result).to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
