import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import scipy.integrate
from scipy.integrate import IntegrationWarning, quad
from scipy.stats import gaussian_kde


# initial conditions#
class Diffusion:
    P0 = 3.2e-2
    a_x = 3 * (2 * np.pi) ** 0.5 * P0
    a_xx = 4 * (2 * np.pi) ** 0.5 * P0
    a_j = (2 * np.pi) ** 0.5 * P0
    a_jj = (2 * np.pi) ** 0.5 * P0
    a_xj = 2 * (2 * np.pi) ** 0.5 * P0

    def __init__(self, x, j, g):
        self.x = x
        self.j = j
        self.g = g
        self.xap = 2 * x / (1 + (1 - j**2) ** 0.5)
        self.xp = 2 * x / (1 - (1 - j**2) ** 0.5)

    # def xap(x, j):
    #    !!! return 2 * x / (1 + (1 - j) ** 2) ** 0.5

    # def xp(x, j):
    #    !!! return 2 * x / (1 - (1 - j) ** 2) ** 0.5

    # define useful functions#

    def x1(self, y):
        return 1 / (1 / max(y, self.xap) - 1 / self.xp)

    def x2(self, y):
        return 1 / (1 / min(y, self.xap) - 1 / self.xp)

    def x3(self, x):
        return 1 / (1 / x - 1 / self.xp)

    def z1(self, theta, y):
        return 1 + (self.xp / self.x1(y)) * math.sin(theta) ** 2

    def z2(self, theta, y):
        return 1 - (self.x2(y) / self.x1(y)) * math.sin(theta) ** 2

    def z3(self, theta, y, x):
        return 1 - (self.x3(x) / self.x1(y)) * math.sin(theta) ** 2

    def I1(self, x):
        return 1 / 4 * np.pi * x ** (-3 / 2)

    def I2(self, y):

        x = self.x

        def integrant(theta, y, x):
            return (
                self.z1(theta, y)
                * self.z2(theta, y) ** 0.5
                * self.z3(theta, y, x) ** -0.5
            )

        I, error = scipy.integrate.quad(integrant, 0, 2 * np.pi, args=(y, x))
        return (y * self.x3(x) / (x**2 * self.xp**2 * self.x2(y))) ** 0.5 * I

    def I3(self, y):

        x = self.x

        def integrant(theta, y, x):
            return (
                math.cos(theta) ** 2
                * self.z1(theta, y)
                * self.z2(theta, y) ** -0.5
                * self.z3(theta, y, x) ** -0.5
            )

        I, error = scipy.integrate.quad(integrant, 0, 2 * np.pi, args=(y, x))
        return (
            y * self.x3(x) * self.x2(y) / (x**2 * self.xp**2 * self.x1(y) ** 2)
        ) ** 0.5 * I

    def I4(self, x):
        return 1 / 4 * np.pi * x ** (-1 / 2)

    def I5(self, y):

        x = self.x

        def integrant(theta, y, x):
            return self.z2(theta, y) ** (3 / 2) * self.z3(theta, y, x) ** (-1 / 2)

        I, error = scipy.integrate.quad(integrant, 0, np.pi / 2, args=(y, x))
        return (y**3 * self.x3(x) / (x**2 * self.x2(y) ** 3)) ** (1 / 2) * I

    def I6(self, y):

        x = self.x

        def integrant(theta, y, x):
            return (
                math.cos(theta) ** 4
                * self.z2(theta, y) ** (-1 / 2)
                * self.z3(theta, y, x) ** (-1 / 2)
            )

        I, error = scipy.integrate.quad(integrant, 0, np.pi / 2, args=(y, x))
        return (y**3 * self.x2(y) * self.x3(x) / (x**2 * self.x1(y) ** 4)) ** (
            1 / 2
        ) * I

    def I7(self, y):

        x = self.x

        def integrant(theta, y):
            return self.z1(theta, y) ** 3

        I, error = scipy.integrate.quad(integrant, 0, np.pi / 2, args=(y,))
        return (1 / (x * self.xp**6)) ** 0.5 * I

    def I8(self, y):

        x = self.x

        def integrant(theta, y, x):
            return (
                self.z1(theta, y) ** 3
                * self.z2(theta, y) ** 0.5
                * self.z3(theta, y, x) ** (-0.5)
            )

        I, error = scipy.integrate.quad(integrant, 0, np.pi / 2, args=(y, x))
        return (y * self.x3(x) / (x**2 * self.xp**6 * self.x2(y))) ** 0.5 * I

    def I9(self, y):

        x = self.x

        def integrant(theta, y, x):
            return (
                math.cos(theta) ** 2
                * self.z1(theta, y) ** 3
                * self.z2(theta, y) ** (-0.5)
                * self.z3(theta, y, x) ** (-0.5)
            )

        I, error = scipy.integrate.quad(integrant, 0, np.pi / 2, args=(y, x))
        return (
            y * self.x2(y) * self.x3(x) / (x**2 * self.xp**6 * self.x1(y) ** 2)
        ) ** 0.5 * I

    def I10(self, y):

        x = self.x

        def integrant(theta, y, x):
            return (
                self.z1(theta, y) ** 3
                * self.z2(theta, y) ** 1.5
                * self.z3(theta, y, x) ** (-1.5)
            )

        I, error = scipy.integrate.quad(integrant, 0, np.pi / 2, args=(y, x))
        return (
            y**3 * self.x3(x) ** 3 / (x**4 * self.xp**6 * self.x2(y) ** 3)
        ) ** 0.5 * I

    def I11(self, y):

        x = self.x

        def integrant(theta, y, x):
            return (
                math.cos(theta) ** 4
                * self.z1(theta, y) ** 3
                * self.z2(theta, y) ** (-0.5)
                * self.z3(theta, y, x) ** (-1.5)
            )

        I, error = scipy.integrate.quad(integrant, 0, np.pi / 2, args=(y, x))
        return (
            y**3 * self.x2(y) * self.x3(x) ** 3 / (x**4 * self.xp**6 * self.x1(y) ** 4)
        ) ** 0.5 * I

    def I12(self, y):

        x = self.x

        def integrant(theta, y, x):
            return (
                math.sin(theta) ** 2
                * math.cos(theta) ** 2
                * self.z1(theta, y) ** 2
                * self.z2(theta, y) ** 0.5
                * self.z3(theta, y, x) ** (-1.5)
            )

        I, error = scipy.integrate.quad(integrant, 0, np.pi / 2, args=(y, x))
        return (
            4 * y * self.x3(x) ** 3 / (x**2 * self.xp**4 * self.x1(y) ** 4 * self.x2(y))
        ) ** 0.5 * I

    def I13(self, y):

        x = self.x

        def integrant(theta, y, x):
            return (
                math.cos(theta) ** 2
                * math.sin(theta) ** 2
                * self.z1(theta, y) ** 2
                * self.z2(theta, y) ** 1.5
                * self.z3(theta, y, x) ** (-2.5)
            )

        I, error = scipy.integrate.quad(integrant, 0, np.pi / 2, args=(y, x))
        return (
            4
            * y**3
            * self.x3(x) ** 5
            / (x**4 * self.xp**4 * self.x1(y) ** 4 * self.x2(y) ** 3)
        ) ** 0.5 * I

    def I14(self, y):

        x = self.x

        def integrant(theta, y, x):
            return (
                math.cos(theta) ** 4
                * math.sin(theta) ** 2
                * self.z1(theta, y) ** 2
                * self.z2(theta, y) ** 0.5
                * self.z3(theta, y, x) ** (-2.5)
            )

        I, error = scipy.integrate.quad(integrant, 0, np.pi / 2, args=(y, x))
        return (
            4
            * y**3
            * self.x3(x) ** 5
            / (x**4 * self.xp**4 * self.x1(y) ** 6 * self.x2(y))
        ) ** 0.5 * I

    def I15(self, y):

        x = self.x

        def integrant(theta, y, x):
            return (
                self.z1(theta, y)
                * self.z2(theta, y) ** 1.5
                * self.z3(theta, y, x) ** (-1.5)
            )

        I, error = scipy.integrate.quad(integrant, 0, np.pi / 2, args=(y, x))
        return (
            y**3 * self.x3(x) ** 3 / (x**4 * self.xp**2 * self.x2(y) ** 3)
        ) ** 0.5 * I

    def I16(self, y):

        x = self.x

        def integrant(theta, y, x):
            return (
                math.cos(theta) ** 4
                * self.z1(theta, y) ** 1
                * self.z2(theta, y) ** (-0.5)
                * self.z3(theta, y, x) ** (-1.5)
            )

        I, error = scipy.integrate.quad(integrant, 0, np.pi / 2, args=(y, x))
        return (
            y**3 * self.x2(y) * self.x3(x) ** 3 / (x**4 * self.xp**2 * self.x1(y) ** 4)
        ) ** 0.5 * I

    def integrant1(self, y):

        x = self.x
        A = Diffusion.a_x
        if y <= x:
            return A * self.g(y) * self.I1(x)
        elif x < y and y <= self.xap:
            return -(A * self.g(y)) * self.I2(y)
        elif self.xap < y and y <= self.xp:
            return -(A * self.g(y)) * self.I3(y)

    def integrant2(self, y):

        x = self.x
        A = Diffusion.a_xx
        if y <= x:
            return A * self.g(y) * self.I4(x)
        elif x < y and y <= self.xap:
            return A * self.g(y) * self.I5(y)
        elif self.xap < y and y <= self.xp:
            return A * self.g(y) * self.I6(y)

    def integrant3(self, y):

        x = self.x
        j = self.j
        A = Diffusion.a_j * x / j
        if y < x:
            return A * self.g(y) * 2 * self.I7(y)
        elif x <= y and y < self.xap:
            return A * self.g(y) * (6 * self.I12(y) - 9 * self.I8(y) - self.I10(y))
        elif self.xap <= y and y <= self.xp:
            return A * self.g(y) * (6 * self.I12(y) - 9 * self.I9(y) - self.I11(y))

    def integrant4(self, y):

        x = self.x
        A = Diffusion.a_jj * x
        if y < x:
            return A * self.g(y) * 4 * self.I7(y)
        elif x <= y and y < self.xap:
            return A * self.g(y) * (3 * self.I12(y) + 4 * self.I10(y) - 3 * self.I13(y))
        elif self.xap <= y and y <= self.xp:
            return A * self.g(y) * (3 * self.I12(y) + 4 * self.I11(y) - 3 * self.I14(y))

    def integrant5(self, y):
        x = self.x
        j = self.j
        A = Diffusion.a_xj * j
        if y < x:
            return A * self.g(y) * self.I1(x)
        elif x <= y and y < self.xap:
            return A * self.g(y) * self.I15(y)
        elif self.xap <= y and y <= self.xp:
            return A * self.g(y) * self.I16(y)


MAX_ENERGY_X = 1.0e4
DEFAULT_RUNTIME_DATA_DIR = Path("data/runtime")


def _read_energy_samples(ind, data_dir):
    input_path = Path(data_dir) / f"copy_data_{ind}.csv"
    return pd.read_csv(input_path)["xvalue"].values


def _integrate_coefficients(energy_x, angular_momentum_j, distribution_function):
    model = Diffusion(energy_x, angular_momentum_j, distribution_function)
    integrands = [
        model.integrant1,
        model.integrant2,
        model.integrant3,
        model.integrant4,
        model.integrant5,
    ]
    coefficients = np.zeros(5)
    warnings.filterwarnings("ignore", category=IntegrationWarning)
    for index, integrand in enumerate(integrands):
        coefficients[index], _ = quad(
            integrand,
            -1.0e3,
            min(MAX_ENERGY_X, model.xp),
        )
    return coefficients


def empirical_particle_coefficients(
    energy_x,
    angular_momentum_j,
    distribution_index,
    data_dir=DEFAULT_RUNTIME_DATA_DIR,
):
    """Compute coefficients using the KDE of one saved particle population."""
    energy_samples = _read_energy_samples(distribution_index, data_dir)
    kde = gaussian_kde(energy_samples)
    kde_at_zero = kde(0)[0]

    def empirical_distribution(y):
        if y < 0:
            return np.exp(y)
        if y > 0:
            return kde(y)[0] / kde_at_zero

    return _integrate_coefficients(
        energy_x,
        angular_momentum_j,
        empirical_distribution,
    )


def thermal_particle_coefficients(
    energy_x,
    angular_momentum_j,
    distribution_index,
    data_dir=DEFAULT_RUNTIME_DATA_DIR,
):
    """Compute coefficients using the fixed thermal distribution."""
    # Preserve the original input validation even though this distribution is fixed.
    _read_energy_samples(distribution_index, data_dir).tolist()

    def thermal_distribution(y):
        if y < 0:
            return np.exp(y)
        if y < 1:
            return 1
        return y ** (1 / 4)

    return _integrate_coefficients(
        energy_x,
        angular_momentum_j,
        thermal_distribution,
    )
