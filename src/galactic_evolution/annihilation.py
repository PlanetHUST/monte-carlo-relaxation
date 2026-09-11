from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .diffusion import thermal_particle_coefficients


class Particle:

    def __init__(self, x, j):
        self.energy = x
        self.angularmomentum = j

    # def __repr__(self):
    #     return f"Particle(energy={self.energy},angularmomentum={self.angularmomentum})"

    def steps_evolution(self, Dx, Dxx, Dj, Djj, Dxj, counts):
        # step evolution for every particle cluster
        x0 = self.energy
        j0 = self.angularmomentum
        # cor = Dxj / np.sqrt(Dxx * Djj)
        X = np.random.normal(0, 1)
        Z = np.random.normal(0, 1)
        # Y = cor * X + np.sqrt(1 - cor**2) * Z
        Y = Z
        # generate random numbers with given correlavant
        dx = counts * Dx + counts ** (1 / 2) * X * np.sqrt(Dxx)
        dj = counts * Dj + counts ** (1 / 2) * Y * np.sqrt(Djj)
        new = Particle(x0 + dx, j0 + dj)
        return new, dx, dj

    # counts computation
    def count_computation(self, Dx, Dxx, Dj, Djj):
        x = self.energy
        j = self.angularmomentum
        counts = max(
            0.15 * x / np.sqrt(Dxx),
            0.1 / np.sqrt(Djj),
            0.4 * (0.0075 - j) / np.sqrt(Djj),
        )
        return counts


# define program parameters
times = int(1e3)
tde_num = 0
unphy_num = 0
RUNTIME_DATA_DIR = Path("data/runtime")


def main():

    # define energy/angulermomentum track index
    ind_track = list(range(10, 1001, 10))
    energy_df = pd.DataFrame(columns=ind_track)
    angular_df = pd.DataFrame(columns=ind_track)

    start_time = time.time()
    print("start")

    # maximal value of x, NO MORE THAN xd
    xd = 1e4

    # correct_particle for conserved number
    x_new = np.random.uniform(1, 0.1 * xd)
    j_min = np.sqrt(2 * (x_new / xd) - (x_new / xd) ** 2)
    j_new = np.random.uniform(j_min, 1)
    correct_particle = Particle(x_new, j_new)

    # this task aim to get the full evolving new particles in one step
    def step_tasks(particlelist, ind, local_unphy=0, local_tde=0):
        global tde_num
        global unphy_num
        newparticlelist = particlelist[:]
        x_list = [_.energy for _ in particlelist]
        j_list = [_.angularmomentum for _ in particlelist]
        i = [ind for _ in particlelist]
        with ProcessPoolExecutor() as executor:
            coefficients = list(
                executor.map(thermal_particle_coefficients, x_list, j_list, i)
            )

        for _ in particlelist:
            iid = particlelist.index(_)
            x = x_list[iid]
            j = j_list[iid]
            jmin = np.sqrt(2 * (x / xd) - (x / xd) ** 2)
            Dx, Dxx, Dj, Djj, Dxj = coefficients[iid]
            counts = _.count_computation(Dx, Dxx, Dj, Djj)
            newparticlelist[iid], dx, dj = _.steps_evolution(
                Dx, Dxx, Dj, Djj, Dxj, counts
            )
            if dj >= 1 - j:
                unphy_num += 1  # tidal disruption event number count
                local_unphy += 1
                newparticlelist[iid] = correct_particle
            if dj <= jmin - j:
                tde_num += 1
                local_tde += 1
                newparticlelist[iid] = correct_particle
            if (x + dx <= 0) or (x + dx >= xd):
                unphy_num += 1
                local_unphy += 1
                newparticlelist[iid] = (
                    correct_particle  # unphysical, energy is negative or too high
                )
        return newparticlelist, local_unphy, local_tde

    # 1 define initial index
    ind = 0

    # 2 evolve and justify #no annihilation scheme !!!
    for ii in range(times):
        df = pd.read_csv(RUNTIME_DATA_DIR / f"copy_data_{ind}.csv")
        xlist = df["xvalue"].values
        jlist = df["jvalue"].values
        test = [Particle(x, j) for x, j in zip(xlist, jlist)]
        newtest, local_unphy, local_tde = step_tasks(test, ind, local_unphy=0, local_tde=0)

        xlist = [_.energy for _ in newtest]
        jlist = [_.angularmomentum for _ in newtest]
        data = {"xvalue": xlist, "jvalue": jlist}
        df = pd.DataFrame(data)

        # get energy/angulermomentum trajectory
        energy_df.loc[ii] = {i: xlist[i] for i in ind_track}
        angular_df.loc[ii] = {i: jlist[i] for i in ind_track}

        # Save trajectory data to csv files
        def save_trajectories():
            energy_df.to_csv(RUNTIME_DATA_DIR / "energytrack.csv", index=False)
            angular_df.to_csv(RUNTIME_DATA_DIR / "angulertrack.csv", index=False)

        if ii % 100 == 0:  # save every 100 iterations
            print(f"Saving trajectories at iteration {ii}...")
            save_trajectories()
        # Determine index based on iteration count
        if ii <= times / 1e3:
            ind = 1
        elif ii <= times / 1e2:
            ind = 2
        elif ii <= times / 10:
            ind = 3
        else:
            ind = 4
        df.to_csv(RUNTIME_DATA_DIR / f"copy_data_{ind}.csv", index=False)

        print(f"the {ii+1} loop is done. in this loop, {local_unphy} unphysical particles and {local_tde} tde particles are corrected.")


    end_time = time.time()
    print(f"The program took {(end_time-start_time)/60:.2f} minutes to execute.")


if __name__ == "__main__":
    main()
