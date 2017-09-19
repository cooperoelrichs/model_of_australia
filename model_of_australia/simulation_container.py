import os
import pandas as pd
import matplotlib.pyplot as plt
from scripts import settings


class SimulationContainer(object):
    def __init__(self, name, folder, values_deltas_pair, simulator,
                 parameters, n_years, n_iter):
        self.name = name
        self.folder = folder
        self.simulator = simulator

        self.values = values_deltas_pair[0]
        self.deltas = values_deltas_pair[1]

        self.parameters = parameters
        self.n_years = n_years
        self.n_iter = n_iter

    def simulate(self):
        self.simulate = self.run()
        self.plot_sim()

    def run(self):
        simulation = self.simulator.run(
            self.values.values,
            self.parameters,
            self.n_years, self.n_iter
        )
        return simulation

    def make_date_range(self):
        return pd.date_range(
            self.values.index.max() + pd.Timedelta(1, 'Y'),
            periods=self.n_years,
            freq='BAS'
        )

    def plot_sim(self):
        plt.figure(figsize=(20, 10))

        print(self.values.index.shape, self.values.values.sum(axis=1).shape)

        plt.plot(self.values.index, self.values.values.sum(axis=1))
        plt.plot(self.make_date_range(), self.simulate.T)
        file_path = os.path.join(
            settings.OUTPUTS_DIR,
            self.folder,
            'simulation_plot.png'
        )
        print(file_path)
        plt.savefig(file_path)
