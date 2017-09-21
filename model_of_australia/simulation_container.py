import os
import pandas as pd
import matplotlib.pyplot as plt
from model_container import ModelContainer
from scripts import settings


class SimulationContainer(object):
    def __init__(self, name, folder, values_deltas_pair, simulator,
                 n_years, n_iter, parameters=None, load_parameters=False):
        self.name = name
        self.folder = folder
        self.simulator = simulator

        self.values = values_deltas_pair[0]
        self.deltas = values_deltas_pair[1]

        self.n_years = n_years
        self.n_iter = n_iter

        if parameters is None and load_parameters is True:
            self.parameters = ModelContainer.load_parameters(self.folder)
            print(self.parameters)
        elif parameters is not None and load_parameters is False:
            self.parameters = parameters
        else:
            raise ValueError(
                'Must either provide parameters or set load_parameters to True.'
            )

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

        if self.values.values.ndim == 1:
            reduced_values = self.values.values
        elif self.values.values.ndim == 2:
            reduced_values = self.values.values.sum(axis=1)
        else:
            raise ValueError('%i ndim is not supported yet.' % self.values.ndim)

        plt.plot(self.values.index, reduced_values)
        plt.plot(self.make_date_range(), self.simulate.T)
        file_path = os.path.join(
            settings.OUTPUTS_DIR,
            self.folder,
            'simulation_plot.png'
        )
        plt.savefig(file_path)
