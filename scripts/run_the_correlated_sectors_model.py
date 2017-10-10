import os
import sys
sys.path.insert(0, ".")
from scipy import stats

import numpy as np

from model_of_australia.simulation_container import SimulationContainer
from model_of_australia.plotting_tools import PlottingTools
from model_of_australia.data_loader import DataLoader
from model_of_australia.simulators import GDPSimulatorWithCorrelatedSectors

from scripts.load_data import load_gva_pc_d, load_gdp_pc_d, load_un_gdp_pc_d
from scripts import settings
from scripts.fit_models import fit_correlated_sectors_model
from scripts.simulation_summariser import (
    summarise_gdp_data, summarise_simulations
)


correlated_sectors_model = fit_correlated_sectors_model()
correlated_sectors_sim = SimulationContainer(
    name='Correlated Sectors Simulation',
    folder='correlated_sectors_model',
    outputs_dir=settings.OUTPUTS_DIR,
    simulator=GDPSimulatorWithCorrelatedSectors,
    values_deltas_pair=load_gva_pc_d(),
    load_parameters=True,
    n_years=settings.N_YEARS, n_iter=settings.N_ITER
)
correlated_sectors_sim.simulate()


gdp_data = [
    ('gva data', load_gva_pc_d()[0].values),
]
summarise_gdp_data(gdp_data)

simulation_results = [
    ('correlated sectors model', correlated_sectors_sim,
     load_gdp_pc_d()[0].values, load_gdp_pc_d()[0].index),
]
summarise_simulations(simulation_results)
