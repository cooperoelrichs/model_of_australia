import os
import sys
sys.path.insert(0, "../")
from scipy import stats

import numpy as np

from simulation_container import SimulationContainer
from plotting_tools import PlottingTools
from data_loader import DataLoader
from scripts.load_data import load_gva_pc_d, load_gdp_pc_d, load_un_gdp_pc_d
from scripts import settings
from scripts.fit_models import fit_correlated_sectors_model
from scripts.simulation_summariser import (
    summarise_gdp_data, summarise_simulations
)
from simulators import GDPSimulatorWithCorrelatedSectors


correlated_sectors_model = fit_correlated_sectors_model()
correlated_sectors_sim = SimulationContainer(
    name='Correlated Sectors Simulation',
    folder='correlated_sectors_model',
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
