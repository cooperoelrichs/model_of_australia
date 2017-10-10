import os
import sys
sys.path.insert(0, ".")
from scipy import stats

import numpy as np

from model_of_australia.simulation_container import SimulationContainer
from model_of_australia.plotting_tools import PlottingTools
from model_of_australia.data_loader import DataLoader
from model_of_australia.simulators import CommonDistrubutionSimulator

from scripts.load_data import load_gva_pc_d, load_gdp_pc_d, load_un_gdp_pc_d
from scripts import settings
from scripts.fit_models import fit_simple_international_model
from scripts.simulation_summariser import (
    summarise_gdp_data, summarise_simulations
)


simple_internation_model = fit_simple_international_model()
X, D = load_un_gdp_pc_d()
simple_internation_simulation = SimulationContainer(
    name='Simple Internation Simulation',
    folder='simple_international_model',
    outputs_dir=settings.OUTPUTS_DIR,
    simulator=CommonDistrubutionSimulator,
    values_deltas_pair=(X['Australia'], D['Australia']),
    load_parameters=True,
    n_years=settings.N_YEARS, n_iter=settings.N_ITER
)
simple_internation_simulation.simulate()


gdp_data = [
    ('un gdp aust data', load_un_gdp_pc_d()[0]['Australia'].values),
]
summarise_gdp_data(gdp_data)

simulation_results = [
    ('simple international model', simple_internation_simulation,
     load_un_gdp_pc_d()[0]['Australia'].values, load_un_gdp_pc_d()[0].index),
]
summarise_simulations(simulation_results)
