import os
import sys
sys.path.insert(0, ".")
from scipy import stats

import numpy as np

from model_of_australia.simulation_container import SimulationContainer
from model_of_australia.plotting_tools import PlottingTools
from model_of_australia.data_loader import DataLoader
from model_of_australia.simulators import SimpleSimulator

from scripts.load_data import load_gva_pc_d, load_gdp_pc_d, load_un_gdp_pc_d
from scripts import settings
from scripts.fit_models import fit_simple_australian_model
from scripts.simulation_summariser import (
    summarise_gdp_data, summarise_simulations
)


simple_australian_model = fit_simple_australian_model()
simple_australian_simulation = SimulationContainer(
    name='Simple Australian Simulation',
    folder='simple_australian_model',
    outputs_dir=settings.OUTPUTS_DIR,
    simulator=SimpleSimulator,
    values_deltas_pair=load_gdp_pc_d(),
    load_parameters=True,
    n_years=settings.N_YEARS, n_iter=settings.N_ITER
)
simple_australian_simulation.simulate()


gdp_data = [
    ('gdp data', load_gdp_pc_d()[0].values),
]
summarise_gdp_data(gdp_data)

simulation_results = [
    ('simple normal model', simple_australian_simulation,
     load_gdp_pc_d()[0].values, load_gdp_pc_d()[0].index),
]
summarise_simulations(simulation_results)
