import os
import sys
sys.path.insert(0, ".")
from scipy import stats

import numpy as np

from model_of_australia.simulation_container import SimulationContainer
from model_of_australia.plotting_tools import PlottingTools
from model_of_australia.data_loader import DataLoader
from model_of_australia.simulators import SharedVarianceInternationalGDPSimulator

from scripts.load_data import load_gva_pc_d, load_gdp_pc_d, load_un_gdp_pc_d
from scripts import settings
from scripts.fit_models import fit_international_shared_variance_model
from scripts.simulation_summariser import (
    summarise_gdp_data, summarise_simulations
)


international_shared_variance_model = fit_international_shared_variance_model()
X, D = load_un_gdp_pc_d()
international_shared_variance_sim = SimulationContainer(
    name='International Shared Variance Simulation',
    folder='international_shared_variance_model',
    outputs_dir=settings.OUTPUTS_DIR,
    simulator=SharedVarianceInternationalGDPSimulator,
    values_deltas_pair=(X['Australia'], D['Australia']),
    load_parameters=True,
    n_years=settings.N_YEARS, n_iter=settings.N_ITER
)
international_shared_variance_sim.simulate()

gdp_data = [
    ('un gdp aust data', load_un_gdp_pc_d()[0]['Australia'].values),
]
summarise_gdp_data(gdp_data)

simulation_results = [
        ('shared var international model', international_shared_variance_sim,
        load_un_gdp_pc_d()[0]['Australia'].values, load_un_gdp_pc_d()[0].index),
]
summarise_simulations(simulation_results)
