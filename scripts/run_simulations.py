import os
import sys
sys.path.insert(0, ".")
from scipy import stats
import numpy as np

# import model_of_australia
# print(model_of_australia)
# print(dir(model_of_australia))
#
# from model_of_australia import scripts
# print(scripts)
# print(dir(scripts))

from model_of_australia.simulation_container import SimulationContainer
from model_of_australia.plotting_tools import PlottingTools
from model_of_australia.data_loader import DataLoader
from model_of_australia.simulators import (
    GDPSimulatorWithCorrelatedSectors,
    SharedVarianceInternationalGDPSimulator,
    SimpleSimulator,
    CommonDistrubutionSimulator
)

from scripts import settings
from scripts.load_data import (
    load_gva_pc_d, load_gdp_pc_d, load_un_gdp_pc_d
)
from scripts.fit_models import (
    fit_international_shared_variance_model,
    fit_correlated_sectors_model,
    fit_simple_australian_model,
    fit_simple_international_model
)
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
    ('gdp data        ', load_gdp_pc_d()[0].values),
    ('gva data        ', load_gva_pc_d()[0].values),
    ('un gdp aust data', load_un_gdp_pc_d()[0]['Australia'].values),
    ('un gdp all data ', load_un_gdp_pc_d()[0].values),
]
summarise_gdp_data(gdp_data)

simulation_results = [
    ('simple normal model           ', simple_australian_simulation     , load_gdp_pc_d()[0].values                , load_gdp_pc_d()[0].index),
    ('correlated sectors model      ', correlated_sectors_sim           , load_gdp_pc_d()[0].values                , load_gdp_pc_d()[0].index),
    ('simple international model    ', simple_internation_simulation    , load_un_gdp_pc_d()[0]['Australia'].values, load_un_gdp_pc_d()[0].index),
    ('shared var international model', international_shared_variance_sim, load_un_gdp_pc_d()[0]['Australia'].values, load_un_gdp_pc_d()[0].index),
]
summarise_simulations(simulation_results)
