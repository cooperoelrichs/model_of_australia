import os
import sys
sys.path.insert(0, "../")
from scipy import stats

import numpy as np

from simulation_container import SimulationContainer
from plotting_tools import PlottingTools
from data_loader import DataLoader
from scripts.load_data import load_gva_pc_d, load_gdp_pc_d, load_un_gdp_pc_d
from scripts.fit_models import (
    fit_international_shared_variance_model,
    fit_correlated_sectors_model,
    fit_simple_australian_model,
    fit_simple_international_model
)
from simulators import (
    GDPSimulatorWithCorrelatedSectors,
    SharedVarianceInternationalGDPSimulator,
    SimpleSimulator
)


N_YEARS = 20
N_ITER = 1e4


def pnt_summary_stats(a, r):
    print(' - %s - mean=%.4f, sd=%.5f, min=%.4f, max=%.4f' % (
        a, np.nanmean(r), np.nanstd(r), np.nanmin(r), np.nanmax(r)
    ))

def summarise_gdp_data(data):
    print('gdp_data - all years:')
    for a, R in data:
        pnt_summary_stats(a, DataLoader.fractional_diff(R, axis=0))

def summarise_simulations(results):
    print('simulation_results - all years:')
    for a, S, _, _ in simulation_results:
        pnt_summary_stats(a, DataLoader.fractional_diff(S.simulate, axis=1))

    for y in (1, 19):
        print('simulation_results - year %i:' % y)
        for a, S, _, _ in simulation_results:
            R_d = DataLoader.fractional_diff(S.simulate, axis=1)
            pnt_summary_stats(a, R_d[:, y-1])

    for a, S, R, dates in simulation_results:
        PlottingTools.plot_prediction_cone(a, S, R, dates)


correlated_sectors_model = fit_correlated_sectors_model()
correlated_sectors_sim = SimulationContainer(
    name='Correlated Sectors Simulation',
    folder='correlated_sectors_model',
    simulator=GDPSimulatorWithCorrelatedSectors,
    values_deltas_pair=load_gva_pc_d(),
    parameters=correlated_sectors_model.parameters,
    n_years=N_YEARS, n_iter=N_ITER
)
correlated_sectors_sim.simulate()


international_shared_variance_model = fit_international_shared_variance_model()
X, D = load_un_gdp_pc_d()
international_shared_variance_sim = SimulationContainer(
    name='International Shared Variance Simulation',
    folder='international_shared_variance_model',
    simulator=SharedVarianceInternationalGDPSimulator,
    values_deltas_pair=(X['Australia'], D['Australia']),
    parameters=international_shared_variance_model.parameters,
    n_years=N_YEARS, n_iter=N_ITER
)
international_shared_variance_sim.simulate()


simple_australian_model = fit_simple_australian_model()
simple_australian_simulation = SimulationContainer(
    name='Simple Australian Simulation',
    folder='simple_australian_model',
    simulator=SimpleSimulator,
    values_deltas_pair=load_gdp_pc_d(),
    parameters=simple_australian_model.parameters,
    n_years=N_YEARS, n_iter=N_ITER
)
simple_australian_simulation.simulate()


simple_internation_model = fit_simple_international_model()
X, D = load_un_gdp_pc_d()
simple_internation_simulation = SimulationContainer(
    name='Simple Internation Simulation',
    folder='simple_international_model',
    simulator=SimpleSimulator,
    values_deltas_pair=(X['Australia'], D['Australia']),
    parameters=simple_internation_model.parameters,
    n_years=N_YEARS, n_iter=N_ITER
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
    ('simple normal model           ', simple_australian_simulation     , load_gdp_pc_d()[0].values                , gva_pc['date']),
    ('correlated sectors model      ', correlated_sectors_sim           , load_gdp_pc_d()[0].values                , load_gdp_pc_d()[0].index),
    ('simple international model    ', simple_internation_simulation    , load_un_gdp_pc_d()[0]['Australia'].values, un_gdp_pc['date']),
    ('shared var international model', international_shared_variance_sim, load_un_gdp_pc_d()[0]['Australia'].values, un_gdp_pc['date']),
]
summarise_simulations(simulation_results)
