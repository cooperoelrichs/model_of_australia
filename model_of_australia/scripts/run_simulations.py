import os
import sys
sys.path.insert(0, "../")
from scipy import stats

from simulation_container import SimulationContainer
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
N_ITER = 1e2


correlated_sectors_model = fit_correlated_sectors_model()
correlated_sectors_sim = SimulationContainer(
    name='Correlated Sectors Simulation',
    folder='correlated_sectors_model',
    simulator=GDPSimulatorWithCorrelatedSectors,
    values_deltas_pair=load_gva_pc_d(),
    parameters=correlated_sectors_model.parameters,
    n_years=20, n_iter=1e4
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
