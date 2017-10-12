import os
import sys
sys.path.insert(0, ".")
from scipy import stats
import numpy as np

from model_of_australia.simulation_container import SimulationContainer
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


X_un_gdp_pc, D_un_gdp_pc = load_un_gdp_pc_d()
specs = [
    (
        'simple_australian',
        'Simple Australian Simulation',
        SimpleSimulator,
        load_gdp_pc_d()
    ),
    (
        'correlated_sectors',
        'Correlated Sectors Simulation',
        GDPSimulatorWithCorrelatedSectors,
        load_gva_pc_d()
    ),
    (
        'simple_international',
        'Simple International Simulation',
        CommonDistrubutionSimulator,
        (X_un_gdp_pc['Australia'], D_un_gdp_pc['Australia'])
    ),
    (
        'international_shared_variance',
        'International Shared Variance Simulation',
        SharedVarianceInternationalGDPSimulator,
        (X_un_gdp_pc['Australia'], D_un_gdp_pc['Australia'])
    ),
]

containers = [
    SimulationContainer(
        name=title,
        folder=name + '_model',
        outputs_dir=settings.OUTPUTS_DIR,
        simulator=simulator,
        values_deltas_pair=data_pair,
        load_parameters=True,
        n_years=settings.N_YEARS, n_iter=settings.N_ITER
    )
    for name, title, simulator, data_pair in specs
]

simulations = [
    (container.name, container.run())
    for container in containers
]

def print_summary(name, X, axis):
    D = DataLoader.fractional_diff(X, axis=axis)
    print('%s & %.4f & %.4f \\\\' % (name, np.nanmean(D), np.nanstd(D)))

print('Name & Mean & Std. \\\\\n\\hline')
print_summary('Australia - ABS Data', load_gdp_pc_d()[0]['gdp'].values, 0)
print_summary('Australia - UN Data', X_un_gdp_pc['Australia'].values, 0)
print_summary('All Nations - UN Data', X_un_gdp_pc.values, 0)
print('\\hline')

# max_n = max([len(t) for t, _ in simulations])
for name, X in simulations:
    # s = ' ' * (max_n - len(name))
    print_summary(name, X, 1)

print('Done.')
