import os
import sys
sys.path.insert(0, ".")
from scipy import stats
import numpy as np

from model_of_australia.simulation_container import SimulationContainer
from model_of_australia.data_loader import DataLoader
from model_of_australia.plotting_tools import PlottingTools
from model_of_australia.simulators import (
    GDPSimulatorWithCorrelatedSectors,
    SharedVarianceInternationalGDPSimulator,
    SimpleSimulator,
    CommonDistrubutionSimulator
)
from scripts.fit_models import (
    fit_international_shared_variance_model,
    fit_correlated_sectors_model,
    fit_simple_australian_model,
    fit_simple_international_model
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

for container in containers:
    container.run()

def print_summary(name, X, axis):
    D = DataLoader.fractional_diff(X, axis=axis)
    print('%s & %.4f & %.4f \\\\' % (name, np.nanmean(D), np.nanstd(D)))

def print_tex_summary_table(simulations):
    X_un_gdp_pc, D_un_gdp_pc = load_un_gdp_pc_d()
    print('Name & Mean & Std. \\\\\n\\hline')
    print_summary('Australia - ABS Data', load_gdp_pc_d()[0]['gdp'].values, 0)
    print_summary('Australia - UN Data', X_un_gdp_pc['Australia'].values, 0)
    print_summary('All Nations - UN Data', X_un_gdp_pc.values, 0)
    print('\\hline')

    for sim in simulations:
        print_summary(sim.name, sim.simulate, 1)

def maybe_sum(X):
    if X.ndim == 1:
        return X
    elif X.ndim == 2:
        return X.sum(axis=1)
    else:
        raise RuntimeError('%i dimensional arrays are not supported' % X.ndim)

# print_tex_summary_table(containers)
data_sets = [
    (maybe_sum(data_pair[0].values), data_pair[0].index)
    for _, _, _, data_pair in specs
]

PlottingTools.prediction_cone_comparison_plot(
    containers,
    data_sets,
    'AUD - Chain Volumes',
    os.path.join(settings.OUTPUTS_DIR, 'comparisons')
)
print('Done.')
