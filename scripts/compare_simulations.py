import os
import sys
from itertools import groupby
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


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
    load_gva_pc_d, load_gdp_pc_d, load_un_gdp_pc_d,
    load_imf_gdppc_data, load_imf_gdppc_forecast
)


def print_summary(name, X, axis):
    D = DataLoader.fractional_diff(X, axis=axis)
    print('%s & %.4f & %.4f \\\\' % (name, np.nanmean(D), np.nanstd(D)))

def print_recession_p(name, X, axis):
    p = np.mean(DataLoader.fractional_diff(X, axis=axis) < 0)
    print('%s & %.2f \\\\' % (name, p))

def count_groups(V, length):
    groups = [(k, sum(1 for i in g)) for k,g in groupby(V)]
    count = sum([
        k_count - (length - 1)
        for k, k_count in groups if (k == True and k_count >= length)
    ])
    return count

def print_consecutive_growth(name, X, axis):
    D = DataLoader.fractional_diff(X, axis=axis) < 0
    ps = []
    for length in (2, 5, 10):
        p = [0, 0]
        if axis == 0:
            p[0] = count_groups(D, length)
            p[1] = (D.shape[0] - length + 1)
        elif axis == 1:
            for i in range(D.shape[0]):
                p[0] += count_groups(D[i, :], length)
            p[1] = (D.shape[1] - length + 1) * D.shape[0]
        ps += maybe_less_than_p(p[0] / p[1])
    print('%s & %s%.2f & %s%.2f & %s%.2f \\\\' % tuple([name] + ps))

def print_summary_table(simulations, fn, header_str):
    X_un_gdp_pc, D_un_gdp_pc = load_un_gdp_pc_d()
    print(header_str + ' \\\\\n\\hline')
    fn('Australia - ABS Data', load_gdp_pc_d()[0]['gdp'].values, 0)
    fn('Australia - UN Data', X_un_gdp_pc['Australia'].values, 0)
    fn('All Nations - UN Data', X_un_gdp_pc.values, 1)

    print('\\hline')
    for sim in simulations:
        fn(sim.name, sim.simulate, 1)

def maybe_less_than_p(p):
    if p < 0.005:
        return ['< ', 0.01]
    else:
        return ['', p]

def print_neg_growth(name, X, starter):
    ps = []
    for i in (2, 5, 20):
        p = np.mean((X[:, i-1] - starter) < 0)
        ps += maybe_less_than_p(p)

    print('%s & %s%.2f & %s%.2f & %s%.2f \\\\' % tuple([name] + ps))

def print_neg_growth_table(simulations):
    gdp_pc_starter = load_gdp_pc_d()[0]['gdp'].values[-1]
    X_un_gdp_pc, D_un_gdp_pc = load_un_gdp_pc_d()
    un_gdppc_starter = X_un_gdp_pc['Australia'].values[-1]
    starters = [gdp_pc_starter, gdp_pc_starter, un_gdppc_starter, un_gdppc_starter]

    print('Name & 2 years & 5 years & 20 years \\\\\n\\hline')
    for i, sim in enumerate(simulations):
        print_neg_growth(sim.name, sim.simulate, starters[i])

def maybe_sum(X):
    if X.ndim == 1:
        return X
    elif X.ndim == 2:
        return X.sum(axis=1)
    else:
        raise RuntimeError('%i dimensional arrays are not supported' % X.ndim)


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

containers = np.array([
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
])

for container in containers:
    container.run()


def plot_official_forecast(ax, x, forecast, title):
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='x', labelsize=11)
    ax.plot(x.index, x.values)
    ax.plot(forecast.index, forecast.values, 'k', linewidth=2)

def prediction_cone_comparison_plot_with_official_forecast(
        simulations, data_sets, currency, output_folder,
        official_data, official_forecast
    ):
        fig, axes, cm = PlottingTools.comparison_plot_setup(
            len(simulations) + 1, 'GDP per capita (%s)' % currency
        )
        for i, sim in enumerate(simulations):
            ax = axes[i+1]
            R, dates = data_sets[i]

            num_r = 10
            min_date = dates[-num_r:].min()

            r_filter = dates >= min_date
            PlottingTools.plot_single_prediction_cone(
                fig, ax, cm, R[r_filter], dates[r_filter], sim,
                official_forecast.index.max()
            )

        plot_official_forecast(
            axes[0],
            official_data[official_data.index >= min_date],
            official_forecast,
            'IMF GDP Forecast'
        )

        fig.suptitle('Bayesian Prediction Cones and the IMF Forecast')
        fig.autofmt_xdate()
        plt.savefig(os.path.join(
            output_folder, 'imf-forecast-with-prediction-cones.png'
        ))

def plot_against_official_forecasts(containers, data_sets):
    imf_data = load_imf_gdppc_data()
    imf_forecast = load_imf_gdppc_forecast()
    X, D = load_gdp_pc_d()
    X_un, D_un = load_un_gdp_pc_d()

    prediction_cone_comparison_plot_with_official_forecast(
        containers,
        data_sets,
        'AUD - Chain Volumes',
        os.path.join(settings.OUTPUTS_DIR, 'comparisons'),
        imf_data['gdp_pc'], imf_forecast['gdp_pc']
    )

data_sets = np.array([
    (maybe_sum(data_pair[0].values), data_pair[0].index)
    for _, _, _, data_pair in specs
])


PlottingTools.prediction_cone_comparison_plot(
    containers[[0, 3]],
    data_sets[[0, 3]],
    'AUD - Chain Volumes',
    os.path.join(settings.OUTPUTS_DIR, 'comparisons'),
    max_date=pd.Timestamp('2020-01-01'),
    growth=True
)

exit()

plot_against_official_forecasts(containers[[0, 3]], data_sets[[0, 3]])
print_summary_table(containers, print_consecutive_growth, 'Name & 2 years & 5 years & 10 years')
print_neg_growth_table(containers)
print_summary_table(containers, print_recession_p, 'Name & P(annual growth < 0)')
print_summary_table(containers, print_summary, 'Name & Mean & Std.')

PlottingTools.prediction_cone_comparison_plot(
    containers,
    data_sets,
    'AUD - Chain Volumes',
    os.path.join(settings.OUTPUTS_DIR, 'comparisons'),
)
print('Done.')
