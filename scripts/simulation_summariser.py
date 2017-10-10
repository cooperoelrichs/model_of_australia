import os
import sys
sys.path.insert(0, "../")

import numpy as np

from plotting_tools import PlottingTools
from data_loader import DataLoader


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
    for a, S, _, _ in results:
        pnt_summary_stats(a, DataLoader.fractional_diff(S.simulate, axis=1))

    for y in (1, 19):
        print('simulation_results - year %i:' % y)
        for a, S, _, _ in results:
            R_d = DataLoader.fractional_diff(S.simulate, axis=1)
            pnt_summary_stats(a, R_d[:, y-1])

    for a, S, R, dates in results:
        PlottingTools.plot_prediction_cone(a, S, R, dates)
