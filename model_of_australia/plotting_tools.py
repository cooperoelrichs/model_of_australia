import os

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

from scripts import settings


class PlottingTools():
    def density_plot(y, lims, color=None, linewidth=1):
        density = stats.gaussian_kde(y)
        x = np.linspace(lims[0], lims[1] , 1e5)
        # plt.plot(x, density(x), color=color, linewidth=linewidth)

    def pdf_plot(d, params, lims, color=None, linewidth=1):
        x = np.linspace(lims[0], lims[1], int(1e5))
        # plt.plot(x, d.pdf(x, *params), color=color, linewidth=linewidth)

    def add_legend(fig, ax, cm, quantiles):
        patches = [
            matplotlib.patches.Patch(
                color=cm((1-q)),
                label='P = %.2f' % q
            )
            for q in quantiles
        ]
        lg = plt.legend(
            handles=patches, loc=2, framealpha=1, frameon=True
        )  # facecolor='white'
        lg.get_frame().set_facecolor('white')

    def plot_prediction_cone(a, S, R, dates):
        last_date = pd.DatetimeIndex(dates).max()
        date_range = pd.to_datetime(
            ['%i-%i-1' % (y, last_date.month)
            for y in range(last_date.year+1, last_date.year+21)]
        )

        mean = S.simulate.mean(axis=0)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_ylabel('GDP (AUD) per capita')
        cm = plt.get_cmap('Blues')
        plt.plot(dates, R)
        # plt.plot(date_range, mean, color='k', lw=1)

        quantiles = [0.99, 0.9, 0.5, 0.1]
        # quantiles = np.linspace(0.98, 0, 50)
        for q in quantiles:
            p = q*100/2
            p1, p2 = 50-p, 50+p

            lower = np.percentile(S.simulate, p1, axis=0)
            upper = np.percentile(S.simulate, p2, axis=0)
            ax.fill_between(date_range, lower, upper, color=cm((1-q)))

            # plt.plot(date_range, lower, color='b')
            # plt.plot(date_range, upper, color='b')

        PlottingTools.add_legend(fig, ax, cm, quantiles)
        plt.title('Bayesian Prediction Cone')
        plt.savefig(os.path.join(
            settings.OUTPUTS_DIR,
            S.folder,
            'prediction cone.png'
        ))
