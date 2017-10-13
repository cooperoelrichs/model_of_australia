import os

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from scipy import stats


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

    def make_future_date_range(dates, future_years):
        last_date = pd.DatetimeIndex(dates).max()
        return pd.to_datetime(
            ['%i-%i-1' % (y, last_date.month)
            for y in range(last_date.year+1, last_date.year+(future_years+1))]
        )

    def plot_prediction_cone_on_axis(sim, date_range, fig, ax, cm):
        quantiles = [0.99, 0.9, 0.5, 0.1]
        for q in quantiles:
            p = q*100/2
            p1, p2 = 50-p, 50+p

            lower = np.percentile(sim.simulate, p1, axis=0)
            upper = np.percentile(sim.simulate, p2, axis=0)
            ax.fill_between(date_range, lower, upper, color=cm((1-q)))

        PlottingTools.add_legend(fig, ax, cm, quantiles)

    def prediction_cone_cmap():
        return plt.get_cmap('Blues')

    def plot_prediction_cone(S, R, dates, ylabel, outputs_dir):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_ylabel(ylabel)  # 'GDP (AUD) per capita'
        cm = PlottingTools.prediction_cone_cmap()
        plt.plot(dates, R)

        date_range = PlottingTools.make_future_date_range(dates, 20)
        PlottingTools.plot_prediction_cone_on_axis(S, date_range, fig, ax, cm)

        plt.title('Bayesian Prediction Cone')
        plt.savefig(os.path.join(outputs_dir, S.folder, 'prediction cone.png'))

    def prediction_cone_comparison_plot(
        simulations, data_sets, output_folder
    ):
        fig, axes = plt.subplots(1, 4, sharey=True, figsize=(14, 5))
        axes[0].set_ylabel('GDP per capita')
        cm = PlottingTools.prediction_cone_cmap()

        for i, sim in enumerate(simulations):
            ax = axes[i]
            R, dates, currency = data_sets[i]

            num_r = 10
            ax.plot(dates[-num_r:], R[-num_r:])
            ax.set_title('%s (%s)' % (sim.name, currency), fontsize=10)
            ax.tick_params(axis='x', labelsize=11)

            date_range = PlottingTools.make_future_date_range(dates, 20)
            PlottingTools.plot_prediction_cone_on_axis(
                sim, date_range, fig, ax, cm
            )

        fig.suptitle('Bayesian Prediction Cones')
        fig.autofmt_xdate()
        plt.savefig(os.path.join(
            output_folder, 'prediction-cone-comparison.png'
        ))
