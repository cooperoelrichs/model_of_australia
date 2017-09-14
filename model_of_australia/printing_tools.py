import os

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from statsmodels.tsa import stattools
from statsmodels.stats.diagnostic import het_arch

from plotting_tools import PlottingTools


class PrintingTools():
    def to_s(x):
        if x < 0:
            return '%.3f' % x
        else:
            return ' %.3f' % x

    def summarise_distributions(data, main, columns, lims, output_dir):
        columns = list(columns)
        print('\nApproximate v. Normal.')
        for a in [main] + columns:
            x = data[a]
            x = x[pd.notnull(x)]

            # Metrics: Mean; STD; Skewness; Kurtosis; ADF p-value; ARCH-LM p-value.
            # A small p-value (≤ 0.05) indicates strong evidence against the null hypothesis.
            # A large p-value (> 0.05) indicates weak evidence against the null hypothesis
            print('mu=%s, sd=%s, s=%s, k=%s, adf=%s, arch-lm=%s - %s' % (
                PrintingTools.to_s(x.mean()),
                PrintingTools.to_s(x.std()),
                PrintingTools.to_s(stats.skew(x)),
                PrintingTools.to_s(stats.kurtosis(x)),
                PrintingTools.to_s(stattools.adfuller(x)[1]),
                PrintingTools.to_s(het_arch(x)[1]),
                a
            ))

        PlottingTools.pdf_plot(stats.norm, (
            data[main].mean(),
            data[main].std()),
            lims, color='k', linewidth=2
        )

        for a in columns:
            x = data[a]
            PlottingTools.pdf_plot(
                stats.norm, (x.mean(), x.std()), lims, linewidth=0.5
            )

        plt.savefig(os.path.join(output_dir, 'distributions summary - pdf.png'))

        x = data[main]
        x = x[x.notnull()]
        PlottingTools.density_plot(x, lims, color='k', linewidth=2)

        for a in columns:
            x = data[a]
            x = x[x.notnull()]
            PlottingTools.density_plot(x, lims, linewidth=0.5)

        plt.savefig(os.path.join(output_dir, 'distributions summary - density.png'))
