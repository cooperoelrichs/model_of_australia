import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('seaborn-whitegrid')

from sklearn.linear_model import LinearRegression

from statsmodels.tsa import stattools
from statsmodels.stats.diagnostic import het_arch



# Model features:
#  - sector forecasts (by year, by industry,
#    income v. expenditure, based on customers,
#    market share, growth declining in the long term.)
#     * Normal, Asymetric Laplace [AL], Normal Mixture, Normal-AL [NAL], convolution of N and AL)
#     * I thing we should try one Normal and one Kurtostic (Cauchy or Laplace) distribution.
#     * Use logarimithic difference for growth
#  - Relationships between economic sectors, i.e. is growth between the sectors correlated?
#  - Risk:
#     * the increased variance model; and
#     * the suvivorship model.
#  - GDP = gdp_national_model * international_gdp_variation_model * risk_of_national_failure_model

# Other things to consider modelling:
#  - tax (% of profit)
#  - inflation
#  - assets and cash
#  - debt
#  - is there a market trend (a 'beta') causing mean variance?
#  - is there a market trend causing variance in the variance?

# Resources:
#  - https://www.diva-portal.org/smash/get/diva2:311764/FULLTEXT01.pdf
