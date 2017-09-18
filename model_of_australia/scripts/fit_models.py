import os
import sys
sys.path.insert(0, "../")

import numpy as np
from scipy import stats

from models import (
    correlated_sectors_model_fn,
    international_shared_variance_model_fn,
    internationally_influenced_australian_model_fn,
    simple_international_model_fn,
    simple_australian_model_fn,
    ModelPostProcessor, ModelSummariser
)
from scripts.load_data import load_gva_pc_d, load_un_gdp_pc_d
from scripts import settings


class ModelContainer(object):
    def __init__(
        self, name, folder, model_fn, data, parameter_spec,
        fn_args={}, iters=1e4, tune_iters=2e3
    ):
        self.name = name
        self.folder = folder
        self.model_fn = model_fn
        self.data = data
        self.parameter_spec = parameter_spec
        self.fn_args = fn_args
        self.iters = iters
        self.tune_iters = tune_iters

    def run(self):
        print('Running %s.' % self.name)
        self.results = self.fit_model()
        self.summarise()
        self.parameters = self.calculate_parameters()
        print('Done.')

    def fit_model(self):
        results = self.model_fn(
            self.data, iters=self.iters, tune_iters=self.tune_iters,
            **self.fn_args
        )
        return results

    def summarise(self):
        ModelSummariser.summarise(
            self.results, self.parameter_spec,
            output_dir=os.path.join(settings.OUTPUTS_DIR, self.folder),
            burn=2e4, dic=False, gelman_rubin_mean_only=True
        )

    def calculate_parameters(self):
        parameters = ModelPostProcessor.beta_extracter_version2(
            self.results[1],
            self.parameter_spec
        )
        return parameters


simple_australian_model = ModelContainer(
    name='Simple Australian Model',
    folder='simple_australian_model',
    model_fn=simple_australian_model_fn,
    data=load_un_gdp_pc_d()['Australia'].values,
    parameter_spec=[(0, 'mu', stats.norm), (0, 'sd', stats.invgamma)]
)
simple_australian_model.run()


correlated_sectors_model = ModelContainer(
    name='Correlated Sectors Model',
    folder='correlated_sectors_model',
    model_fn=correlated_sectors_model_fn,
    data=load_gva_pc_d().values,
    parameter_spec=[(0, 'mu_eco', stats.norm), (1, 'SD_gva', stats.invgamma)]
)
correlated_sectors_model.run()

international_shared_variance_model = ModelContainer(
    name='International Shared Variance Model',
    folder='international_shared_variance_model',
    model_fn=international_shared_variance_model_fn,
    data=load_un_gdp_pc_d().values,
    parameter_spec=[(1, 'mu', stats.norm), (0, 'sd', stats.invgamma)]
)

def extract_mu_for_australia(model, data):
    model.parameters['mu_australia'] = [
        model.parameters['mu'][
            np.array(range(len(model.parameters['mu'])))[
                np.where(data.columns == 'Australia')[0][0]
            ]
        ]
    ]

international_shared_variance_model.run()
extract_mu_for_australia(
    international_shared_variance_model,
    load_un_gdp_pc_d(),
)

simple_international_model = ModelContainer(
    name='Simple International Model',
    folder='simple_international_model',
    model_fn=simple_international_model_fn,
    data=load_un_gdp_pc_d().values,
    parameter_spec=[(0, 'mu', stats.norm), (0, 'sd', stats.invgamma)],
    fn_args={'filter_nans':True}
)
simple_international_model.run()

internationally_influenced_australian_model = ModelContainer(
    name='Internationally Influenced Model',
    folder='internationally_influenced_model',
    model_fn=internationally_influenced_australian_model_fn,
    data=load_un_gdp_pc_d()['Australia'].values,
    parameter_spec=[(0, 'mu', stats.norm), (0, 'sd', stats.invgamma)],
    fn_args={'priors':simple_international_model.parameters}
)
internationally_influenced_australian_model.run()
