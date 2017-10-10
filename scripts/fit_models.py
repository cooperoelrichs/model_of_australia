import os
import sys
sys.path.insert(0, ".")
import numpy as np
from scipy import stats

from model_of_australia.model_container import ModelContainer
from model_of_australia.models import (
    correlated_sectors_model_fn,
    international_shared_variance_model_fn,
    internationally_influenced_australian_model_fn,
    simple_international_model_fn,
    simple_australian_model_fn,
    ModelPostProcessor, ModelSummariser
)

from scripts.load_data import load_gva_pc_d, load_un_gdp_pc_d
from scripts import settings

def fit_simple_australian_model():
    simple_australian_model = ModelContainer(
        name='Simple Australian Model',
        folder='simple_australian_model',
        model_fn=simple_australian_model_fn,
        data=load_un_gdp_pc_d()[1]['Australia'].values,
        parameter_spec=[(0, 'd', stats.norm)],
        outputs_dir=settings.OUTPUTS_DIR
    )
    simple_australian_model.run()
    return simple_australian_model

def fit_correlated_sectors_model():
    correlated_sectors_model = ModelContainer(
        name='Correlated Sectors Model',
        folder='correlated_sectors_model',
        model_fn=correlated_sectors_model_fn,
        data=load_gva_pc_d()[1].values,
        parameter_spec=[(0, 'mu_eco', stats.norm), (1, 'SD_gva', stats.invgamma)],
        outputs_dir=settings.OUTPUTS_DIR
    )
    correlated_sectors_model.run()
    return correlated_sectors_model

def fit_international_shared_variance_model():
    international_shared_variance_model = ModelContainer(
        name='International Shared Variance Model',
        folder='international_shared_variance_model',
        model_fn=international_shared_variance_model_fn,
        data=load_un_gdp_pc_d()[1].values,
        parameter_spec=[(1, 'mu', stats.norm), (0, 'sd', stats.invgamma)],
        outputs_dir=settings.OUTPUTS_DIR
    )
    international_shared_variance_model.run()
    parameters = extract_mu_for_australia(
        international_shared_variance_model.parameters,
        load_un_gdp_pc_d()[1],
    )
    ModelContainer.save_parameters(
        parameters, international_shared_variance_model.folder,
        settings.OUTPUTS_DIR
    )
    international_shared_variance_model.parameters = parameters
    return international_shared_variance_model

def extract_mu_for_australia(parameters, data):
    parameters['mu_australia'] = [
        parameters['mu'][
            np.array(range(len(parameters['mu'])))[
                np.where(data.columns == 'Australia')[0][0]
            ]
        ]
    ]
    return parameters

def fit_simple_international_model():
    simple_international_model = ModelContainer(
        name='Simple International Model',
        folder='simple_international_model',
        model_fn=simple_international_model_fn,
        data=load_un_gdp_pc_d()[1].values,
        parameter_spec=[(0, 'mu', stats.norm), (0, 'sd', stats.invgamma)],
        outputs_dir=settings.OUTPUTS_DIR
    )
    simple_international_model.run()
    return simple_international_model

def fit_internationally_influenced_australian_model():
    internationally_influenced_australian_model = ModelContainer(
        name='Internationally Influenced Model',
        folder='internationally_influenced_model',
        model_fn=internationally_influenced_australian_model_fn,
        data=load_un_gdp_pc_d()[1]['Australia'].values,
        parameter_spec=[(0, 'mu', stats.norm), (0, 'sd', stats.invgamma)],
        fn_args={'priors':simple_international_model.parameters},
        outputs_dir=settings.OUTPUTS_DIR
    )
    internationally_influenced_australian_model.run()
    return internationally_influenced_australian_model
