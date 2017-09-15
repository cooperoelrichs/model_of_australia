import sys
sys.path.insert(0, "../")

from models import correlated_sectors_model
from scripts.load_data import load_gva_pc_d

class ModelContainer(object):
    def __init__(self, name, model_fn, data, parameter_spec):
        self.name = name
        self.model_fn = model_fn
        self.data = data
        self.parameter_spec = parameter_spec

    def run(self):
        self.fit_model()
        self.summarise()
        self.calculate_parameters()
        print('Done.')

    def fit_model(self):
        self.results = self.model_fn(self.data)

    def summarise(self):
        summarise(
            self.results, self.parameter_spec,
            burn=2e4, dic=False, gelman_rubin_mean_only=True
        )

    def calculate_parameters(self):
        self.parameters = beta_extracter_version2(
            self.results[1],
            self.parameter_spec
        )

correlated_sectors_model = ModelContainer(
    name='Correlated Sectors Model',
    model_fn=correlated_sectors_model,
    data=load_gva_pc_d().values,
    parameter_spec=[
        (0, 'mu_eco', stats.norm),
        (1, 'SD_gva', stats.invgamma),
    ]

)

correlated_sectors_model.fit_model()

raise RuntimeError('Up to here.')

print('Running International Shared Variance Model.')
shared_variance_international_gdp_model_results = shared_variance_international_gdp_model(
    un_gdp_pc_d[national_gdp_columns].values
)

names = (
    [(1, 'mu', stats.norm), (0, 'sd', stats.invgamma)]
)
# summarise(
#     shared_variance_international_gdp_model_results,
#     names, burn=2e4, dic=False, gelman_rubin_mean_only=True
# )

shared_variance_international_gdp_model_parameters = beta_extracter_version2(
    shared_variance_international_gdp_model_results[1],
    names
)
mu = shared_variance_international_gdp_model_parameters['mu']
shared_variance_international_gdp_model_parameters['mu_australia'] = (
    [mu[np.array(range(len(mu)))[np.where(un_gdp_pc_d[national_gdp_columns].columns == 'Australia')[0][0]]]]
)
print(shared_variance_international_gdp_model_parameters['mu_australia'])
print('Done.')


# priors = {
#     'mu': (0.01958, 0.00043),
#     'sd': (1035.71509, 31.58431),
#     'nu': (0.38714, 0.07650),
# }

print('Running Internationally Infulenced Model.')
influenced_model_results = internationally_influenced_australian_gdp_growth_model(
    un_gdp_pc_d['Australia'].values,
    simple_international_model_parameters
)

names = [
    # (0, 'y', stats.norm),
    (0, 'mu', stats.norm), (0, 'sd', stats.invgamma),
    # (0, 'nu', stats.expon)
]
summarise(influenced_model_results, names, burn=1e4, dic=False)
print('Done.')


print('Running Simple Australian Model.')
simple_model_results = simple_australian_gdp_growth_model(
    un_gdp_pc_d['Australia'].values
)

names = [
    # (0, 'y', stats.t),
    (0, 'mu', stats.norm), (0, 'sd', stats.invgamma),
    # (0, 'nu', stats.expon)
]
summarise(simple_model_results, names, burn=1e4, dic=False)

simple_model_parameters = beta_extracter_version2(simple_model_results[1], names)
print(simple_model_parameters)
print('Done.')


national_gdp_columns = un_gdp_pc_d.columns.difference(['date'])
# columns_without_null = pd.isnull(un_gdp_pc_d[national_gdp_columns].values).sum(axis=0)==0
# national_gdp_columns = un_gdp_pc_d[national_gdp_columns].columns[columns_without_null]

# print('Checking that every country has some data.')
# for a in data_columns:
#     x = un_gdp_pc_d[a][np.isfinite(un_gdp_pc_d[a])].shape
#     if not (len(x) > 0):
#         raise ValueError('Empty: %s, %s' % (a, str(x.shape)))

print('Running Simple International Model.')
simple_international_model_results = simple_international_gdp_model(
    un_gdp_pc_d[national_gdp_columns].values,
    # un_gdp_pc_d['Australia'].values,
    filter_nans=True
)

names = (
    # [(0, 'y%i' % i, stats.t) for i in range(X.shape[1])] +
    [(0, 'mu', stats.norm), (0, 'sd', stats.invgamma)]  # , (0, 'nu', stats.expon)
)
summarise(simple_international_model_results, names, burn=2e4, dic=False, gelman_rubin_mean_only=True)

simple_international_model_parameters = beta_extracter_version2(simple_international_model_results[1], names)
print(simple_international_model_parameters)
print('Done.')
