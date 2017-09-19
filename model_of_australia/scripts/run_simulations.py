import os
import sys
sys.path.insert(0, "../")
from scipy import stats

from model_container import ModelContainer
from simulation_container import SimulationContainer
from scripts.load_data import load_gva_pc_d, load_un_gdp_pc_d
from models import (
    correlated_sectors_model_fn,
    international_shared_variance_model_fn,
    internationally_influenced_australian_model_fn,
    simple_international_model_fn,
    simple_australian_model_fn,
)
from simulators import (
    GDPSimulatorWithCorrelatedSectors,
    SharedVarianceInternationalGDPSimulator
)


correlated_sectors_model = ModelContainer(
    name='Correlated Sectors Model',
    folder='correlated_sectors_model',
    model_fn=correlated_sectors_model_fn,
    data=load_gva_pc_d()[1].values,
    parameter_spec=[(0, 'mu_eco', stats.norm), (1, 'SD_gva', stats.invgamma)]
)
correlated_sectors_model.run()

correlated_sectors_sim = SimulationContainer(
    name='Correlated Sectors Simulation',
    folder='correlated_sectors_model',
    simulator=GDPSimulatorWithCorrelatedSectors,
    values_deltas_pair=load_gva_pc_d(),
    parameters=correlated_sectors_model.parameters,
    n_years=20, n_iter=1e4
)
correlated_sectors_sim.simulate()


international_shared_variance_model = ModelContainer(
    name='International Shared Variance Model',
    folder='international_shared_variance_model',
    model_fn=international_shared_variance_model_fn,
    data=load_un_gdp_pc_d().values,
    parameter_spec=[(1, 'mu', stats.norm), (0, 'sd', stats.invgamma)]
)

X, D = load_gva_pc_d()
international_shared_variance_sim = SimulationContainer(
    name='International Shared Variance Simulation',
    folder='international_shared_variance_model',
    simulator=SharedVarianceInternationalGDPSimulator,
    values_deltas_pair=(X['Australia'], D['Australia']),
    parameters=international_shared_variance_model.parameters,
    n_years=20, n_iter=1e4
)
correlated_sectors_sim.simulate()

raise RuntimeError('...')


distributions = {
    'normal': {'fit': stats.norm, 'sim':np.random.normal},
    # 'laplace': {'fit': stats.laplace, 'sim':np.random.laplace},
    # 'cauchy': {'fit': stats.cauchy, 'sim':stats.cauchy.rvs}
}

n_years = 20
n_iter = 1e4
simple_gdp_simulation = GDPSimulator.run(
    gva_pc_d['gdp'].values,
    gva_pc['gdp'].values,
    {'fit': stats.norm, 'sim':np.random.normal}
)

fig, ax = plt.subplots(figsize=(4, 3))
ax.set_ylabel('GDP (AUD) per capita')
plt.plot(gva_pc['date'], gva_pc['gdp'])
date_range = pd.date_range(gva_pc['date'].max(), periods=n_years, freq='BAS')
plt.plot(date_range, simple_gdp_simulation.T)
plt.show()



n_years = 20
n_iter = 1e4
simple_internation_gdp_sim = SimpleInternationalGDPSimulator.run(
    un_gdp_pc['Australia'].values,
    shared_variance_international_gdp_model_parameters,
    n_years, n_iter
)


subset = un_gdp_pc.columns.difference(['date'])

fig = plt.figure(figsize=(20, 10))
plt.plot(un_gdp_pc['date'], un_gdp_pc[subset])  # un_gdp_pc['Australia'])
date_range = pd.date_range(un_gdp_pc['date'].max() + pd.Timedelta(1, 'Y'), periods=n_years, freq='BAS')
plt.plot(date_range, simple_internation_gdp_sim.T)
plt.show()

print('Done.')
