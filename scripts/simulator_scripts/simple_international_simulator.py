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
