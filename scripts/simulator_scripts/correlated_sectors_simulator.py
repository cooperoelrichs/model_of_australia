n_years = 20
n_iter = 1e4

print('Running simulation.')
correlated_sectors_gdp_sim = GDPSimulatorWithCorrelatedSectors.run(
    gva_pc[gva_categories].values,
    correlated_sectors_model_parameters,
    n_years, n_iter
)

plt.figure(figsize=(20, 10))
plt.plot(gva_pc['date'], gva_pc['gdp'])
date_range = pd.date_range(gva_pc['date'].max(), periods=n_years, freq='BAS')
plt.plot(date_range, correlated_sectors_gdp_sim.T)
plt.show()

print('Done.')
