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
    distributions['normal'], n_years, n_iter
)

fig, ax = plt.subplots(figsize=(4, 3))
ax.set_ylabel('GDP (AUD) per capita')
plt.plot(gva_pc['date'], gva_pc['gdp'])
date_range = pd.date_range(gva_pc['date'].max(), periods=n_years, freq='BAS')
plt.plot(date_range, simple_gdp_simulation.T)
plt.show()
