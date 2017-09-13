print('Running model.')
simple_model_results = simple_australian_gdp_growth_model(
    un_gdp_pc_d['Australia'].values
)
print('Done.')


names = [
    # (0, 'y', stats.t),
    (0, 'mu', stats.norm), (0, 'sd', stats.invgamma),
    # (0, 'nu', stats.expon)
]
summarise(simple_model_results, names, burn=1e4, dic=False)

simple_model_parameters = beta_extracter_version2(simple_model_results[1], names)
print(simple_model_parameters)
