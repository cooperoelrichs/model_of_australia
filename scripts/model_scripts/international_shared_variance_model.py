print('Running model.')
shared_variance_international_gdp_model_results = shared_variance_international_gdp_model(
    un_gdp_pc_d[national_gdp_columns].values
)

print('Done.')


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
