print('Running model.')
correlated_sectors_model_results = correlated_sectors_model(
    gva_pc_d[gva_categories].values
)

correlated_sectors_parameter_names = [
    (0, 'mu_eco', stats.norm),
    (1, 'SD_gva', stats.invgamma),
    # (1, 'D_gva', stats.norm),
]
summarise(
    correlated_sectors_model_results, correlated_sectors_parameter_names,
    burn=2e4, dic=False, gelman_rubin_mean_only=True
)

correlated_sectors_model_parameters = beta_extracter_version2(
    correlated_sectors_model_results[1],
    correlated_sectors_parameter_names
)
print('Done.')
