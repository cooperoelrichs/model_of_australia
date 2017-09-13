national_gdp_columns = un_gdp_pc_d.columns.difference(['date'])
# columns_without_null = pd.isnull(un_gdp_pc_d[national_gdp_columns].values).sum(axis=0)==0
# national_gdp_columns = un_gdp_pc_d[national_gdp_columns].columns[columns_without_null]

# print('Checking that every country has some data.')
# for a in data_columns:
#     x = un_gdp_pc_d[a][np.isfinite(un_gdp_pc_d[a])].shape
#     if not (len(x) > 0):
#         raise ValueError('Empty: %s, %s' % (a, str(x.shape)))

print('Running model.')
simple_international_model_results = simple_international_gdp_model(
    un_gdp_pc_d[national_gdp_columns].values,
    # un_gdp_pc_d['Australia'].values,
    filter_nans=True
)

print('Done.')


names = (
    # [(0, 'y%i' % i, stats.t) for i in range(X.shape[1])] +
    [(0, 'mu', stats.norm), (0, 'sd', stats.invgamma)]  # , (0, 'nu', stats.expon)
)
summarise(simple_international_model_results, names, burn=2e4, dic=False, gelman_rubin_mean_only=True)

simple_international_model_parameters = beta_extracter_version2(simple_international_model_results[1], names)
print(simple_international_model_parameters)
