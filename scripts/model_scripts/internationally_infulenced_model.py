# priors = {
#     'mu': (0.01958, 0.00043),
#     'sd': (1035.71509, 31.58431),
#     'nu': (0.38714, 0.07650),
# }

print('Running model.')
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
