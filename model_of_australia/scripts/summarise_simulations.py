def pnt_summary_stats(a, r):
    print(' - %s - mean=%.4f, sd=%.5f, min=%.4f, max=%.4f' % (
        a, np.nanmean(r), np.nanstd(r), np.nanmin(r), np.nanmax(r)
    ))

gdp_data = [
    ('gdp data                      ', gva_pc['gdp'].values),
    ('gva data                      ', gva_pc[gva_categories].values),
    ('un gdp aust data              ', un_gdp_pc['Australia'].values),
    ('un gdp all data               ', un_gdp_pc[national_gdp_columns].values),
]
simulation_results = [
    ('simple normal model           ', simple_gdp_simulation                , gva_pc['gdp'].values         , gva_pc['date']),
    ('correlated sectors model      ', correlated_sectors_gdp_sim           , gva_pc['gdp'].values         , gva_pc['date']),
    ('simple international model    ', simple_internation_gdp_sim           , un_gdp_pc['Australia'].values, un_gdp_pc['date']),
    ('shared var international model', shared_variance_international_gdp_sim, un_gdp_pc['Australia'].values, un_gdp_pc['date']),
]

print(correlated_sectors_model_parameters['mu_eco'])
print('%0.7f' % np.mean([a[0] for _, a, _ in correlated_sectors_model_parameters['mu_eco']]))

print('gdp_data - all years:')
for a, R in gdp_data:
    pnt_summary_stats(a, fractional_diff(R, axis=0))

print('simulation_results - all years:')
for a, R, _, _ in simulation_results:
    pnt_summary_stats(a, fractional_diff(R, axis=1))

for y in (1, 19):
    print('%iy:' % y)
    for a, R, _, _ in simulation_results:
        R_d = fractional_diff(R, axis=1)
        pnt_summary_stats(a, R_d[:, y-1])

for a, S, R, dates in simulation_results:
    print('Prediction cone for %s' % a)

    last_date = pd.DatetimeIndex(dates).max()
    date_range = pd.to_datetime(['%i-%i-1' % (y, last_date.month) for y in range(last_date.year+1, last_date.year+21)])

    mean = S.mean(axis=0)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_ylabel('GDP (AUD) per capita')
    cm = plt.get_cmap('Blues')
    plt.plot(dates, R)
    # plt.plot(date_range, mean, color='k', lw=1)

    quantiles = [0.99, 0.9, 0.5, 0.1]
    # quantiles = np.linspace(0.98, 0, 50)
    for q in quantiles:
        p = q*100/2
        p1, p2 = 50-p, 50+p

        lower = np.percentile(S, p1, axis=0)
        upper = np.percentile(S, p2, axis=0)
        ax.fill_between(date_range, lower, upper, color=cm((1-q)))

        # plt.plot(date_range, lower, color='b')
        # plt.plot(date_range, upper, color='b')

    add_legend(fig, ax, cm, quantiles)
    plt.show()
