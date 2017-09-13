import os
import sys
sys.path.insert(0, "../")

from data_loader import DataLoader


specs = DataLoader.load_specs(
    '../data_specs',
    {
        'abs_cpi': 'abs_cpi.json',
        'abs_gdp': 'abs_gdp.json',
        'abs_gva': 'abs_gva.json',
        'abs_population1': 'abs_population1.json',
        'rba_cash_rate': 'rba_cash_rate.json',
        'rba_gdp': 'rba_gdp.json',
        'un_gdppc': 'un_gdppc.json',
        'un_population2': 'un_population2.json',
    }
)

for name in ['rba_gdp', 'rba_cash_rate', 'abs_cpi', 'abs_gdp']:
    rba_gdp_data = DataLoader.read_data_file(specs[name])
    DataLoader.data_summary(
        rba_gdp_data,
        os.path.join('../../outputs', specs[name]['name'])
    )

gva_data = DataLoader.read_data_file(specs['abs_gva'])
DataLoader.data_summary(
    gva_data,
    os.path.join('../../outputs', specs['abs_gva']['name'])
)
gva_abs_full_names = set([(c, e) for _, b, c, _, e in gva_data_classifications if b is not None and b is True])
gva_abs_categories = set([(a, e) for a, b, _, _, e in gva_data_classifications if a is not None and b is True])
gva_categories = [a for a, _ in gva_abs_full_names]

un_gdp_pc = read_transposed_data_file(un_gdp_pc_data_spec)
un_gdp_pc.plot(x='date', y=['Afghanistan', 'Australia', 'United States', 'USSR (Former)'])
plt.show()

abs_pop1 = read_transposed_data_file(abs_pop1_data_spec)
abs_pop2 = read_data_file(abs_pop2_data_spec)

abs_pop1['date'] = abs_pop1['date'].apply(lambda dt: dt.replace(month=6))

date_filter = pd.to_datetime(
    ['06-%i' % i for i in range(1981, 2017)],
    format='%m-%Y'
)
abs_pop2_annual = abs_pop2[abs_pop2['date'].isin(date_filter)]

abs_pop = abs_pop1[abs_pop1['date'] < abs_pop2['date'].min()].append(
    abs_pop2_annual, ignore_index=True
)

abs_pop.plot(x='date', y='value')
plt.show()


date_filter = (abs_pop['date']>=gva.date.min()) & (abs_pop['date']<=gva.date.max())

gva_pc = gva[gva.columns.difference(['date'])].divide(abs_pop['value'][date_filter].values, axis=0)
gva_pc['date'] = gva['date']
gva_pc = gva_pc[15:]

plt.figure()
ax = plt.axes()
un_gdp_pc.plot(x='date', y=['Afghanistan', 'Australia', 'United States', 'USSR (Former)'], ax=ax)
plt.show()


gva_pc_ld = make_log_diff(gva_pc, gva_pc.columns.difference(['date']))
gva_pc_d = make_fractional_diff(gva_pc, gva_pc.columns.difference(['date']))
un_gdp_pc_ld = make_log_diff(un_gdp_pc, un_gdp_pc.columns.difference(['date']))
un_gdp_pc_d = make_fractional_diff(un_gdp_pc, un_gdp_pc.columns.difference(['date']))

example_countries = ['Afghanistan', 'Australia', 'United States', 'USSR (Former)']

plt.figure()
ax = plt.axes()
un_gdp_pc_d.plot(x='date', y=example_countries, ax=ax)
gva_pc_d.plot(x='date', y='gdp', ax=ax)
plt.show()


gva_pc['annual gdp delta / gdp'] = np.nan
gva_pc['annual gdp delta / gdp'][1:] = fractional_diff(gva_pc['gdp'].values, axis=0)
# plt.figure()
# ax = plt.axes()
gva_pc.plot(
    x='date', y=['gdp', 'annual gdp delta / gdp'],
    subplots=True,
    title='Australian GDP Per Capita', figsize=(10, 5)
)
# ax.patch.set_facecolor('white')
plt.show()


# 1. GDP and CPI forecasts
# 2. GDP by sector

# Need annual gdp data - use abs and deflate - No, use the abs gva gdp data.
# Model the by sector data!

quarterly_cpi = cpi.iloc[3::4]
date_range = [pd.Timestamp(d) for d in ('1960-06-01', '2016-06-01')]
deflated_abs_gdp = deflate(
    abs_gdp, quarterly_cpi,
    abs_gdp['date'].max(), # pd.Timestamp('2014-06-01'),
    date_range, ('value', 'deflated-2014-06-01')
)

print('GDP data sources.')
plt.plot(rba_gdp['date'], rba_gdp['value'] * 4, label='rba gdp *4')
plt.plot(abs_gdp['date'], abs_gdp['value'], label='abs gdp')
plt.plot(gva['date'], gva['gdp'], label='abs gva gdp')
plt.plot(abs_gdp['date'], deflated_abs_gdp['deflated-2014-06-01'], label='deflated abs gdp')
plt.legend(loc=0)
plt.show()

print('GDP and sum of GVA.')
plt.plot(gva['date'], gva['gdp'])
plt.plot(gva['date'], gva[gva_categories].values.sum(axis=1))
plt.show()


gva_ld = make_log_diff(gva, ['gdp'] + gva_categories)
gva_d = make_fractional_diff(gva, ['gdp'] + gva_categories)

print('Post 2006.')
gva_d_date_filter = gva_d['date'] >= pd.Timestamp('2006-06-01')
gva_date_filter = gva['date'] >= pd.Timestamp('2006-06-01')

for a in ['gdp'] + gva_categories:
    d = gva_d[a][gva_d_date_filter]
    print('mean=%.3f, sd=%.3f - %s' % (d.mean(), d.std(), a))

print('\nGVA.')
plt.figure()
ax = plt.axes()
# ax.plot(gva[date_filter]['date'], gva[date_filter]['gdp'], color='k', linewidth=2)
gva[gva_date_filter].plot(x='date', y=gva_categories, legend=False, ax=ax, linewidth=1)
plt.show()

print('\nGVA growth.')
plt.figure()
ax = plt.axes()
ax.plot(gva_d[gva_d_date_filter]['date'], gva_d[gva_d_date_filter]['gdp'], color='k', linewidth=2)
gva_d[gva_d_date_filter].plot(x='date', y=gva_categories, legend=False, ax=ax, linewidth=0.6)
plt.ylim((-0.05, 0.11))
plt.show()


summarise_distributions(gva_d, 'gdp', gva_categories, (-0.2, 0.3))

# print('Approximated v. Laplace v. Cauchy.')
# lims=(-0.1, 0.2)
# for a in ['gdp'] + gva_categories:
#     x = gva_d[a]
#     x = x[x.notnull()]

#     print(a)
#     density_plot(x, lims)
#     for d in (stats.laplace, stats.cauchy, stats.norm):
#         pdf_plot(d, d.fit(x), lims)

#     plt.show()


summarise_distributions(
    un_gdp_pc_d,
    'Australia',
    example_countries,
    (-0.1, 0.2)
)
