import os
import sys
sys.path.insert(0, "../")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import DataLoader
from printing_tools import PrintingTools
from data_tools import DataTools
from scripts import settings


DATA_SPECS = DataLoader.load_specs(
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

def load_gva_pc_d():
    gva_data = DataLoader.read_data_file(DATA_SPECS['abs_gva'])

    gva_abs_full_names = set([
        (c, e) for _, b, c, _, e in DATA_SPECS['abs_gva']['gva_data_classifications']
        if b is not None and b is True
    ])
    gva_categories = [a for a, _ in gva_abs_full_names]

    abs_pop = load_abs_pop()

    date_filter = (
        abs_pop['date'] >= gva_data.date.min()
    ) & (abs_pop['date'] <= gva_data.date.max())

    gva_pc_data = gva_data[
        gva_data.columns.difference(['date'])
    ].divide(
        abs_pop['value'][date_filter].values, axis=0
    )
    gva_pc_data['date'] = gva_data['date']
    gva_pc_data = gva_pc_data[15:]

    gva_pc_d = DataLoader.make_fractional_diff(
        gva_pc_data, gva_pc_data.columns.difference(['date'])
    )

    gva_pc_data_by_date = pd.DataFrame(
        data=gva_pc_data[gva_categories].values,
        index=gva_pc_data['date']
    )
    gva_pc_d_by_date = pd.DataFrame(
        data=gva_pc_d[gva_categories].values,
        index=gva_pc_d['date']
    )
    return gva_pc_data_by_date, gva_pc_d_by_date

def load_abs_pop():
    abs_pop1 = DataLoader.read_transposed_data_file(DATA_SPECS['abs_population1'])
    un_pop2 = DataLoader.read_data_file(DATA_SPECS['un_population2'])

    abs_pop1['date'] = abs_pop1['date'].apply(lambda dt: dt.replace(month=6))

    date_filter = pd.to_datetime(
        ['06-%i' % i for i in range(1981, 2017)],
        format='%m-%Y'
    )
    un_pop2_annual = un_pop2[un_pop2['date'].isin(date_filter)]

    abs_pop = abs_pop1[abs_pop1['date'] < un_pop2_annual['date'].min()].append(
        un_pop2_annual, ignore_index=True
    )
    return abs_pop

def load_un_gdp_pc_d():
    un_gdp_pc_data = DataLoader.read_transposed_data_file(
        DATA_SPECS['un_gdppc']
    )
    un_gdp_pc_d = DataLoader.make_fractional_diff(
        un_gdp_pc_data, un_gdp_pc_data.columns.difference(['date'])
    )
    national_gdp_columns = un_gdp_pc_d.columns.difference(['date'])
    return un_gdp_pc_d[national_gdp_columns]

def load_all():
    datas = {}
    for name in ['rba_gdp', 'rba_cash_rate', 'abs_cpi', 'abs_gdp']:
        datas[name] = DataLoader.read_data_file(specs[name])
        DataLoader.data_summary(
            datas[name],
            os.path.join(settings.OUTPUTS_DIR, specs[name]['name']),
            title=specs[name]['name']
        )

    gva_data = DataLoader.read_data_file(specs['abs_gva'])
    DataLoader.data_summary(
        gva_data,
        os.path.join(settings.OUTPUTS_DIR, specs['abs_gva']['name']),
        title=specs['abs_gva']['name']
    )
    gva_abs_full_names = set([
        (c, e) for _, b, c, _, e in specs['abs_gva']['gva_data_classifications']
        if b is not None and b is True
    ])
    gva_abs_categories = set([
        (a, e) for a, b, _, _, e in specs['abs_gva']['gva_data_classifications']
        if a is not None and b is True
    ])
    gva_categories = [a for a, _ in gva_abs_full_names]

    example_countries = [
        'Afghanistan', 'Australia', 'United States', 'USSR (Former)'
    ]

    DataLoader.data_summary(
        un_gdp_pc_data,
        os.path.join(settings.OUTPUTS_DIR, specs['un_gdppc']['name']),
        fields=example_countries,
        title=specs['un_gdppc']['name']
    )


    # abs_pop.plot(x='date', y='value')
    DataLoader.data_summary(
        abs_pop,
        os.path.join(settings.OUTPUTS_DIR, specs['abs_population1']['name']),
        fields=['value'],
        title=specs['abs_population1']['name']
    )

    gva_pc_ld = DataLoader.make_log_diff(
        gva_pc_data, gva_pc_data.columns.difference(['date'])
    )

    un_gdp_pc_ld = DataLoader.make_log_diff(
        un_gdp_pc_data, un_gdp_pc_data.columns.difference(['date'])
    )

    DataLoader.data_summary(
        un_gdp_pc_d,
        os.path.join(settings.OUTPUTS_DIR, 'un_gdp_pc_d'),
        fields=example_countries,
        title='un_gdp_pc_d'
    )
    DataLoader.data_summary(
        gva_pc_d,
        os.path.join(settings.OUTPUTS_DIR, 'gva_pc_d'),
        fields=['gdp'],
        title='gva_pc_d'
    )

    gva_pc_data['annual gdp delta / gdp'] = np.nan
    gva_pc_data['annual gdp delta / gdp'][1:] = DataLoader.fractional_diff(
        gva_pc_data['gdp'].values, axis=0
    )
    # plt.figure()
    # ax = plt.axes()
    DataLoader.data_summary(
        gva_pc_data,
        os.path.join(settings.OUTPUTS_DIR, 'gva_pc_data'),
        fields=['gdp', 'annual gdp delta / gdp'],
        title='gva_pc_data'
    )

    quarterly_cpi = datas['abs_cpi'].iloc[3::4]
    date_range = [pd.Timestamp(d) for d in ('1960-06-01', '2016-06-01')]
    deflated_abs_gdp = DataTools.deflate(
        datas['abs_gdp'], quarterly_cpi,
        datas['abs_gdp']['date'].max(), # pd.Timestamp('2014-06-01'),
        date_range, ('value', 'deflated-2014-06-01')
    )

    # GDP data sources.
    plt.plot(datas['rba_gdp'], datas['rba_gdp']['value'] * 4, label='rba gdp *4')
    plt.plot(datas['abs_gdp']['date'], datas['abs_gdp']['value'], label='abs gdp')
    plt.plot(gva_data['date'], gva_data['gdp'], label='abs gva gdp')
    plt.plot(
        datas['abs_gdp']['date'],
        deflated_abs_gdp['deflated-2014-06-01'],
        label='deflated abs gdp'
    )
    plt.legend(loc=0)
    DataLoader.maybe_make_dir(os.path.join(settings.OUTPUTS_DIR, 'general'))
    plt.savefig(os.path.join(settings.OUTPUTS_DIR, 'general', 'GDP data sources.png'))

    # GDP and sum of GVA
    plt.plot(gva_data['date'], gva_data['gdp'])
    plt.plot(gva_data['date'], gva_data[gva_categories].values.sum(axis=1))
    plt.savefig(os.path.join(settings.OUTPUTS_DIR, 'general', 'GDP and sum of GVA.png'))

    gva_ld = DataLoader.make_log_diff(gva_data, ['gdp'] + gva_categories)
    gva_d = DataLoader.make_fractional_diff(gva_data, ['gdp'] + gva_categories)

    print('Post 2006.')
    gva_d_date_filter = gva_d['date'] >= pd.Timestamp('2006-06-01')
    gva_date_filter = gva_data['date'] >= pd.Timestamp('2006-06-01')

    for a in ['gdp'] + gva_categories:
        d = gva_d[a][gva_d_date_filter]
        print('mean=%.3f, sd=%.3f - %s' % (d.mean(), d.std(), a))

    # GVA.
    plt.figure()
    ax = plt.axes()
    # ax.plot(gva[date_filter]['date'], gva[date_filter]['gdp'], color='k', linewidth=2)
    gva_data[gva_date_filter].plot(
        x='date', y=gva_categories, legend=False, ax=ax, linewidth=1
    )
    plt.savefig(os.path.join(settings.OUTPUTS_DIR, 'general', 'GVA.png'))

    print('\nGVA growth.')
    plt.figure()
    ax = plt.axes()
    ax.plot(
        gva_d[gva_d_date_filter]['date'],
        gva_d[gva_d_date_filter]['gdp'],
        color='k', linewidth=2
    )
    gva_d[gva_d_date_filter].plot(
        x='date', y=gva_categories, legend=False, ax=ax, linewidth=0.6
    )
    plt.ylim((-0.05, 0.11))
    plt.savefig(os.path.join(settings.OUTPUTS_DIR, 'general', 'GVA growth.png'))


    PrintingTools.summarise_distributions(
        gva_d, 'gdp', gva_categories, (-0.2, 0.3),
        os.path.join(settings.OUTPUTS_DIR, 'general')
    )

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


    PrintingTools.summarise_distributions(
        un_gdp_pc_d,
        'Australia',
        example_countries,
        (-0.1, 0.2),
        os.path.join(settings.OUTPUTS_DIR, 'general')
    )
