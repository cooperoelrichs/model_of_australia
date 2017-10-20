import os
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model_of_australia.data_loader import DataLoader
from model_of_australia.printing_tools import PrintingTools
from model_of_australia.data_tools import DataTools


DATA_SPECS = DataLoader.load_specs(
    'data_specs',
    {
        'abs_cpi': 'abs_cpi.json',
        'abs_gdp': 'abs_gdp.json',
        'abs_gva': 'abs_gva.json',
        'abs_population1': 'abs_population1.json',
        'rba_cash_rate': 'rba_cash_rate.json',
        'rba_gdp': 'rba_gdp.json',
        'un_gdppc': 'un_gdppc.json',
        'un_population2': 'un_population2.json',
        'un_cxr_data': 'un_cxr.json',
        'imf_gdp_pc': 'imf_gdp_pc.json',
        'imf_gdp_pc_forecast': 'imf_gdp_pc_forecast.json',
    }
)

def make_date_indexed(df):
    value_columns = df.columns.difference(['date'])
    return pd.DataFrame(
        data=df[value_columns].values,
        index=df['date'],
        columns=df[value_columns].columns
    )

def load_imf_gdppc_data():
    return load_imf_data(DATA_SPECS['imf_gdp_pc'])

def load_imf_gdppc_forecast():
    return load_imf_data(DATA_SPECS['imf_gdp_pc_forecast'])

def load_imf_data(spec):
    data = DataLoader.read_data_file(spec)
    data = make_date_indexed(data)
    return data

def load_gva_pc_d():
    gva_pc_data, gva_pc_d, gva_categories = load_all_gva_data()
    gva_pc_data_by_date = make_date_indexed(gva_pc_data[gva_categories + ['date']])
    gva_pc_d_by_date = make_date_indexed(gva_pc_d[gva_categories + ['date']])
    return gva_pc_data_by_date, gva_pc_d_by_date

def load_gdp_pc_d():
    gva_pc_data, gva_pc_d, _ = load_all_gva_data()
    gdp_pc_data_by_date = make_date_indexed(gva_pc_data[['gdp', 'date']])
    gdp_pc_d_by_date = make_date_indexed(gva_pc_d[['gdp', 'date']])
    return gdp_pc_data_by_date, gdp_pc_d_by_date

def get_gva_category_names():
    gva_abs_full_names = set([
        (c, e) for _, b, c, _, e in DATA_SPECS['abs_gva']['gva_data_classifications']
        if b is not None and b is True
    ])
    gva_categories = [a for a, _ in gva_abs_full_names]
    return gva_categories

def load_all_gva_data():
    gva_data = DataLoader.read_data_file(DATA_SPECS['abs_gva'])
    gva_categories = get_gva_category_names()

    abs_pop = load_abs_pop()

    date_filter = (
        abs_pop['date'] >= gva_data.date.min()
    ) & (abs_pop['date'] <= gva_data.date.max())

    gva_pc_data = gva_data[
        gva_data.columns.difference(['date'])
    ].divide(
        abs_pop['value'][date_filter].values,
        axis=0
    )

    gva_pc_data['date'] = gva_data['date']
    gva_pc_data = gva_pc_data[15:]

    gva_pc_d = DataLoader.make_fractional_diff(
        gva_pc_data, gva_pc_data.columns.difference(['date'])
    )
    return gva_pc_data, gva_pc_d, gva_categories

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

    DataLoader.convert_to_aud(
        un_gdp_pc_data, DATA_SPECS['un_gdppc']
    )

    DataLoader.convert_chain_volumes_to_constant_prices(
        un_gdp_pc_data, DATA_SPECS['un_gdppc']
    )

    un_gdp_pc_d = DataLoader.make_fractional_diff(
        un_gdp_pc_data, un_gdp_pc_data.columns.difference(['date'])
    )
    national_gdp_columns = un_gdp_pc_d.columns.difference(['date'])

    un_gdp_pc_data_by_date = make_date_indexed(
        un_gdp_pc_data[national_gdp_columns | ['date']]
    )
    un_gdp_pc_d_by_date = make_date_indexed(
        un_gdp_pc_d[national_gdp_columns | ['date']]
    )
    return un_gdp_pc_data_by_date, un_gdp_pc_d_by_date
