import os
import sys
sys.path.insert(0, ".")

import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')

from model_of_australia.data_loader import DataLoader
from scripts.load_data import (
    DATA_SPECS,
    load_gdp_pc_d, load_un_gdp_pc_d, load_gva_pc_d, get_gva_category_names
)
from scripts import settings


DEFAULT_PLOT_SIZE = (12, 6)


def plot_data(title, ylabels, values, deltas, make_deltas_plot_green):
    fig, axes = plt.subplots(2, sharex=True, figsize=DEFAULT_PLOT_SIZE)
    axes[0].set_title(title)
    axes[0].plot(values.index, values)
    axes[0].set_ylabel(ylabels[0])

    if make_deltas_plot_green:
        axes[1].plot(deltas.index, deltas, 'g')
    else:
        axes[1].plot(deltas.index, deltas)

    axes[1].set_ylabel(ylabels[1])
    axes[1].set_xlabel('date')
    return fig, axes


def save_fig(name):
    output_dir = os.path.join(settings.OUTPUTS_DIR, name)
    DataLoader.maybe_make_dir(output_dir)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_summary.png'))
    plt.close()


def summarise_abs_gdp_data():
    values, deltas = load_gdp_pc_d()
    plot_data(
        'Australian GDP Per Capita',
        ('GDP (AUD) per capita', 'Annual fractional change'),
        values, deltas,
        make_deltas_plot_green=True
    )
    save_fig('abs_gdp_pc')


def summarise_international_gdp_data():
    values, deltas = load_un_gdp_pc_d()
    example_countries = [
        'Australia', 'Afghanistan', 'United States', 'USSR (Former)'
    ]

    fig, axes = plot_data(
        'GDP Per Capita by Country',
        ('GDP (AUD) per capita', 'Annual fractional change'),
        values[example_countries], deltas[example_countries],
        make_deltas_plot_green=False
    )
    axes[0].legend(example_countries, loc=4, prop={'size': 10})

    save_fig('un_gdp_pc_data')


def summarise_gva_data():
    values, deltas = load_gva_pc_d()
    gva_categories = get_gva_category_names()

    fig, axes = plot_data(
        'GVA Per Capita by Industry',
        ('GVA (AUD) per capita', 'Annual fractional change'),
        values[gva_categories], deltas[gva_categories],
        make_deltas_plot_green=False
    )

    save_fig('gva_pc_data')


summarise_abs_gdp_data()
summarise_international_gdp_data()
summarise_gva_data()
