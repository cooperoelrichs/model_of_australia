import os
import sys
sys.path.insert(0, ".")

import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')

from model_of_australia.data_loader import DataLoader
from scripts.load_data import (
    DATA_SPECS,
    load_un_gdp_pc_d
)
from scripts import settings

def summarise_international_gdp_data():
    output_dir = os.path.join(settings.OUTPUTS_DIR, 'un_gdp_pc_data')
    values, deltas = load_un_gdp_pc_d()

    print(os.path.abspath(output_dir))

    example_countries = [
        'Australia', 'Afghanistan', 'United States', 'USSR (Former)'
    ]

    print(values.shape)
    for a in example_countries:
        print(a)

    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 6))
    axes[0].set_title('GDP per capita by Country')
    axes[0].plot(values.index, values[example_countries])
    axes[0].set_ylabel('GDP per capita')
    axes[0].legend(example_countries, loc=4, prop={'size': 10})
    axes[1].plot(deltas.index, deltas[example_countries])
    axes[1].set_ylabel('annual GDP pc delta / GDP pc')
    axes[1].set_xlabel('date')
    # axes[1].legend(example_countries, loc=1, prop={'size': 10})

    # df.plot(
    #     x='date', y=gva_categories,
    #     subplots=False,
    #     title='GVA by Industry',
    #     figsize=(8,6)
    # )
    DataLoader.maybe_make_dir(output_dir)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_summary.png'))

summarise_international_gdp_data()
