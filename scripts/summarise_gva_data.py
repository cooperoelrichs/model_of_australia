import os
import sys
sys.path.insert(0, ".")

import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')

from model_of_australia.data_loader import DataLoader
from scripts.load_data import (
    DATA_SPECS,
    load_all_gva_data, get_gva_category_names
)
from scripts import settings

def summarise_gva_data():
    output_dir = os.path.join(settings.OUTPUTS_DIR, 'gva_pc_data')

    values, deltas, _ = load_all_gva_data()
    gva_categories = get_gva_category_names()

    print(len(gva_categories))
    for a in gva_categories:
        print(a)

    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 6))
    axes[0].set_title('GVA per capita by Industry')
    axes[0].plot(values['date'], values[gva_categories])
    axes[0].set_ylabel('GVA per capita')
    axes[1].plot(deltas['date'], deltas[gva_categories])
    axes[1].set_ylabel('annual GVA pc delta / GVA pc')
    axes[1].set_xlabel('date')

    # df.plot(
    #     x='date', y=gva_categories,
    #     subplots=False,
    #     title='GVA by Industry',
    #     figsize=(8,6)
    # )
    DataLoader.maybe_make_dir(output_dir)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_summary.png'))

summarise_gva_data()
