import os
import sys
sys.path.insert(0, ".")

import numpy as np

from model_of_australia.data_loader import DataLoader
from scripts.load_data import DATA_SPECS


def load_un_cxr_data():
    spec = DATA_SPECS['un_cxr_data']
    data = DataLoader.read_data_file(spec)

    base_year = data[data['date'] == np.datetime64(spec['base-year-date'])]
    aud_exchange_rate = base_year[spec['aud-column']]
    print(aud_exchange_rate)


load_un_cxr_data()
