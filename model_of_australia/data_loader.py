import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def maybe_make_dir(path):
    os.makedirs(path, exist_ok=True)


class DataLoader():
    def load_specs(dir_path, to_load):
        loaded = {}
        for name, file_name in to_load.items():
            loaded[name] = DataLoader.load_spec(
                os.path.join(dir_path, file_name)
            )
        return loaded

    def load_spec(file_path):
        full_file_path = os.path.abspath(file_path)
        with open(full_file_path) as f:
            spec = json.load(f)
        return DataLoader.process_spec(spec)

    def process_spec(spec):
        if spec['params']['dtype'] is not None:
            for a in spec['params']['dtype'].keys():
                b = spec['params']['dtype'][a]
                if b == 'np.float64':
                    spec['params']['dtype'][a] = np.float64
        return spec

    def make_delta(df, method, field):
        if method == 'calculated':
            return df[field] - df[field].shift(1)
        elif method == 'existing':
            return df[field]
        else:
            raise ValueError

    def read_data_file(spec):
        raw = pd.read_csv(**spec['params'])

        row_is_all_null_filter = raw.notnull().sum(axis=1)!=0
        raw = raw[row_is_all_null_filter]

        raw['date'] = pd.to_datetime(
            raw[spec['date_spec']['col']].values,
            format=spec['date_spec']['format']
        )

        print('\n')
        print(spec['dollar_columns'])
        print(raw[spec['dollar_columns']][:5])
        # print(spec['dollar_conversion'])
        # print(raw[spec['dollar_columns']] * spec['dollar_conversion'])
        print('\n')

        raw[spec['dollar_columns']] = (
            raw[spec['dollar_columns']] * spec['dollar_conversion']
        )

        raw = raw.sort_values(by='date')
        raw = raw.reindex(index=range(len(raw)))
        values = [(a, raw[b]) for a, b in spec['spec-values']]
        deltas = [
            (a, DataLoader.make_delta(raw, b, c))
            for a, b, c in spec['spec-deltas']
        ]

        data = pd.DataFrame(
            data=dict(values + deltas + [('date', raw['date'])])
        )
        return data

    def data_summary(df, output_dir, fields=None, title=None):
        if fields is None:
            a, b = ('value', 'delta')
        else:
            a, b = fields

        print(df['date'][:5])
        df.plot(x='date', y=[a, b], subplots=True, title=title, figsize=(4,4))
        maybe_make_dir(output_dir)
        plt.savefig(os.path.join(output_dir, 'data_summary.png'))

    def log_diff(R, axis=None):
        if axis == 0:
            return np.log(R[1:]) - np.log(R[:-1])
        elif axis == 1:
            return np.log(R[:, 1:]) - np.log(R[:, :-1])
        else:
            raise ValueError('Number of dimensions is not supported: %i' % R.ndim)

    def fractional_diff(R, axis=None):
        if axis == 0:
            return R[1:] / R[:-1] - 1
        elif axis == 1:
            return R[:, 1:] / R[:, :-1] - 1
        else:
            raise ValueError('Number number is not supported: %i' % axis)

    def make_log_diff(df, fields):
        spec = {'date': df['date']}
        for a in fields:
            spec[a] = log_diff(df[a], axis=0)  # np.log(df[a] / df[a].shift(1))
            # (gva[a] - gva[a].shift(1)) / gva[a].shift(1)

        ld = pd.DataFrame(data=spec)
        ld = ld[1:]  # First row is all NaNs
        return ld

    def make_fractional_diff(df, fields):
        spec = {'date': df['date']}
        for a in fields:
            spec[a] = fractional_diff(df[a].values, axis=0)

        ld = pd.DataFrame(data=spec, index=df.index[1:])
        return ld

    def read_transposed_data_file(spec):
        raw = pd.read_csv(**spec['params'], thousands=',')

        dates = pd.to_datetime(
            raw.columns[spec['date_spec']['from']:spec['date_spec']['to']].values,
            format=spec['date_spec']['format']
        )

        column_names = raw[spec['column_names']].values
        rawT = pd.DataFrame(
            data=raw.values[:, 1:].T,
            columns=column_names
        )
        rawT['date'] = dates

        row_is_all_null_filter = rawT.notnull().sum(axis=1)!=0
        rawT = rawT[row_is_all_null_filter]

        rawT = rawT.sort_values(by='date')
        rawT = rawT.reindex(index=range(len(rawT)))

        if spec['spec-values'] is not None:
            values = [(a, rawT[b].astype(np.float64)) for a, b in spec['spec-values']]
        else:
            values = [(a, rawT[a].astype(np.float64)) for a in rawT.columns.difference(['date']).values]

        if spec['spec-deltas'] is not None:
            deltas = [(a, make_delta(rawT, b, c)) for a, b, c in spec['spec-deltas']]
        else:
            deltas = []

        data = pd.DataFrame(
            data=dict(values + deltas + [('date', rawT['date'])])
        )
        return data
