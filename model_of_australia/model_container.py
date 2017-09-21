import os
import numpy as np
import json
from scipy import stats
from scripts import settings
from models import ModelPostProcessor, ModelSummariser


class ModelContainer(object):
    def __init__(
        self, name, folder, model_fn, data, parameter_spec,
        fn_args={}, iters=1e4, tune_iters=1e3
    ):
        self.name = name
        self.folder = folder
        self.model_fn = model_fn
        self.data = data
        self.parameter_spec = parameter_spec
        self.fn_args = fn_args
        self.iters = iters
        self.tune_iters = tune_iters

    def run(self):
        start_str = 'Running %s.' % self.name
        print('\n' + start_str + '\n' + '-' * len(start_str))

        self.results = self.fit_model()
        self.summarise()
        self.parameters = self.calculate_parameters()
        self.save_parameters()
        print('Done.')

    def fit_model(self):
        results = self.model_fn(
            self.data, iters=self.iters, tune_iters=self.tune_iters,
            **self.fn_args
        )
        return results

    def summarise(self):
        ModelSummariser.summarise(
            self.results, self.parameter_spec,
            output_dir=os.path.join(settings.OUTPUTS_DIR, self.folder),
            burn=2e3, dic=False, gelman_rubin_mean_only=True
        )

    def calculate_parameters(self):
        parameters = ModelPostProcessor.beta_extracter_version2(
            self.results[1],
            self.parameter_spec
        )
        return parameters

    def save_parameters(self):
        string_parameters = self.make_json_ready(self.parameters)
        with open(ModelContainer.parameters_file_path(self.folder), 'w') as f:
            json.dump(string_parameters, f, indent=4)

    def parameters_file_path(folder_name):
        return os.path.join(
            settings.OUTPUTS_DIR,
            folder_name,
            'parameters.json'
        )

    def make_json_ready(self, parameters):
        string_parameters = {}
        for a, P in self.parameters.items():
            string_parameters[a] = []
            for p_i in P:
                n, betas, dist = p_i
                betas = [float(b) for b in betas]
                dist_str = ModelContainer.dist_to_str(dist)
                string_parameters[a] += [[n, betas, dist_str]]
        return string_parameters

    def unmake_json_ready(string_parameters):
        parameters = {}
        for a, P in string_parameters.items():
            parameters[a] = []
            for p_i in P:
                n, betas, dist_str = p_i
                betas = tuple([np.float64(b) for b in betas])
                dist = ModelContainer.str_to_dist(dist_str)
                parameters[a] += [(n, betas, dist_str)]
        return parameters

    def dist_to_str(dist):
        if isinstance(dist, stats._continuous_distns.norm_gen):
            return 'stats.norm'
        elif isinstance(dist, stats._continuous_distns.invgamma_gen):
            return 'stats.invgamma'

    def str_to_dist(dist_str):
        if dist_str == 'stats.norm':
            return stats.norm
        elif dist_str == 'stats.invgamma':
            return stats.invgamma

    def load_parameters(folder_name):
        with open(ModelContainer.parameters_file_path(folder_name), 'r') as f:
            string_parameters = json.load(f)
        return ModelContainer.unmake_json_ready(string_parameters)
