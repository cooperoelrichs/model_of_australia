import os
from scripts import settings
from models import ModelPostProcessor, ModelSummariser


class ModelContainer(object):
    def __init__(
        self, name, folder, model_fn, data, parameter_spec,
        fn_args={}, iters=1e4, tune_iters=2e3
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
