import logging
import os
import pickle
import time

from dswizard.core.master import Master
from dswizard.core.model import Dataset as DsDataset
from dswizard.optimizers.bandit_learners import HyperbandLearner
from dswizard.optimizers.config_generators import Hyperopt
from dswizard.optimizers.structure_generators.mcts import MCTS, TransferLearning
from dswizard.util import util

from frameworks.shared.callee import call_run, result, output_subdir

logging.getLogger().setLevel(logging.DEBUG)
log = logging.getLogger(os.path.basename(__file__))


def run(dataset, config):
    log.info("\n**** dswizzard ****\n")
    print(config)

    log_file = os.path.join(output_subdir('models', config), 'log.log')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(name)-15s %(threadName)-10s %(message)s')

    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    metrics_mapping = dict(
        acc='accuracy',
        auc='rocauc',
        f1='f1',
        logloss='logloss'
    )
    scoring_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if scoring_metric is None:
        raise ValueError("Performance metric {} not supported.".format(config.metric))

    is_classification = config.type == 'classification'

    X_train, X_test = dataset.train.X, dataset.test.X
    y_train, y_test = dataset.train.y.flatten(), dataset.test.y.flatten()

    log.info(X_train.shape)
    log.info(y_train.shape)

    workdir = os.path.join(output_subdir('models', config))

    n_jobs = config.framework_params.get('_n_jobs', config.cores)
    log.info('Running dswizard with a maximum time of %ss on %s cores, optimizing %s.',
             config.max_runtime_seconds, n_jobs, scoring_metric)

    data_file = os.path.join(output_subdir('models', config), 'data.pkl')
    with open(data_file, 'wb') as f:
        pickle.dump((X_train, y_train, X_test, y_test), f)

    try:
        ds = DsDataset(X_train, y_train, metric=scoring_metric)
        master = Master(
            ds=ds,
            working_directory=workdir,
            n_workers=n_jobs,
            model='./frameworks/dswizard/assets/rf_complete.pkl',

            wallclock_limit=config.max_runtime_seconds,
            cutoff=min(max(config.max_runtime_seconds // 10, 60), 300),
            pre_sample=True,

            config_generator_class=Hyperopt,

            structure_generator_class=MCTS,
            structure_generator_kwargs={'policy': TransferLearning},

            bandit_learner_class=HyperbandLearner,
            bandit_learner_kwargs={'min_budget': 1,
                                   'max_budget': 10}
        )
        with Timer() as training:
            predictor, run_history = master.optimize()
            ensemble = master.build_ensemble()
        log.info('Incumbent: {}'.format(ensemble))
        scores = []

        predictor.fit(X_train, y_train)
        predictions = predictor.predict(X_test)
        probabilities = predictor.predict_proba(X_test) if is_classification else None
        scores.append(util.score(y_test, probabilities, predictions, scoring_metric))

        predictions = ensemble.predict(X_test)
        probabilities = ensemble.predict_proba(X_test) if is_classification else None
        scores.append(util.score(y_test, probabilities, predictions, scoring_metric))

        log.info('Final performances: {}'.format(scores))
        model_file = os.path.join(output_subdir('models', config), 'models.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump((predictor, ensemble), f)
    except Exception as ex:
        log.exception('Unhandled exception', ex, exc_info=True)
        raise ex

    log.info("Finished")
    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=False,
                  training_duration=training.duration)


class Timer:

    @staticmethod
    def _zero():
        return 0

    def __init__(self, clock=time.time, enabled=True):
        self.start = 0
        self.stop = 0
        self._time = clock if enabled else Timer._zero

    def __enter__(self):
        self.start = self._time()
        return self

    def __exit__(self, *args):
        self.stop = self._time()

    @property
    def duration(self):
        if self.stop > 0:
            return self.stop - self.start
        return self._time() - self.start


if __name__ == '__main__':
    call_run(run)
