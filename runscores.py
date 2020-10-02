import argparse
# prevent asap other modules from defining the root logger using basicConfig
import glob
import os
import traceback

import pandas as pd

import amlb
from amlb.utils import config_load

parser = argparse.ArgumentParser()
parser.add_argument('predictions', type=str,
                    help='The predictions file to load and compute the scores for.')
args = parser.parse_args()

# script_name = os.path.splitext(os.path.basename(__file__))[0]
# log_dir = os.path.join(args.outdir if args.outdir else '.', 'logs')
# os.makedirs(log_dir, exist_ok=True)
# now_str = datetime_iso(date_sep='', time_sep='')
amlb.logger.setup(root_level='DEBUG', console_level='INFO')

config = config_load("resources/config.yaml")
config.run_mode = 'script'
config.script = os.path.basename(__file__)
config.root_dir = os.path.dirname(__file__)
amlb.resources.from_config(config)

data = {'id': [],
        'task': [],
        'framework': [],
        'constraint': [],
        'fold': [],
        'result': [],
        'metric': [],
        'mode': [],
        'version': [],
        'params': [],
        'tag': [],
        'utc': [],
        'duration': [],
        'models': [],
        'seed': [],
        'info': [],
        'acc': [],
        'auc': [],
        'logloss': []}
for filepath in glob.iglob(os.path.join(args.predictions, '*.csv')):
    try:
        score = amlb.TaskResult.score_from_predictions_file(filepath)
        for key in data.keys():
            data[key].append(vars(score).get(key, None))
    except Exception:
        traceback.print_exc()

df = pd.DataFrame(data)
df.to_csv('tmp.csv', index=False)
