"""Predict, execute the shared robot, and return level-wise results."""
from datetime import datetime
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pandas as pd
import torch
from tqdm.auto import tqdm

from . import tracking
from .catalog import DATA, VIEWS, samples, save_json, dataset_index_hashes
from .evaluation import score_predictions
from .public_io import load_observations
from .request import parse_request
from robot.adapter import validate_prediction, command_from_prediction


def evaluate_picking(predictor, *, max_level=1, data_root=DATA, views=('sw',),
               model_name='baseline', output=None, wandb_project='cafe-cv-project',
               wandb_entity=None, use_wandb=False, robot_workers=1):
    """Evaluate all VAL requests in the selected cumulative levels and return a DataFrame.

    predictor must provide predict(observations, request).
    Save the result table to level_scores.csv and per-request outcomes to records.json.
    CSV/JSON files are always saved; W&B logging is optional and disabled by default.
    robot_workers controls CPU simulations only; model prediction stays unchanged.
    """
    if type(robot_workers) is not int or robot_workers < 1:
        raise ValueError('robot_workers must be a positive integer.')
    if not callable(getattr(predictor, 'predict', None)):
        raise TypeError('Provide a model object with predict(observations, request).')
    views = list(views)
    if not views or len(views) != len(set(views)) or not set(views).issubset(VIEWS):
        raise ValueError('Specify valid camera names without duplicates.')
    if model_name == 'baseline' and views != ['sw']:
        raise ValueError('Baseline evaluation uses the fixed SW view only.')
    if model_name != 'baseline' and set(views) != set(VIEWS):
        raise ValueError('Improved evaluation requires all four views: SW, SE, NE, NW.')
    if not model_name or Path(model_name).name != model_name:
        raise ValueError('The model name must not contain a path.')
    data_root = Path(data_root).resolve()
    rows = samples(max_level, 'val', data_root)
    if not rows:
        raise ValueError('No VAL requests found in the selected scope.')
    if output is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output = Path('runs')/(model_name+'_validation_'+stamp)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    config = {'model_name':model_name, 'split':'val', 'max_level':max_level,
              'views':views, 'requests':len(rows), 'partial_run':False,
              'robot_workers':robot_workers,
              'dataset_indexes_sha256':dataset_index_hashes(max_level, 'val', data_root),
              'execution':'current notebook predictor; final TEST uses instructor process'}
    args = SimpleNamespace(wandb_project=wandb_project,
                           wandb_entity=wandb_entity, output=output, run_name=output.name)
    run = tracking.start_run(args, config, enabled=use_wandb)
    run_url = run.url if run is not None else None
    save_json(output/'run.json', {**config, 'use_wandb':use_wandb, 'wandb_run_url':run_url})
    records = []
    summary = None
    try:
        for row in tqdm(rows, desc=model_name+' prediction', unit='request'):
            # Do not include labels or robot scenes in model inputs.
            observations = load_observations(row, data_root)
            allowed = {view:observations[view] for view in views}
            request = parse_request(row['request']['text'])
            record = {'sample_id':row['sample_id'], 'scene_id':row['scene_id'],
                      'request_id':row['request_id'], 'level':row['level'],
                      'split':'val', 'allowed_views':views}
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    prediction = predictor.predict(copy.deepcopy(allowed), request)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prediction_seconds = time.perf_counter()-started
                # Check JSON serialization, as required by the final grading process.
                prediction = json.loads(json.dumps(prediction, allow_nan=False))
                validate_prediction(prediction, allowed)
                conversion = command_from_prediction(allowed, prediction)
                record.update(prediction=prediction, conversion=conversion,
                              prediction_seconds=prediction_seconds)
            except Exception as error:
                record.update(success=False, outcome='model_error',
                              error=f'{type(error).__name__}: {error}')
            records.append(record)
            save_json(output/'predictions.json', {'records':records})

        summary = score_predictions(
            rows, records, output, max_level,
            method=model_name, robot_workers=robot_workers,
        )
        table = pd.DataFrame([{'level':level, **values} for level, values in summary['by_level'].items()])
        table.to_csv(output/'level_scores.csv', index=False)
        save_json(output/'summary.json', summary)
        if run is not None:
            import wandb
            run.log({'validation/level_scores':wandb.Table(dataframe=table)})
            for level, values in summary['by_level'].items():
                run.summary['validation/'+level+'/success_rate'] = values['success_rate']
        table.attrs['output'] = str(output)
        table.attrs['wandb_url'] = run_url
        return table
    finally:
        import sys
        tracking.finish_run(run, summary or {'status':'interrupted'}, failed=sys.exc_info()[0] is not None)
