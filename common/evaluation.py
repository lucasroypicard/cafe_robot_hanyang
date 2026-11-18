"""Shared robot scoring with optional independent CPU worker processes."""
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from unittest.mock import patch

from tqdm.auto import tqdm
from .catalog import read_json, save_json


def score_predictions(rows, records, output, max_level, *, run_robot=True,
                      method='model', record=False, robot_workers=1):
    """Score predictions with the same controller and success criteria.

    Existing callers remain sequential. Validation can request more CPU workers.
    Recording and interface-only checks keep their existing sequential behavior.
    """
    if type(robot_workers) is not int or robot_workers < 1:
        raise ValueError('robot_workers must be a positive integer.')
    if robot_workers == 1 or not run_robot or record:
        return _score_predictions_sequential(
            rows, records, output, max_level,
            run_robot=run_robot, method=method, record=record,
        )
    return _score_predictions_parallel(
        rows, records, output, max_level, method=method, robot_workers=robot_workers,
    )


def _score_predictions_sequential(rows, records, output, max_level, *, run_robot=True, method='model', record=False):
    folder = Path(output)
    selected_level = max_level
    record_video = bool(record)
    if len(rows) != len(records) or any(row['sample_id'] != result['sample_id'] for row,result in zip(rows,records)):
        raise ValueError('Evaluation requests and predictions must match in count and order.')
    from robot.executor import execute
    for row,record in tqdm(list(zip(rows,records)),desc=method+' robot',unit='request'):
        if record.get('outcome')=='model_error':continue
        if not run_robot:
            record.update(success=None,outcome='interface_valid')
            continue
        started=time.perf_counter()
        try:
            episode = folder/'episodes'/row['scene_id']/row['request_id'] if record_video else None
            result=execute(row,record['conversion'],record=record_video,output=episode)
        except Exception as error:
            save_json(folder/'infrastructure_error.json',{'sample_id':row['sample_id'],'error':str(error)})
            raise  # Infrastructure faults never become student failures.
        status=record['prediction']['status']
        success=status=='PICK' and bool(result['metric']['success'])
        outcome='success' if success else 'pick_failed'
        if status=='NO_PICK':outcome='no_pick'
        elif not success and any(v>=.10 for v in result['other_cup_final_rise_m'].values()):outcome='other_cup_lifted'
        record.update(success=success,outcome=outcome,robot=result,robot_seconds=time.perf_counter()-started)
        save_json(folder/'records.json',{'records':records})
    save_json(folder/'records.json',{'records':records})
    by_level={}
    for level in range(1,selected_level+1):
        items=[r for r in records if r['level']==f'L{level}']
        successes=sum(r.get('success') is True for r in items)
        by_level[f'L{level}']={'requests':len(items),'successes':successes if run_robot else None,
                              'success_rate':successes/len(items) if items and run_robot else None,
                              'model_errors':sum(r.get('outcome')=='model_error' for r in items),
                              'other_cup_lifted':sum(r.get('outcome')=='other_cup_lifted' for r in items),
                              'pick_failed':sum(r.get('outcome')=='pick_failed' for r in items),
                              'no_pick':sum(r.get('outcome')=='no_pick' for r in items)}
    summary={'requests':len(records),'by_level':by_level,
             'interface_valid_requests':sum('prediction' in r for r in records),
             'successes':sum(r.get('success') is True for r in records) if run_robot else None,
             'prediction_seconds':sum(r.get('prediction_seconds',0) for r in records),
             'robot_seconds':sum(r.get('robot_seconds',0) for r in records)}
    return summary


def _score_predictions_parallel(rows, records, output, max_level, *, method, robot_workers):
    if len(rows) != len(records) or any(
        row['sample_id'] != prediction['sample_id']
        for row, prediction in zip(rows, records)
    ):
        raise ValueError('Requests and predictions must match in count and order.')

    output = Path(output)
    started = time.perf_counter()
    summaries = []
    worker_environment = {
        'TQDM_DISABLE': '1',
        'OMP_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
    }

    # Use local temporary files for workers, even when SAVE_PATH is on Drive.
    # The imported sequential scorer is compatible with spawned notebook workers.
    with tempfile.TemporaryDirectory(prefix='cafe_robot_') as temporary:
        with patch.dict(os.environ, worker_environment):
            with ProcessPoolExecutor(
                max_workers=robot_workers,
                mp_context=mp.get_context('spawn'),
            ) as pool:
                jobs = {}
                try:
                    for index, (row, prediction) in enumerate(zip(rows, records)):
                        job_folder = Path(temporary) / str(index)
                        future = pool.submit(
                            _score_predictions_sequential,
                            [row], [prediction], job_folder, max_level, method=method,
                        )
                        jobs[future] = (index, job_folder)

                    for completed, future in enumerate(
                        tqdm(
                            as_completed(jobs), total=len(jobs),
                            desc=f'{method} robot ({robot_workers} workers)',
                            unit='request', disable=False,
                        ), start=1,
                    ):
                        index, job_folder = jobs[future]
                        try:
                            summaries.append(future.result())
                        except Exception as error:
                            save_json(output/'infrastructure_error.json', {
                                'sample_id': rows[index]['sample_id'],
                                'error': str(error),
                            })
                            raise
                        result = read_json(job_folder/'records.json')
                        records[index] = result['records'][0]

                        # Preserve request order while reducing writes to Drive.
                        if completed % 20 == 0:
                            save_json(output/'records.json', {'records': records})
                except BaseException:
                    for future in jobs:
                        future.cancel()
                    save_json(output/'records.json', {'records': records})
                    raise

    save_json(output/'records.json', {'records': records})
    count_fields = (
        'requests', 'successes', 'model_errors', 'other_cup_lifted',
        'pick_failed', 'no_pick',
    )
    by_level = {}
    for level_number in range(1, max_level + 1):
        level = f'L{level_number}'
        totals = {}
        for field in count_fields:
            totals[field] = sum(s['by_level'][level][field] for s in summaries)
        # Keep the original table's column order, including success rate first.
        by_level[level] = {
            'requests': totals['requests'],
            'successes': totals['successes'],
            'success_rate': totals['successes']/totals['requests'] if totals['requests'] else None,
            'model_errors': totals['model_errors'],
            'other_cup_lifted': totals['other_cup_lifted'],
            'pick_failed': totals['pick_failed'],
            'no_pick': totals['no_pick'],
        }

    elapsed = time.perf_counter() - started
    print(f'Robot evaluation elapsed time: {elapsed / 60:.1f} minutes')
    return {
        'requests': len(records),
        'by_level': by_level,
        'interface_valid_requests': sum(s['interface_valid_requests'] for s in summaries),
        'successes': sum(s['successes'] for s in summaries),
        'prediction_seconds': sum(s['prediction_seconds'] for s in summaries),
        # Total simulation duration differs from elapsed wall time in parallel.
        'robot_seconds': sum(s['robot_seconds'] for s in summaries),
        'robot_wall_seconds': elapsed,
        'robot_workers': robot_workers,
    }
