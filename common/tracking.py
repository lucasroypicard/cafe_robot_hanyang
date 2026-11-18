"""Optional W&B logging. Local results are saved independently of this module."""
from pathlib import Path
import os
import platform
import shutil
import torch


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def hardware(device):
    return {'device':str(device),
            'gpu_name':torch.cuda.get_device_name() if str(device).startswith('cuda') else None,
            'torch_version':str(torch.__version__), 'python_version':platform.python_version(),
            'runtime':'colab' if os.environ.get('COLAB_RELEASE_TAG') else 'local',
            'cpu_threads':torch.get_num_threads()}


def start_run(args, config, *, enabled=False):
    # Return before importing W&B or checking credentials when logging is disabled.
    if not enabled:
        return None
    mode=os.environ.get('WANDB_MODE','online')
    if mode not in ('online','offline'):
        raise ValueError('Optional W&B logging supports online or offline mode.')
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError('Install wandb to enable optional experiment logging.') from error
    if mode=='online' and not os.environ.get('WANDB_API_KEY') and not wandb.api.api_key:
        raise RuntimeError('Run the optional W&B login cell or wandb login first. Do not put keys in code.')
    return wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                      name=args.run_name or Path(args.output).name, config=config,
                      mode=mode, dir=str(args.output), save_code=False,
                      settings=wandb.Settings(disable_git=True))


def log_epoch(run, record):
    if run is not None:
        run.log(record,step=record['epoch'])


def finish_run(run, summary, failed=False):
    if run is not None:
        run.summary.update(summary)
        run.finish(exit_code=1 if failed else 0)


def backup_epoch(output, destination):
    """Copy small checkpoints and logs without copying the dataset."""
    if destination is None:
        return
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('best.pt','run.json','history.json','timing.json'):
        source=Path(output)/name
        if source.is_file():
            target=destination/name
            if source.resolve()==target.resolve():
                continue
            temporary=destination/(name+'.partial')
            shutil.copyfile(source,temporary)
            temporary.replace(target)
