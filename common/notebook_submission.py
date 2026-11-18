"""Read two manually prepared notebooks. Students do not generate manifests or exports."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

from .catalog import VIEWS, read_json
from .hashing import sha256

TAG = 'cafe.inference'
SCHEMA = 'cafe_four_view_submission_v10'
NOTEBOOKS = {'baseline':'baseline.ipynb', 'improved':'improved.ipynb'}
ROOT_FILES = {*NOTEBOOKS.values(), 'submission.json', 'requirements.txt', 'report.pdf'}

def relative_file(name):
    if not isinstance(name, str):
        raise ValueError('Submission paths must be strings.')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name:
        raise ValueError('Invalid submission file path: '+name)
    return path

def inference_source(notebook):
    """Extract tagged definitions only. Training, path setup and login cells never run."""
    if notebook.get('nbformat') != 4:
        raise ValueError('Submit an ipynb v4 notebook.')
    pieces = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code' or TAG not in cell.get('metadata', {}).get('tags', []):
            continue
        source = cell['source']
        if isinstance(source, list):
            source = ''.join(source)
        ast.parse(source)
        pieces.append(source)
    if not pieces:
        raise ValueError('No code cells tagged cafe.inference were found.')
    return '\n\n'.join(pieces)+'\n'

def validate_submission_folder(root, max_bytes=8*1024**3):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Select an existing submission folder.')
    files = []
    total = 0
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Links are not allowed in submissions.')
        relative = relative_file(path.relative_to(root).as_posix())
        if path.is_dir():
            if relative.parts[0] != 'weights':
                raise ValueError('Submission folder is not allowed: '+str(relative))
            continue
        if not path.is_file() or (str(relative) not in ROOT_FILES and relative.parts[0] != 'weights'):
            raise ValueError('Submission file is not allowed: '+str(relative))
        size = path.stat().st_size
        total += size
        if total > max_bytes:
            raise ValueError('Submission exceeds the file size limit')
        files.append({'path':str(relative), 'bytes':size, 'sha256':sha256(path)})
    actual = {item['path'] for item in files}
    required_files = {*NOTEBOOKS.values(), 'submission.json', 'requirements.txt'}
    if not required_files.issubset(actual):
        raise ValueError('Missing submission files: '+', '.join(sorted(required_files-actual)))
    user_config = read_json(root/'submission.json')
    required_keys = {'max_level','baseline_checkpoint','improved_checkpoint'}
    if set(user_config) != required_keys:
        raise ValueError('submission.json requires max_level, baseline_checkpoint, and improved_checkpoint. Input views are fixed.')
    level = user_config['max_level']
    if type(level) is not int or level not in (1,2,3):
        raise ValueError('max_level must be an integer from 1 to 3.')
    views = list(VIEWS)
    config = {'schema':SCHEMA, 'max_level':level, 'models':{}, 'files':files, 'inference_sha256':{}}
    sources = {}
    for method, notebook_name in NOTEBOOKS.items():
        checkpoint = user_config[method+'_checkpoint']
        if method == 'baseline' and checkpoint is None:
            raise ValueError('The baseline checkpoint is required.')
        if checkpoint is not None:
            path = relative_file(checkpoint)
            if path.parts[0] != 'weights' or checkpoint not in actual:
                raise ValueError('Declared checkpoint is missing from weights/: '+checkpoint)
        source = inference_source(read_json(root/notebook_name))
        sources[method] = source
        config['models'][method] = {
            'notebook':notebook_name, 'class_name':'Predictor', 'checkpoint':checkpoint,
            'views':['sw'] if method=='baseline' else views,
            'inference_file':method+'_inference.py',
        }
        config['inference_sha256'][method] = hashlib.sha256(source.encode()).hexdigest()
    # Every team compares against the same supplied baseline. Markdown, outputs,
    # paths and optional training cells may change; its inference and weights may not.
    reference_path = Path(__file__).resolve().parents[1]/'package/baseline_reference.json'
    reference = read_json(reference_path)
    if config['inference_sha256']['baseline'] != reference['inference_sha256']:
        raise ValueError('Keep the supplied baseline inference code. Implement improvements in improved.ipynb.')
    if sha256(root/user_config['baseline_checkpoint']) != reference['checkpoint_sha256']:
        raise ValueError('Use the supplied baseline weights as the comparison reference.')
    return config, sources

def unpack_submission(source, destination, max_bytes=8*1024**3):
    """Copy/read a submission into the grader's own directory, then extract each model separately."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    root = destination/'submission'
    if source.is_dir():
        validate_submission_folder(source, max_bytes)
        shutil.copytree(source, root)
    else:
        with zipfile.ZipFile(source) as archive:
            entries = archive.infolist()
            if sum(item.file_size for item in entries) > max_bytes:
                raise ValueError('Submission exceeds the extracted size limit')
            normalized = [item.filename.rstrip('/') for item in entries]
            if len(normalized) != len(set(normalized)):
                raise ValueError('Duplicate ZIP path')
            for item in entries:
                path = relative_file(item.filename.rstrip('/'))
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Links are not allowed in a submission ZIP.')
                if path.parts[0] != 'submission':
                    raise ValueError('The ZIP must have one top-level submission folder.')
                if len(path.parts) == 1:
                    if not item.is_dir():
                        raise ValueError('Invalid submission root')
                    continue
                relative = PurePosixPath(*path.parts[1:])
                if item.is_dir():
                    if relative.parts[0] != 'weights':
                        raise ValueError('Submission folder is not allowed')
                elif str(relative) not in ROOT_FILES and relative.parts[0] != 'weights':
                    raise ValueError('Submission file is not allowed: '+str(relative))
            archive.extractall(destination)
    config, sources = validate_submission_folder(root, max_bytes)
    for method, source_text in sources.items():
        (root/config['models'][method]['inference_file']).write_text(source_text, encoding='utf-8')
    return root, config
