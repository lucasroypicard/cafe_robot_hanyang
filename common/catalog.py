"""Fixed L1-L3 request lists for training, validation, and private evaluation."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset'
DATASET_SCHEMA = 'cafe_three_level_v10'
VIEWS = ('sw', 'se', 'ne', 'nw')
COLORS = ('red', 'blue', 'green', 'yellow', 'black', 'white')
PATTERNS = ('plain', 'flower', 'bird', 'leaf', 'stripes')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def installed_levels(data_root=DATA):
    return [level for level in (1,2,3) if (Path(data_root)/f'L{level}.json').is_file()]


def level_metadata(level, data_root=DATA):
    if level not in (1,2,3):
        raise ValueError('Use level 1..3')
    root = Path(data_root)
    if not (root/'index.json').is_file():
        raise ValueError('DATA_DIR must contain the supplied dataset/index.json.')
    if read_json(root/'index.json').get('schema') != DATASET_SCHEMA:
        raise ValueError('Use the three-level v10 dataset, not older four-level data.')
    path = root/f'L{level}.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing dataset/L{level}.json. Check DATA_DIR.')
    metadata = read_json(path)
    if metadata.get('schema') != DATASET_SCHEMA or metadata.get('level') != f'L{level}':
        raise ValueError('Level metadata differs from the supplied dataset')
    return metadata


def scene_folder(row, data_root=DATA):
    root = Path(data_root).resolve()
    scene_id = row['scene_id']
    if not re.fullmatch(r'scene\d{3,}', scene_id):
        raise ValueError('Invalid scene ID')
    if row['level'] not in ('L1','L2','L3') or row['split'] not in ('train','val','test'):
        raise ValueError('Invalid level or split')
    if row['scene_path'] != f'scenes/{scene_id}':
        raise ValueError('Use the shared scenes/<scene_id> path from the supplied JSON')
    path = (root/row['scene_path']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Scene path escapes dataset root')
    return path


def samples(max_level=1, split='train', data_root=DATA, exact=False):
    if max_level not in (1,2,3) or split not in ('train','val','test'):
        raise ValueError('Use level 1..3 and train/val/test')
    if split not in read_json(Path(data_root)/'index.json').get('allowed_splits', ('train','val','test')):
        raise PermissionError('This split is not included in the release: '+split)
    levels = [max_level] if exact else range(1,max_level+1)
    rows = []
    for level in levels:
        metadata = level_metadata(level,data_root)
        batch = metadata['splits'][split]
        if len(batch) != metadata['counts'][split]:
            raise ValueError('Request count differs from metadata')
        if any(r['split']!=split or r['level']!=f'L{level}' for r in batch):
            raise ValueError('Split/level differs from the supplied JSON')
        for row in batch:
            scene_folder(row,data_root)
        rows.extend(batch)
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate request across levels')
    return rows


def find_sample(sample_id, split, data_root=DATA):
    matches=[]
    for level in installed_levels(data_root):
        matches.extend(r for r in samples(level,split,data_root,exact=True) if r['sample_id']==sample_id)
    if len(matches)!=1:
        raise ValueError('Sample is not uniquely listed in the selected split: '+sample_id)
    return matches[0]


def dataset_index_hashes(max_level, split, data_root=DATA, exact=False):
    from .hashing import sha256
    levels=[max_level] if exact else range(1,max_level+1)
    paths=['index.json']+[f'L{level}.json' for level in levels]
    for level in levels:
        level_metadata(level,data_root)
    return {relative:sha256(Path(data_root)/relative) for relative in paths}
