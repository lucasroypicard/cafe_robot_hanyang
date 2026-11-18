"""Optional one-request VAL demonstration using the same scorer as full evaluation.

Original RGB is the model input. Target outlines are diagnostic annotations drawn
only after prediction. Robot rendering runs in DIRECT mode and never opens a GUI.
"""
import copy
from datetime import datetime
from pathlib import Path
import json
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch

from .catalog import DATA, VIEWS, samples, scene_folder, read_json, save_json
from .public_io import load_observations
from .request import parse_request
from .evaluation import score_predictions
from robot.adapter import validate_prediction, command_from_prediction


def _project_world(point, geometry):
    camera = np.linalg.inv(geometry['camera_to_world']) @ np.r_[point, 1.]
    if not np.isfinite(camera).all() or camera[2] <= 0:
        return None
    pixel = geometry['K'] @ camera[:3]
    return pixel[:2] / pixel[2]


def _input_image(row, data_root, observations, prediction, outcome, output):
    selected_view = prediction.get('selected_view')
    if selected_view not in observations:
        selected_view = next(iter(observations))
    observation = observations[selected_view]
    rgb = observation['rgb'].copy()

    # This VAL label is for the human-facing picture, not the predictor packet.
    label_folder = scene_folder(row, data_root) / 'label' / row['request_id']
    labels = read_json(label_folder / 'points.json')
    with Image.open(label_folder / labels[selected_view]['target_mask']) as image:
        mask = image.convert('L')
    outer = np.asarray(mask.filter(ImageFilter.MaxFilter(5))) > 0
    inner = np.asarray(mask.filter(ImageFilter.MinFilter(3))) > 0
    rgb[outer & ~inner] = [0, 255, 255]

    image = Image.fromarray(rgb)
    drawing = ImageDraw.Draw(image)

    def marker(uv, color, cross):
        if uv is None:
            return
        point = np.asarray(uv, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            return
        x, y = point
        if not (0 <= x < image.width and 0 <= y < image.height):
            return
        radius = 8
        if cross:
            drawing.line((x-radius, y-radius, x+radius, y+radius), fill=color, width=3)
            drawing.line((x-radius, y+radius, x+radius, y-radius), fill=color, width=3)
        else:
            drawing.line((x-radius, y, x+radius, y), fill=color, width=3)
            drawing.line((x, y-radius, x, y+radius), fill=color, width=3)

    # Show both coordinates: an accurate 2D pixel can still have an incorrect 3D conversion.
    marker(prediction.get('pixel_uv'), (255, 0, 160), cross=True)
    if prediction.get('status') == 'PICK':
        marker(_project_world(prediction['world_xyz_m'], observation['geometry']),
               (255, 220, 0), cross=False)

    panel = Image.new('RGB', (image.width, image.height + 92), (25, 29, 35))
    panel.paste(image, (0, 92))
    drawing = ImageDraw.Draw(panel)
    request = row['request']
    drawing.text((10, 8), f"Request: {request['color']} / {request['pattern']} | view: {selected_view}", fill='white')
    drawing.text((10, 28), f"{row['sample_id']} | {outcome.upper()}", fill='white')
    drawing.text((10, 48), 'Cyan: target cup (VAL label) | magenta x: model pixel', fill='white')
    drawing.text((10, 68), 'Yellow +: model world point projected into this image', fill='white')
    panel.save(output / 'prediction.png')


def visualize_picking(predictor, sample_id, *, max_level=1, data_root=DATA,
                      views=('sw',), model_name='baseline', output=None, show=True):
    """Run one selected VAL request, save the input overlay/GIF/result, optionally display.

    This is a selected example, not a success-rate estimate. Full VAL scoring remains
    evaluate_picking(). No simulator state, target mask or labels are sent to the model.
    """
    views = tuple(views)
    if not views or len(set(views)) != len(views) or not set(views).issubset(VIEWS):
        raise ValueError('Specify the views actually used, without duplicates.')
    if model_name == 'baseline' and views != ('sw',):
        raise ValueError('The baseline uses SW only.')
    if model_name != 'baseline' and set(views) != set(VIEWS):
        raise ValueError('Provide all four views to the improved model.')
    if not callable(getattr(predictor, 'predict', None)):
        raise TypeError('Provide a predictor implementing predict(observations, request).')
    rows = samples(max_level, 'val', data_root)
    matches = [row for row in rows if row['sample_id'] == sample_id]
    if len(matches) != 1:
        raise ValueError('Select a VAL sample_id in the chosen levels. This function does not access TRAIN/TEST.')
    row = matches[0]
    if output is None:
        output = Path('runs') / ('picking_demo_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    observations = load_observations(row, data_root)
    allowed = {view: observations[view] for view in views}
    request = parse_request(row['request']['text'])
    record = {key: row[key] for key in ('sample_id', 'scene_id', 'request_id', 'level')}
    record.update(split='val', allowed_views=list(views))
    prediction = {}
    try:
        started = time.perf_counter()
        with torch.inference_mode():
            returned = predictor.predict(copy.deepcopy(allowed), request)
        candidate = json.loads(json.dumps(returned, allow_nan=False))
        validate_prediction(candidate, allowed)
        prediction = candidate
        conversion = command_from_prediction(allowed, prediction)
        record.update(prediction=prediction, conversion=conversion,
                      prediction_seconds=time.perf_counter() - started)
    except Exception as error:
        record.update(success=False, outcome='model_error', error=f'{type(error).__name__}: {error}')

    # The shared scorer owns success/NO_PICK/wrong-cup decisions and physics errors.
    summary = score_predictions([row], [record], output, max_level, method=model_name, record=True)
    _input_image(row, data_root, allowed, prediction, record['outcome'], output)
    episode = output / 'episodes' / row['scene_id'] / row['request_id']
    gif_path = episode / 'robot.gif'
    result = {
        'sample_id': sample_id, 'request': request['text'], 'level': row['level'],
        'success': record['success'], 'outcome': record['outcome'], 'prediction': prediction,
        'selected_example_only': True, 'output': str(output),
        'prediction_image': str(output / 'prediction.png'),
        'robot_gif': str(gif_path) if gif_path.is_file() else None,
        'robot_start': str(episode / 'robot_start.png') if gif_path.is_file() else None,
        'robot_final': str(episode / 'robot_final.png') if gif_path.is_file() else None,
        'summary': summary,
    }
    if 'error' in record:
        result['error'] = record['error']
    save_json(output / 'demo.json', result)
    if show:
        from IPython.display import Image as NotebookImage, display
        print(request['text'])
        print('Selected request:', result['outcome'], '| Use evaluate_picking() for full performance.')
        display(NotebookImage(filename=result['prediction_image']))
        if result['robot_gif']:
            print('Cyan outline: target cup. The robot view does not display the original cup patterns.')
            display(NotebookImage(filename=result['robot_gif']))
        else:
            print('Robot execution skipped because of a model output error:', result.get('error'))
    return result
