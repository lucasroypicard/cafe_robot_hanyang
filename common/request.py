"""Parse the 30 canonical cup requests; this is not unrestricted language understanding.

The current data archive stores canonical English requests. Legacy aliases remain
available so previously extracted Korean datasets can still be read.
"""
import json
from pathlib import Path
from .catalog import COLORS, PATTERNS

_LEGACY = json.loads(Path(__file__).with_name('request_aliases.json').read_text(encoding='utf-8'))


def format_request(color, pattern):
    if color not in COLORS or pattern not in PATTERNS:
        raise ValueError('Unknown cup color or pattern')
    if pattern == 'plain':
        return f'Pick up the plain {color} cup.'
    if pattern == 'stripes':
        return f'Pick up the {color} cup with stripes.'
    return f'Pick up the {color} cup with a {pattern} pattern.'


def parse_request(text):
    normalized = ' '.join(text.strip().split()).rstrip('.')
    legacy = _LEGACY.get(normalized)
    if legacy is not None:
        color, pattern = legacy['color'], legacy['pattern']
        return {'text': format_request(color, pattern), 'color': color, 'pattern': pattern}
    for color in COLORS:
        for pattern in PATTERNS:
            canonical = format_request(color, pattern)
            if normalized == canonical.rstrip('.'):
                return {'text': canonical, 'color': color, 'pattern': pattern}
    raise ValueError('Use a canonical cup request from the supplied dataset')
