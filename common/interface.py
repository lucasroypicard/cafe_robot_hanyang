"""Check output format only, not model internals, accuracy, or ground truth."""
import json
from robot.adapter import validate_prediction


def check_output_format(prediction, observations):
    json.dumps(prediction, allow_nan=False)
    validate_prediction(prediction, observations)
    print('PASS: prediction output format')
