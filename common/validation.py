"""Two-stage VAL evaluation: save predictions, then optionally execute the robot.

The notebook retains the model and learning code. This module handles fixed
request lists, diagnostic metrics, saved results, and the shared robot scorer.
Ground-truth labels are used only after prediction for diagnostic measurements.
"""
import copy
from datetime import datetime
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm.auto import tqdm

from . import tracking
from .catalog import DATA, VIEWS, dataset_index_hashes, read_json, samples, save_json, scene_folder
from .evaluation import score_predictions
from .public_io import load_observations
from .request import parse_request
from robot.adapter import command_from_prediction, validate_prediction


def _prediction_diagnostics(row, record, observations, data_root):
    """Measure target-mask overlap and reference distance, not grasp success."""
    result = {
        "sample_id": row["sample_id"], "level": row["level"],
        "request": row["request"]["text"], "status": "model_error",
        "selected_view": None, "point_on_target": False,
        "reference_error_px": None,
    }
    if record.get("outcome") == "model_error":
        return result
    prediction = record["prediction"]
    result["status"] = prediction["status"]
    if prediction["status"] == "NO_PICK":
        return result

    view = prediction["selected_view"]
    result["selected_view"] = view
    pixel = prediction.get("pixel_uv")
    if pixel is None:
        # 3D-only methods need not return a diagnostic image point.
        geometry = observations[view]["geometry"]
        world_point = np.append(prediction["world_xyz_m"], 1.0)
        camera_point = np.linalg.solve(geometry["camera_to_world"], world_point)[:3]
        if camera_point[2] <= 0:
            return result
        projected = geometry["K"] @ camera_point
        pixel = projected[:2] / projected[2]
    pixel = np.asarray(pixel, dtype=float)
    if not np.isfinite(pixel).all():
        return result

    # These labels never enter predictor.predict() or the robot command.
    label_folder = scene_folder(row, data_root) / "label" / row["request_id"]
    label = read_json(label_folder / "points.json")[view]
    with Image.open(label_folder / label["target_mask"]) as image:
        mask = np.asarray(image)
    height, width = mask.shape
    if 0 <= pixel[0] <= width - 1 and 0 <= pixel[1] <= height - 1:
        u, v = np.rint(pixel).astype(int)
        result["point_on_target"] = bool(mask[v, u] > 0)
    if label["coordinate_loss_valid"] and label["reference_uv"] is not None:
        result["reference_error_px"] = float(
            np.linalg.norm(pixel - np.asarray(label["reference_uv"])))
    return result


def _prediction_scores(details, max_level):
    """Include all requests in mask-hit rates; average only valid point pairs."""
    scores = []
    for level_number in range(1, max_level + 1):
        level = f"L{level_number}"
        items = details[details["level"] == level]
        errors = items["reference_error_px"].dropna()
        scores.append({
            "level": level, "requests": len(items),
            "point_on_target": int(items["point_on_target"].sum()),
            "point_on_target_rate": float(items["point_on_target"].mean()) if len(items) else None,
            "reference_pairs": len(errors),
            "mean_reference_error_px": float(errors.mean()) if len(errors) else None,
            "no_pick": int((items["status"] == "NO_PICK").sum()),
            "model_errors": int((items["status"] == "model_error").sum()),
        })
    return scores


def _log_table(table, output, config, stage, *, use_wandb, wandb_project, wandb_entity):
    """Local CSV/JSON files do not depend on W&B installation or login."""
    args = SimpleNamespace(wandb_project=wandb_project, wandb_entity=wandb_entity,
                           output=output, run_name=output.name)
    run = tracking.start_run(args, config, enabled=use_wandb)
    if run is not None:
        try:
            import wandb
            run.log({stage + "/level_scores": wandb.Table(dataframe=table)})
        finally:
            run.finish()


def evaluate_predictions(predictor, *, max_level=1, data_root=DATA, views=("sw",),
                         model_name="baseline", output_root="runs", checkpoint=None,
                         use_wandb=False, wandb_project="cafe-cv-project", wandb_entity=None):
    """Return (prediction_folder, diagnostic_table) without running the robot.

    Saves predictions.json, prediction_details.csv, prediction_scores.csv, and
    prediction_summary.json. The returned folder can be reused across runtimes.
    The fixed controller's commands are prepared here and executed only later.
    """
    if not callable(getattr(predictor, "predict", None)):
        raise TypeError("Provide a model object with predict(observations, request).")
    views = list(views)
    if model_name == "baseline" and views != ["sw"]:
        raise ValueError("The baseline uses only SW.")
    if model_name != "baseline" and (len(views) != 4 or set(views) != set(VIEWS)):
        raise ValueError("The improved model must receive all four views.")
    data_root = Path(data_root).resolve()
    rows = samples(max_level, "val", data_root)
    if not rows:
        raise ValueError("No VAL requests found in the selected scope.")
    output = Path(output_root).resolve() / (
        "predictions_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "model_name": model_name, "split": "val", "max_level": max_level,
        "views": views, "requests": len(rows), "partial_run": False,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "use_wandb": use_wandb,
        "dataset_indexes_sha256": dataset_index_hashes(max_level, "val", data_root),
        "stage": "prediction_only", "robot_executed": False,
    }
    save_json(output / "run.json", config)
    records = []
    diagnostics = []
    save_json(output / "predictions.json", {"complete": False, "records": records})

    for row in tqdm(rows, desc=model_name + " prediction", unit="request"):
        observations = load_observations(row, data_root)
        allowed = {view: observations[view] for view in views}
        request = parse_request(row["request"]["text"])
        record = {key: row[key] for key in
                  ("sample_id", "scene_id", "request_id", "level", "split")}
        record["allowed_views"] = views
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                prediction = predictor.predict(copy.deepcopy(allowed), request)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            prediction = json.loads(json.dumps(prediction, allow_nan=False))
            validate_prediction(prediction, allowed)
            conversion = command_from_prediction(allowed, prediction)
            record.update(prediction=prediction, conversion=conversion,
                          prediction_seconds=elapsed)
        except Exception as error:
            record.update(success=False, outcome="model_error",
                          error=f"{type(error).__name__}: {error}")
        records.append(record)
        diagnostics.append(_prediction_diagnostics(row, record, allowed, data_root))
        if len(records) % 20 == 0:
            save_json(output / "predictions.json", {"complete": False, "records": records})

    details = pd.DataFrame(diagnostics)
    scores = _prediction_scores(details, max_level)
    table = pd.DataFrame(scores)
    details.to_csv(output / "prediction_details.csv", index=False)
    table.to_csv(output / "prediction_scores.csv", index=False)
    save_json(output / "prediction_summary.json", {"by_level": scores, "robot_executed": False})
    save_json(output / "predictions.json", {"complete": True, "records": records})
    _log_table(table, output, config, "prediction", use_wandb=use_wandb,
               wandb_project=wandb_project, wandb_entity=wandb_entity)
    return output, table


def evaluate_saved_picking(prediction_dir, *, data_root=DATA, robot_workers=2,
                           use_wandb=False, wandb_project="cafe-cv-project", wandb_entity=None):
    """Return picking scores using saved commands; no model is needed.

    Reuse the saved request scope even if notebook settings have since changed.
    Save level_scores.csv, records.json, and summary.json in a new picking folder.
    The source predictions remain unchanged and can be evaluated again.
    """
    if type(robot_workers) is not int or robot_workers < 1:
        raise ValueError("robot_workers must be a positive integer.")
    prediction_dir = Path(prediction_dir).resolve()
    data_root = Path(data_root).resolve()
    config = read_json(prediction_dir / "run.json")
    saved = read_json(prediction_dir / "predictions.json")
    if not saved.get("complete"):
        raise ValueError("Prediction is incomplete. Finish prediction before picking.")
    max_level = config["max_level"]
    if config["dataset_indexes_sha256"] != dataset_index_hashes(max_level, "val", data_root):
        raise ValueError("Use the same dataset that produced these predictions.")
    rows = samples(max_level, "val", data_root)
    records = saved["records"]
    if [row["sample_id"] for row in rows] != [item["sample_id"] for item in records]:
        raise ValueError("Saved predictions must match all requests in their saved scope.")

    output = prediction_dir / ("picking_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    config = {**config, "stage": "picking", "robot_executed": True,
              "prediction_dir": str(prediction_dir), "robot_workers": robot_workers,
              "use_wandb": use_wandb}
    save_json(output / "run.json", config)
    summary = score_predictions(rows, records, output, max_level,
                                method=config["model_name"], robot_workers=robot_workers)
    table = pd.DataFrame([{"level": level, **values}
                          for level, values in summary["by_level"].items()])
    table.to_csv(output / "level_scores.csv", index=False)
    save_json(output / "summary.json", summary)
    _log_table(table, output, config, "picking", use_wandb=use_wandb,
               wandb_project=wandb_project, wandb_entity=wandb_entity)
    table.attrs["output"] = str(output)
    print("Saved picking results:", output)
    return table
