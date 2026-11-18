"""Outcome-only lift scoring for the open-ended mug project prototype.

This module deliberately knows nothing about images, segmentation, predicted
points, a preferred part of the mug, or a placement target.  A successful episode
ends with the mug held above the table for a continuous, sufficiently quiet
interval.  Any allowed gripper contact counts: bilateral finger contact and an
upright mug are *not* requirements.

Thresholds below are initial engineering choices, not validated classroom grade
boundaries.  The simulator, not the submitted student program, must produce the
telemetry.  It must record every physics tick, including the final tick after any
evaluation/settling period; an earlier successful prefix must never replace the
full episode.  These checks cannot authenticate a fabricated/truncated trace.

Continuity means all recorded physics ticks qualify and adjacent timestamps are
one ``expected_dt`` apart.  Hold duration is elapsed time between the first and
last qualifying samples, NOT sample_count * dt.  Thus a 1-second hold at 240 Hz
requires at least 241 qualifying samples.  No assumptions about unobserved
sub-tick motion are made.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any


# Numerical tolerances only, not additional physical margins.  The time check
# permits rounding in timestamps but cannot hide a missing or duplicate tick.
_VALUE_TOL = 1e-12
_TIME_ABS_TOL = 1e-9
_TIME_REL_TOL = 1e-6
_NUMERIC_FIELDS = (
    "time_s", "center_z_m", "bottom_z_m", "linear_speed_m_s",
    "angular_speed_rad_s",
)
_BOOL_FIELDS = ("gripper_contact", "forbidden_contact")


def _number(value: Any, name: str) -> float:
    """Require a finite real scalar; booleans are never valid measurements."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar, not {type(value).__name__}")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be representable as a finite real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class LiftRules:
    """Prototype physical thresholds, all in SI units.

    ``min_rise_m`` uses the mug center relative to its episode-start height;
    ``min_clearance_m`` uses the lowest point relative to the table top.  Both
    must hold, so tipping a mug without lifting its bottom cannot pass.  Speed
    limits apply to magnitudes, not signed z velocities.  ``expected_dt`` must
    match the trusted simulator's actual physics tick duration.
    """

    min_rise_m: float = 0.10
    min_clearance_m: float = 0.05
    hold_seconds: float = 1.0
    max_linear_speed_m_s: float = 0.15
    max_angular_speed_rad_s: float = 1.5
    expected_dt: float = 1.0 / 240.0

    def __post_init__(self) -> None:
        for name in (
            "min_rise_m", "min_clearance_m", "hold_seconds",
            "max_linear_speed_m_s", "max_angular_speed_rad_s", "expected_dt",
        ):
            value = _number(getattr(self, name), f"rules.{name}")
            if name in ("min_rise_m", "min_clearance_m", "hold_seconds", "expected_dt"):
                if value <= 0:
                    raise ValueError(f"rules.{name} must be positive")
            elif value < 0:
                raise ValueError(f"rules.{name} must be nonnegative")
            object.__setattr__(self, name, value)


def _at_least(value: float, threshold: float) -> bool:
    return value >= threshold or math.isclose(
        value, threshold, rel_tol=_VALUE_TOL, abs_tol=0.0,
    )


def _at_most(value: float, threshold: float) -> bool:
    return value <= threshold or math.isclose(
        value, threshold, rel_tol=_VALUE_TOL, abs_tol=0.0,
    )


def evaluate_lift(
    samples: Iterable[Mapping[str, Any]],
    initial_center_z: float,
    table_z: float = 0.0,
    rules: LiftRules | None = None,
) -> dict[str, Any]:
    """Score the final contiguous hold, returning a JSON-friendly dictionary.

    Each sample requires ``time_s``, ``center_z_m``, ``bottom_z_m``,
    ``linear_speed_m_s``, ``angular_speed_rad_s`` (finite real scalars), and
    ``gripper_contact``, ``forbidden_contact`` (actual Python booleans).
    Additional fields are ignored.  ``forbidden_contact`` means contact of the
    mug with the table, clutter, or any robot link outside the allowed gripper;
    ``gripper_contact`` means contact with *any* allowed hand/finger link.

    Invalid/incomplete telemetry raises ``ValueError``.  A physically failed
    but well-formed episode returns ``success=False`` and diagnostic reasons.
    Reasons describe the final sample plus insufficient terminal hold, not an
    earlier table contact that every normal pickup would contain.  A transient
    successful lift followed by a drop always fails.  ``max_rise_m`` is only a
    diagnostic and cannot make an episode successful.

    The caller must distinguish evaluator errors from student-method failures;
    malformed evaluator telemetry is not a valid low score for a student.
    """
    if rules is None:
        rules = LiftRules()
    if not isinstance(rules, LiftRules):
        raise ValueError("rules must be a LiftRules instance")
    initial_z = _number(initial_center_z, "initial_center_z")
    table_height = _number(table_z, "table_z")
    try:
        iterator = iter(samples)
    except TypeError as exc:
        raise ValueError("samples must be an iterable of mappings") from exc

    previous_t: float | None = None
    terminal_start_t: float | None = None
    terminal_hold = 0.0
    max_rise = -math.inf
    final_reasons: list[str] = []
    sample_count = 0

    for index, sample in enumerate(iterator):
        if not isinstance(sample, Mapping):
            raise ValueError(f"samples[{index}] must be a mapping")
        missing = set(_NUMERIC_FIELDS + _BOOL_FIELDS) - sample.keys()
        if missing:
            raise ValueError(f"samples[{index}] is missing fields: {sorted(missing)}")
        values = {
            field: _number(sample[field], f"samples[{index}].{field}")
            for field in _NUMERIC_FIELDS
        }
        for field in _BOOL_FIELDS:
            if type(sample[field]) is not bool:
                raise ValueError(f"samples[{index}].{field} must be a bool")

        t = values["time_s"]
        if t < 0:
            raise ValueError(f"samples[{index}].time_s must be nonnegative")
        if previous_t is not None:
            dt = t - previous_t
            if dt <= 0 or not math.isclose(
                dt, rules.expected_dt, rel_tol=_TIME_REL_TOL,
                abs_tol=min(_TIME_ABS_TOL, rules.expected_dt * 1e-3),
            ):
                raise ValueError(
                    f"samples[{index}] timestamp step {dt!r} does not match "
                    f"expected_dt={rules.expected_dt!r}; missing, duplicate, "
                    "or out-of-order physics tick"
                )
        previous_t = t
        if not _at_most(values["bottom_z_m"], values["center_z_m"]):
            raise ValueError(f"samples[{index}].bottom_z_m cannot exceed center_z_m")
        for field in ("linear_speed_m_s", "angular_speed_rad_s"):
            if values[field] < 0:
                raise ValueError(f"samples[{index}].{field} must be nonnegative")

        rise = values["center_z_m"] - initial_z
        clearance = values["bottom_z_m"] - table_height
        max_rise = max(max_rise, rise)
        final_reasons = []
        if not _at_least(rise, rules.min_rise_m):
            final_reasons.append("insufficient_rise")
        if not _at_least(clearance, rules.min_clearance_m):
            final_reasons.append("insufficient_clearance")
        if not sample["gripper_contact"]:
            final_reasons.append("no_gripper_contact")
        if sample["forbidden_contact"]:
            final_reasons.append("forbidden_contact")
        if not _at_most(values["linear_speed_m_s"], rules.max_linear_speed_m_s):
            final_reasons.append("linear_speed_exceeded")
        if not _at_most(values["angular_speed_rad_s"], rules.max_angular_speed_rad_s):
            final_reasons.append("angular_speed_exceeded")

        if final_reasons:
            terminal_start_t = None
            terminal_hold = 0.0
        else:
            if terminal_start_t is None:
                terminal_start_t = t
            terminal_hold = t - terminal_start_t
        sample_count += 1

    if not sample_count:
        raise ValueError("samples must contain at least one physics tick")
    if not _at_least(terminal_hold, rules.hold_seconds):
        final_reasons.append("insufficient_terminal_hold")
    success = not final_reasons
    return {
        "success": success,
        "reason": "stable_lift" if success else final_reasons[0],
        "reasons": final_reasons,
        "terminal_hold_seconds": terminal_hold,
        "required_hold_seconds": rules.hold_seconds,
        "max_rise_m": max_rise,
        "sample_count": sample_count,
    }
