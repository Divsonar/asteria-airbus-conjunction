"""Transparent Airbus challenge tier rules.

These are demonstration thresholds, not flight-qualified maneuver decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def collision_probability(log_risk):
    """Convert recorded log10(Pc), honoring the organizer's -30 sentinel.

    Returns zero for -30 (effectively uncomputably small), and NaN for invalid
    values. This function is for recorded telemetry, never classifier scores.
    """
    risk = pd.to_numeric(pd.Series(log_risk), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(risk) & (risk <= 0)
    result = np.full(len(risk), np.nan)
    result[valid] = np.power(10.0, risk[valid])
    result[valid & (risk == -30)] = 0.0
    return result


def classify(log_risk, miss_distance):
    """Return tier, rule and data-quality arrays for log10(Pc) and metres.

Critical rules can establish a critical tier on their own. Otherwise both
values must be usable before an event is called elevated or safe.
    """
    tiers, rules, quality, _ = classify_with_rule_ids(log_risk, miss_distance)
    return tiers, rules, quality


def classify_with_rule_ids(log_risk, miss_distance):
    """Return tiers, existing explanations, data quality and stable rule IDs.

    IDs map to the Airbus challenge brief: C1/C2 are critical probability /
    distance, E1/E2 are elevated probability / distance, and S1 is safe.
    A critical condition still takes precedence when the other input is absent.
    """
    risk = pd.to_numeric(pd.Series(log_risk), errors="coerce").to_numpy(dtype=float)
    dist = pd.to_numeric(pd.Series(miss_distance), errors="coerce").to_numpy(dtype=float)
    valid_risk = np.isfinite(risk) & (risk <= 0)
    valid_dist = np.isfinite(dist) & (dist >= 0)
    critical_risk = valid_risk & (risk >= -4)
    critical_dist = valid_dist & (dist < 1000)
    complete = valid_risk & valid_dist
    elevated_risk = valid_risk & (risk >= -6)
    elevated_dist = valid_dist & (dist <= 5000)

    tiers, rules, quality, rule_ids = [], [], [], []
    for i in range(len(risk)):
        issues = []
        if not valid_risk[i]:
            issues.append("invalid or missing risk")
        if not valid_dist[i]:
            issues.append("invalid or missing miss distance")
        quality.append("; ".join(issues) if issues else "Complete")
        if critical_risk[i] or critical_dist[i]:
            triggers = []
            ids = []
            if critical_risk[i]:
                triggers.append("estimated Pc >= 1e-4")
                ids.append("C1")
            if critical_dist[i]:
                triggers.append("miss distance < 1 km")
                ids.append("C2")
            tiers.append("Critical")
            rules.append("; ".join(triggers))
            rule_ids.append("; ".join(ids))
        elif not complete[i]:
            tiers.append("Incomplete")
            rules.append("insufficient valid data")
            rule_ids.append("")
        elif elevated_risk[i] or elevated_dist[i]:
            triggers = []
            ids = []
            if elevated_risk[i]:
                triggers.append("estimated Pc >= 1e-6")
                ids.append("E1")
            if elevated_dist[i]:
                triggers.append("miss distance <= 5 km")
                ids.append("E2")
            tiers.append("Elevated")
            rules.append("; ".join(triggers))
            rule_ids.append("; ".join(ids))
        else:
            tiers.append("Safe")
            rules.append("estimated Pc < 1e-6 and miss distance > 5 km")
            rule_ids.append("S1")
    return tiers, rules, quality, rule_ids
