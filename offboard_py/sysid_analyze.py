#!/usr/bin/env python3
"""Offline analysis for sysid CSV data.

Loads PX4 SITL step response CSVs from sysid_node, computes metrics
(matching auto_tune.py definitions), generates comparison PDFs, and
outputs a mismatch summary table.

No ROS2 dependency — pure Python + numpy + matplotlib.

Usage:
    python3 sysid_analyze.py --sysid-dir /path/to/sysid_output
    python3 sysid_analyze.py --sysid-dir /path/to/sysid_output --threshold 0.2
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Metric dataclass (matching auto_tune.py OscillationMetrics)
# ---------------------------------------------------------------------------

@dataclass
class OscillationMetrics:
    """Oscillation characterization for a single signal."""
    damping_ratio: float = 1.0
    zero_crossings: int = 0
    ss_amplitude: float = 0.0
    frequency: float = 0.0


@dataclass
class VelocityStepMetrics:
    """Metrics for a velocity step response test."""
    settling_time: float = 0.0  # seconds to reach 95% of target
    overshoot_pct: float = 0.0  # % overshoot beyond target
    ss_error: float = 0.0       # steady-state error (last 0.5s mean)
    oscillation: OscillationMetrics = field(default_factory=OscillationMetrics)


@dataclass
class HoverMetrics:
    """Metrics for hover drift test."""
    drift_mean: float = 0.0  # mean horizontal drift [m]
    drift_max: float = 0.0   # max horizontal drift [m]


@dataclass
class YawStepMetrics:
    """Metrics for yaw step response."""
    settling_time: float = 0.0
    overshoot_pct: float = 0.0
    ss_error: float = 0.0


@dataclass
class GimbalRateMetrics:
    """Metrics for gimbal LOS rate step test."""
    pitch_settling_time: float = 0.0
    pitch_overshoot: float = 0.0
    yaw_settling_time: float = 0.0
    yaw_overshoot: float = 0.0


@dataclass
class GimbalStabilizationMetrics:
    """Metrics for gimbal LOS stabilization test."""
    peak_az_drift_deg: float = 0.0
    peak_el_drift_deg: float = 0.0
    rms_az_drift_deg: float = 0.0
    rms_el_drift_deg: float = 0.0


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_sysid_csv(path: str) -> pd.DataFrame:
    """Load a sysid CSV file, dropping duplicate timestamps."""
    df = pd.read_csv(path)
    # Drop duplicate timestamps (keep last — most recent telemetry)
    df = df.drop_duplicates(subset=["timestamp_s"], keep="last").reset_index(drop=True)
    # Normalize time to start at 0
    df["t"] = df["timestamp_s"] - df["timestamp_s"].iloc[0]
    return df


# ---------------------------------------------------------------------------
# Metric computation (matching auto_tune.py definitions)
# ---------------------------------------------------------------------------

def compute_oscillation_metrics(
    error_signal: np.ndarray, settling_idx: int, dt: float
) -> OscillationMetrics:
    """Compute oscillation metrics from error signal after settling.

    Reimplementation of auto_tune.py::compute_oscillation_metrics using numpy.
    """
    T = len(error_signal)
    settling_idx = max(0, min(settling_idx, T - 4))
    ss = error_signal[settling_idx:]

    if len(ss) < 4:
        return OscillationMetrics()

    # Zero crossings
    signs = np.sign(ss)
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    sign_changes = (signs[1:] * signs[:-1]) < 0
    zero_crossings = int(sign_changes.sum())

    # Peak-to-peak steady-state amplitude
    ss_amplitude = float(ss.max() - ss.min())

    # Damping ratio via logarithmic decrement
    damping_ratio = 1.0
    abs_ss = np.abs(ss)
    peaks = []
    for i in range(1, len(abs_ss) - 1):
        if abs_ss[i] > abs_ss[i - 1] and abs_ss[i] > abs_ss[i + 1]:
            peaks.append((i, abs_ss[i]))

    if len(peaks) >= 2:
        p1, p2 = peaks[0][1], peaks[1][1]
        if p1 > 1e-8 and p2 > 1e-8 and p1 > p2:
            log_dec = math.log(p1 / p2)
            damping_ratio = log_dec / math.sqrt(4.0 * math.pi**2 + log_dec**2)
            damping_ratio = min(max(damping_ratio, 0.0), 1.0)
        elif p1 > 1e-8 and p2 >= p1:
            damping_ratio = 0.0

    # Frequency from zero-crossing intervals
    frequency = 0.0
    if zero_crossings >= 2:
        crossing_indices = np.where(sign_changes)[0]
        if len(crossing_indices) >= 2:
            intervals = np.diff(crossing_indices) * dt
            mean_half_period = intervals.mean()
            if mean_half_period > 1e-8:
                frequency = 1.0 / (2.0 * mean_half_period)

    return OscillationMetrics(
        damping_ratio=damping_ratio,
        zero_crossings=zero_crossings,
        ss_amplitude=ss_amplitude,
        frequency=frequency,
    )


def compute_velocity_step_metrics(
    df: pd.DataFrame, target: float, axis: str = "vel_x"
) -> VelocityStepMetrics:
    """Compute velocity step response metrics from sysid CSV data."""
    vel = df[axis].values
    t = df["t"].values
    dt = np.median(np.diff(t))

    # Settling time (95% of target)
    threshold = 0.95 * target
    settled = vel >= threshold
    if settled.any():
        settling_time = t[np.argmax(settled)]
    else:
        settling_time = t[-1]

    # Overshoot
    max_vel = vel.max()
    overshoot_pct = max(0.0, (max_vel - target) / target * 100.0)

    # Steady-state error (last 0.5s)
    last_n = max(1, int(0.5 / dt))
    ss_error = abs(target - vel[-last_n:].mean())

    # Oscillation metrics
    vel_error = vel - target
    settling_idx = int(settling_time / dt) if settling_time < t[-1] else 0
    osc = compute_oscillation_metrics(vel_error, settling_idx, dt)

    return VelocityStepMetrics(
        settling_time=settling_time,
        overshoot_pct=overshoot_pct,
        ss_error=ss_error,
        oscillation=osc,
    )


def compute_hover_metrics(df: pd.DataFrame) -> HoverMetrics:
    """Compute hover drift metrics."""
    # Horizontal drift from initial position
    dx = df["pos_x"].values - df["pos_x"].iloc[0]
    dy = df["pos_y"].values - df["pos_y"].iloc[0]
    drift = np.sqrt(dx**2 + dy**2)
    return HoverMetrics(drift_mean=float(drift.mean()), drift_max=float(drift.max()))


def compute_yaw_step_metrics(df: pd.DataFrame) -> YawStepMetrics:
    """Compute yaw step response metrics from angular velocity."""
    # Yaw rate is ang_vel_z (body FLU → yaw is Z)
    yaw_rate = df["ang_vel_z"].values
    t = df["t"].values
    dt = np.median(np.diff(t))
    target = 0.5  # rad/s

    # Only analyze first 2s (command period)
    cmd_mask = t <= 2.0
    yaw_cmd = yaw_rate[cmd_mask]
    t_cmd = t[cmd_mask]

    threshold = 0.95 * target
    settled = yaw_cmd >= threshold
    settling_time = t_cmd[np.argmax(settled)] if settled.any() else 2.0

    max_rate = yaw_cmd.max()
    overshoot_pct = max(0.0, (max_rate - target) / target * 100.0)

    # SS error during command (last 0.5s of command period)
    last_n = max(1, int(0.5 / dt))
    cmd_end = yaw_cmd[-last_n:]
    ss_error = abs(target - cmd_end.mean())

    return YawStepMetrics(
        settling_time=settling_time,
        overshoot_pct=overshoot_pct,
        ss_error=ss_error,
    )


def compute_gimbal_rate_metrics(df: pd.DataFrame) -> GimbalRateMetrics:
    """Compute gimbal LOS rate step metrics."""
    t = df["t"].values
    el = df["gimbal_los_el_deg"].values
    az = df["gimbal_los_az_deg"].values

    # Pitch phase: 0-0.5s rate, 0.5-1.0s hold
    # Yaw phase: 1.0-1.5s rate, 1.5-2.0s hold
    # Measure settling after each pulse (hold period)

    # Pitch: rate of el change during hold (0.5-1.0s)
    pitch_hold = (t >= 0.5) & (t < 1.0)
    if pitch_hold.any():
        el_hold = el[pitch_hold]
        pitch_settling = float(np.abs(np.diff(el_hold)).max()) if len(el_hold) > 1 else 0.0
    else:
        pitch_settling = 0.0

    # Yaw: rate of az change during hold (1.5-2.0s)
    yaw_hold = (t >= 1.5) & (t < 2.0)
    if yaw_hold.any():
        az_hold = az[yaw_hold]
        yaw_settling = float(np.abs(np.diff(az_hold)).max()) if len(az_hold) > 1 else 0.0
    else:
        yaw_settling = 0.0

    return GimbalRateMetrics(
        pitch_settling_time=pitch_settling,
        yaw_settling_time=yaw_settling,
    )


def compute_gimbal_stabilization_metrics(df: pd.DataFrame) -> GimbalStabilizationMetrics:
    """Compute gimbal LOS stabilization metrics."""
    az = df["gimbal_los_az_deg"].values
    el = df["gimbal_los_el_deg"].values

    # Target is the initial LOS (first sample)
    az_target = az[0]
    el_target = el[0]

    az_drift = az - az_target
    el_drift = el - el_target

    return GimbalStabilizationMetrics(
        peak_az_drift_deg=float(np.abs(az_drift).max()),
        peak_el_drift_deg=float(np.abs(el_drift).max()),
        rms_az_drift_deg=float(np.sqrt(np.mean(az_drift**2))),
        rms_el_drift_deg=float(np.sqrt(np.mean(el_drift**2))),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _quat_to_rp_deg(qx, qy, qz, qw):
    """Extract roll/pitch in degrees from quaternion arrays (ENU/FLU)."""
    roll = np.degrees(2.0 * (qw * qx + qy * qz))
    pitch = np.degrees(np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1, 1)))
    return roll, pitch


def _has_setpoint_data(df: pd.DataFrame) -> bool:
    """Check if attitude setpoint columns exist and have valid data."""
    return "att_sp_qw" in df.columns and not df["att_sp_qw"].isna().all()


def plot_velocity_test(df: pd.DataFrame, target: float, test_name: str,
                       metrics: VelocityStepMetrics, output_path: str) -> None:
    """Plot velocity step response with cascade comparison.

    4 rows: velocity, attitude (actual vs setpoint), rate (actual vs setpoint), thrust.
    """
    has_sp = _has_setpoint_data(df)
    n_rows = 4 if has_sp else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.5 * n_rows), sharex=True)
    t = df["t"].values

    # Row 0: Velocity
    ax = axes[0]
    ax.plot(t, df["vel_x"].values, "b-", label="vel_x (forward)", linewidth=0.8)
    ax.axhline(target, color="r", linestyle="--", label=f"target={target} m/s")
    ax.axhline(0.95 * target, color="gray", linestyle=":", alpha=0.5, label="95% band")
    ax.set_ylabel("Velocity [m/s]")
    ax.legend(fontsize=8)
    ax.set_title(
        f"{test_name}: settling={metrics.settling_time:.2f}s, "
        f"overshoot={metrics.overshoot_pct:.1f}%, "
        f"SS_err={metrics.ss_error:.3f} m/s"
    )

    # Row 1: Attitude actual vs setpoint
    ax = axes[1]
    roll, pitch = _quat_to_rp_deg(
        df["quat_x"].values, df["quat_y"].values,
        df["quat_z"].values, df["quat_w"].values,
    )
    ax.plot(t, roll, "b-", label="roll (actual)", linewidth=0.8)
    ax.plot(t, pitch, "r-", label="pitch (actual)", linewidth=0.8)
    if has_sp:
        sp_roll, sp_pitch = _quat_to_rp_deg(
            df["att_sp_qx"].values, df["att_sp_qy"].values,
            df["att_sp_qz"].values, df["att_sp_qw"].values,
        )
        ax.plot(t, sp_roll, "b--", label="roll (setpoint)", linewidth=0.8, alpha=0.7)
        ax.plot(t, sp_pitch, "r--", label="pitch (setpoint)", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Attitude [deg]")
    ax.legend(fontsize=7, ncol=2)

    # Row 2: Rate actual vs setpoint
    ax = axes[2]
    ax.plot(t, np.degrees(df["ang_vel_x"].values), "b-", label="p (actual)", linewidth=0.8)
    ax.plot(t, np.degrees(df["ang_vel_y"].values), "r-", label="q (actual)", linewidth=0.8)
    ax.plot(t, np.degrees(df["ang_vel_z"].values), "g-", label="r (actual)", linewidth=0.8)
    if has_sp:
        ax.plot(t, np.degrees(df["rate_sp_x"].values), "b--", label="p (setpoint)", linewidth=0.8, alpha=0.7)
        ax.plot(t, np.degrees(df["rate_sp_y"].values), "r--", label="q (setpoint)", linewidth=0.8, alpha=0.7)
        ax.plot(t, np.degrees(df["rate_sp_z"].values), "g--", label="r (setpoint)", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Angular velocity [deg/s]")
    ax.legend(fontsize=7, ncol=2)

    # Row 3: Thrust (if setpoint data available)
    if has_sp:
        ax = axes[3]
        ax.plot(t, df["thrust_sp"].values, "k-", linewidth=0.8, label="thrust cmd")
        ax.set_ylabel("Thrust [normalized]")
        ax.set_xlabel("Time [s]")
        ax.legend(fontsize=8)
    else:
        axes[-1].set_xlabel("Time [s]")

    osc = metrics.oscillation
    fig.text(
        0.99, 0.01,
        f"damping={osc.damping_ratio:.3f}  ZC={osc.zero_crossings}  "
        f"SS_amp={osc.ss_amplitude:.3f}  freq={osc.frequency:.1f}Hz",
        ha="right", fontsize=8, style="italic",
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_hover_test(df: pd.DataFrame, metrics: HoverMetrics, output_path: str) -> None:
    """Plot hover drift test."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t = df["t"].values

    ax = axes[0]
    dx = df["pos_x"].values - df["pos_x"].iloc[0]
    dy = df["pos_y"].values - df["pos_y"].iloc[0]
    dz = df["pos_z"].values - df["pos_z"].iloc[0]
    ax.plot(t, dx, label="dx", linewidth=0.8)
    ax.plot(t, dy, label="dy", linewidth=0.8)
    ax.plot(t, dz, label="dz", linewidth=0.8)
    ax.set_ylabel("Position drift [m]")
    ax.legend(fontsize=8)
    ax.set_title(f"Hover drift: mean={metrics.drift_mean:.3f}m, max={metrics.drift_max:.3f}m")

    ax = axes[1]
    drift = np.sqrt(dx**2 + dy**2)
    ax.plot(t, drift, "k-", linewidth=0.8)
    ax.set_ylabel("Horizontal drift [m]")
    ax.set_xlabel("Time [s]")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_yaw_test(df: pd.DataFrame, metrics: YawStepMetrics, output_path: str) -> None:
    """Plot yaw step response."""
    fig, ax = plt.subplots(figsize=(10, 4))
    t = df["t"].values
    yaw_rate = df["ang_vel_z"].values

    ax.plot(t, yaw_rate, "b-", linewidth=0.8, label="yaw_rate")
    ax.axhline(0.5, color="r", linestyle="--", label="target=0.5 rad/s")
    ax.axvline(2.0, color="gray", linestyle=":", alpha=0.5, label="cmd off at 2s")
    ax.set_ylabel("Yaw rate [rad/s]")
    ax.set_xlabel("Time [s]")
    ax.legend(fontsize=8)
    ax.set_title(
        f"Yaw step: settling={metrics.settling_time:.2f}s, "
        f"overshoot={metrics.overshoot_pct:.1f}%, SS_err={metrics.ss_error:.3f}"
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_gimbal_rate_test(df: pd.DataFrame, metrics: GimbalRateMetrics,
                          output_path: str) -> None:
    """Plot gimbal LOS rate step test."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t = df["t"].values

    ax = axes[0]
    ax.plot(t, df["gimbal_los_el_deg"].values, "b-", linewidth=0.8, label="elevation")
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("LOS elevation [deg]")
    ax.legend(fontsize=8)
    ax.set_title("Gimbal LOS rate step: pitch 0-0.5s, hold 0.5-1.0s, yaw 1.0-1.5s, hold 1.5-2.0s")

    ax = axes[1]
    ax.plot(t, df["gimbal_los_az_deg"].values, "r-", linewidth=0.8, label="azimuth")
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(1.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("LOS azimuth [deg]")
    ax.set_xlabel("Time [s]")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_gimbal_stabilization_test(df: pd.DataFrame,
                                    metrics: GimbalStabilizationMetrics,
                                    output_path: str) -> None:
    """Plot gimbal LOS stabilization test."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t = df["t"].values

    az_drift = df["gimbal_los_az_deg"].values - df["gimbal_los_az_deg"].iloc[0]
    el_drift = df["gimbal_los_el_deg"].values - df["gimbal_los_el_deg"].iloc[0]

    ax = axes[0]
    ax.plot(t, az_drift, "r-", linewidth=0.8, label="az drift")
    ax.plot(t, el_drift, "b-", linewidth=0.8, label="el drift")
    ax.set_ylabel("LOS drift [deg]")
    ax.legend(fontsize=8)
    ax.set_title(
        f"Gimbal stabilization: peak_az={metrics.peak_az_drift_deg:.2f}°, "
        f"peak_el={metrics.peak_el_drift_deg:.2f}°, "
        f"rms_az={metrics.rms_az_drift_deg:.2f}°, rms_el={metrics.rms_el_drift_deg:.2f}°"
    )

    ax = axes[1]
    # Show drone velocity to see disturbance phases
    ax.plot(t, df["vel_x"].values, label="vel_x", linewidth=0.8)
    ax.plot(t, df["vel_y"].values, label="vel_y", linewidth=0.8)
    ax.plot(t, df["ang_vel_z"].values, label="yaw_rate", linewidth=0.8)
    ax.set_ylabel("Drone cmd response")
    ax.set_xlabel("Time [s]")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Mismatch table
# ---------------------------------------------------------------------------

def build_mismatch_table(all_metrics: dict, threshold: float) -> list[dict]:
    """Build mismatch summary. Returns list of {metric, value, pass_fail} dicts."""
    rows = []

    def add(name: str, value, unit: str = ""):
        rows.append({
            "metric": name,
            "value": f"{value:.4f}" if isinstance(value, float) else str(value),
            "unit": unit,
        })

    if "hover_drift" in all_metrics:
        m = all_metrics["hover_drift"]
        add("hover_drift_mean", m.drift_mean, "m")
        add("hover_drift_max", m.drift_max, "m")

    for test_name in ["vel_step_5", "vel_step_10"]:
        if test_name in all_metrics:
            m = all_metrics[test_name]
            add(f"{test_name}_settling_time", m.settling_time, "s")
            add(f"{test_name}_overshoot", m.overshoot_pct, "%")
            add(f"{test_name}_ss_error", m.ss_error, "m/s")
            add(f"{test_name}_damping_ratio", m.oscillation.damping_ratio, "")
            add(f"{test_name}_zero_crossings", m.oscillation.zero_crossings, "")
            add(f"{test_name}_ss_amplitude", m.oscillation.ss_amplitude, "m/s")
            add(f"{test_name}_frequency", m.oscillation.frequency, "Hz")

    if "yaw_step" in all_metrics:
        m = all_metrics["yaw_step"]
        add("yaw_settling_time", m.settling_time, "s")
        add("yaw_overshoot", m.overshoot_pct, "%")
        add("yaw_ss_error", m.ss_error, "rad/s")

    if "gimbal_los_stabilization" in all_metrics:
        m = all_metrics["gimbal_los_stabilization"]
        add("gimbal_stab_peak_az_drift", m.peak_az_drift_deg, "deg")
        add("gimbal_stab_peak_el_drift", m.peak_el_drift_deg, "deg")
        add("gimbal_stab_rms_az_drift", m.rms_az_drift_deg, "deg")
        add("gimbal_stab_rms_el_drift", m.rms_el_drift_deg, "deg")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze sysid test data")
    parser.add_argument("--sysid-dir", type=str, required=True,
                        help="Directory containing sysid CSVs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots/results (default: sysid-dir/analysis)")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to iris_ma6 tuning_results_*.json (current gains to compare against PX4 SITL target)")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Mismatch threshold (default: 0.2 = 20%%)")
    args = parser.parse_args()

    sysid_dir = Path(args.sysid_dir)
    output_dir = Path(args.output_dir) if args.output_dir else sysid_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SYSID ANALYSIS")
    print("=" * 70)
    print(f"Input:  {sysid_dir}")
    print(f"Output: {output_dir}")
    print()

    all_metrics = {}

    # --- Hover drift ---
    hover_csv = sysid_dir / "hover_drift.csv"
    if hover_csv.exists():
        print("Analyzing hover_drift...")
        df = load_sysid_csv(str(hover_csv))
        m = compute_hover_metrics(df)
        all_metrics["hover_drift"] = m
        plot_hover_test(df, m, str(output_dir / "hover_drift.pdf"))
        print(f"  drift_mean={m.drift_mean:.4f}m, drift_max={m.drift_max:.4f}m")

    # --- Velocity steps ---
    for test_name, target in [("vel_step_5", 5.0), ("vel_step_10", 10.0)]:
        csv_path = sysid_dir / f"{test_name}.csv"
        if csv_path.exists():
            print(f"Analyzing {test_name}...")
            df = load_sysid_csv(str(csv_path))
            m = compute_velocity_step_metrics(df, target)
            all_metrics[test_name] = m
            plot_velocity_test(df, target, test_name, m, str(output_dir / f"{test_name}.pdf"))
            print(f"  settling={m.settling_time:.3f}s, overshoot={m.overshoot_pct:.1f}%, "
                  f"ss_err={m.ss_error:.4f}, damping={m.oscillation.damping_ratio:.3f}")

    # --- Velocity impulse recovery ---
    impulse_csv = sysid_dir / "vel_impulse_recovery.csv"
    if impulse_csv.exists():
        print("Analyzing vel_impulse_recovery...")
        df = load_sysid_csv(str(impulse_csv))
        has_sp = _has_setpoint_data(df)
        n_rows = 4 if has_sp else 3
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.5 * n_rows), sharex=True)
        t = df["t"].values

        axes[0].plot(t, df["vel_x"].values, "b-", linewidth=0.8, label="vel_x")
        axes[0].axvline(1.0, color="gray", linestyle=":", label="impulse off")
        axes[0].set_ylabel("Velocity [m/s]")
        axes[0].legend(fontsize=8)
        axes[0].set_title("Velocity impulse recovery: 5 m/s for 1s then zero")

        roll, pitch = _quat_to_rp_deg(
            df["quat_x"].values, df["quat_y"].values,
            df["quat_z"].values, df["quat_w"].values,
        )
        axes[1].plot(t, roll, "b-", label="roll (actual)", linewidth=0.8)
        axes[1].plot(t, pitch, "r-", label="pitch (actual)", linewidth=0.8)
        if has_sp:
            sp_roll, sp_pitch = _quat_to_rp_deg(
                df["att_sp_qx"].values, df["att_sp_qy"].values,
                df["att_sp_qz"].values, df["att_sp_qw"].values,
            )
            axes[1].plot(t, sp_roll, "b--", label="roll (sp)", linewidth=0.8, alpha=0.7)
            axes[1].plot(t, sp_pitch, "r--", label="pitch (sp)", linewidth=0.8, alpha=0.7)
        axes[1].set_ylabel("Attitude [deg]")
        axes[1].legend(fontsize=7, ncol=2)

        axes[2].plot(t, np.degrees(df["ang_vel_x"].values), "b-", label="p (actual)", linewidth=0.8)
        axes[2].plot(t, np.degrees(df["ang_vel_y"].values), "r-", label="q (actual)", linewidth=0.8)
        if has_sp:
            axes[2].plot(t, np.degrees(df["rate_sp_x"].values), "b--", label="p (sp)", linewidth=0.8, alpha=0.7)
            axes[2].plot(t, np.degrees(df["rate_sp_y"].values), "r--", label="q (sp)", linewidth=0.8, alpha=0.7)
        axes[2].set_ylabel("Angular velocity [deg/s]")
        axes[2].legend(fontsize=7, ncol=2)

        if has_sp:
            axes[3].plot(t, df["thrust_sp"].values, "k-", linewidth=0.8, label="thrust cmd")
            axes[3].set_ylabel("Thrust [normalized]")
            axes[3].legend(fontsize=8)
        axes[-1].set_xlabel("Time [s]")

        plt.tight_layout()
        out_path = str(output_dir / "vel_impulse_recovery.pdf")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")

    # --- Yaw step ---
    yaw_csv = sysid_dir / "yaw_step.csv"
    if yaw_csv.exists():
        print("Analyzing yaw_step...")
        df = load_sysid_csv(str(yaw_csv))
        m = compute_yaw_step_metrics(df)
        all_metrics["yaw_step"] = m
        plot_yaw_test(df, m, str(output_dir / "yaw_step.pdf"))
        print(f"  settling={m.settling_time:.3f}s, overshoot={m.overshoot_pct:.1f}%, "
              f"ss_err={m.ss_error:.4f}")

    # --- Gimbal LOS rate step ---
    gimbal_rate_csv = sysid_dir / "gimbal_los_rate_step.csv"
    if gimbal_rate_csv.exists():
        print("Analyzing gimbal_los_rate_step...")
        df = load_sysid_csv(str(gimbal_rate_csv))
        m = compute_gimbal_rate_metrics(df)
        all_metrics["gimbal_los_rate_step"] = m
        plot_gimbal_rate_test(df, m, str(output_dir / "gimbal_los_rate_step.pdf"))

    # --- Gimbal LOS stabilization ---
    gimbal_stab_csv = sysid_dir / "gimbal_los_stabilization.csv"
    if gimbal_stab_csv.exists():
        print("Analyzing gimbal_los_stabilization...")
        df = load_sysid_csv(str(gimbal_stab_csv))
        m = compute_gimbal_stabilization_metrics(df)
        all_metrics["gimbal_los_stabilization"] = m
        plot_gimbal_stabilization_test(df, m, str(output_dir / "gimbal_los_stabilization.pdf"))
        print(f"  peak_az={m.peak_az_drift_deg:.2f}°, peak_el={m.peak_el_drift_deg:.2f}°")

    # --- Load reference (iris_ma6 tuning results) ---
    ref_metrics = None
    if args.baseline:
        ref_path = Path(args.baseline)
        if ref_path.exists():
            with open(ref_path) as f:
                ref_data = json.load(f)
            ref_metrics = ref_data["best"]["metrics"]
            print(f"\nReference loaded: {ref_path}")
        else:
            print(f"\nWARN: Reference file not found: {ref_path}")

    # --- Comparison table ---
    print()
    print("=" * 80)
    if ref_metrics:
        print("SIM-TO-SIM COMPARISON: iris_ma6 vs PX4 SITL")
    else:
        print("SYSID METRICS (PX4 SITL — no baseline provided, use --baseline)")
    print("=" * 80)

    threshold = args.threshold
    comparison_rows = []

    def add_row(name, sitl_val, ref_val=None, unit=""):
        if ref_val is not None and abs(ref_val) > 1e-8:
            diff_pct = abs(sitl_val - ref_val) / abs(ref_val) * 100
            passed = diff_pct <= threshold * 100
            flag = "PASS" if passed else "FAIL"
        elif ref_val is not None:
            diff_pct = None
            flag = "N/A"
        else:
            diff_pct = None
            flag = ""
        comparison_rows.append({
            "metric": name, "iris_ma6": ref_val, "px4_sitl": sitl_val,
            "diff_pct": diff_pct, "flag": flag, "unit": unit,
        })

    # Hover
    if "hover_drift" in all_metrics:
        m = all_metrics["hover_drift"]
        add_row("hover_drift_mean", m.drift_mean,
                ref_metrics.get("hover_drift_mean") if ref_metrics else None, "m")
        add_row("hover_drift_max", m.drift_max,
                ref_metrics.get("hover_drift_max") if ref_metrics else None, "m")

    # Velocity 5 m/s
    if "vel_step_5" in all_metrics:
        m = all_metrics["vel_step_5"]
        add_row("vel_5_settling_time", m.settling_time,
                ref_metrics.get("vel_settling_time") if ref_metrics else None, "s")
        add_row("vel_5_overshoot", m.overshoot_pct,
                ref_metrics.get("vel_overshoot") if ref_metrics else None, "%")
        add_row("vel_5_ss_error", m.ss_error,
                ref_metrics.get("vel_ss_error") if ref_metrics else None, "m/s")
        add_row("vel_5_damping_ratio", m.oscillation.damping_ratio,
                ref_metrics["vel_osc_5"]["damping_ratio"] if ref_metrics else None)
        add_row("vel_5_ss_amplitude", m.oscillation.ss_amplitude,
                ref_metrics["vel_osc_5"]["ss_amplitude"] if ref_metrics else None, "m/s")

    # Velocity 10 m/s
    if "vel_step_10" in all_metrics:
        m = all_metrics["vel_step_10"]
        add_row("vel_10_settling_time", m.settling_time, None, "s")
        add_row("vel_10_ss_error", m.ss_error, None, "m/s")
        add_row("vel_10_damping_ratio", m.oscillation.damping_ratio,
                ref_metrics["vel_osc_10"]["damping_ratio"] if ref_metrics else None)
        add_row("vel_10_ss_amplitude", m.oscillation.ss_amplitude,
                ref_metrics["vel_osc_10"]["ss_amplitude"] if ref_metrics else None, "m/s")

    # Yaw
    if "yaw_step" in all_metrics:
        m = all_metrics["yaw_step"]
        add_row("yaw_settling_time", m.settling_time, None, "s")
        add_row("yaw_overshoot", m.overshoot_pct, None, "%")
        add_row("yaw_ss_error", m.ss_error, None, "rad/s")

    # Gimbal
    if "gimbal_los_stabilization" in all_metrics:
        m = all_metrics["gimbal_los_stabilization"]
        add_row("gimbal_stab_peak_az_drift", m.peak_az_drift_deg, None, "deg")
        add_row("gimbal_stab_peak_el_drift", m.peak_el_drift_deg, None, "deg")

    # Print table
    has_ref = ref_metrics is not None
    if has_ref:
        hdr = f"  {'Metric':<30} {'iris_ma6':>10} {'PX4 SITL':>10} {'Diff %':>8} {'':>5} {'Unit'}"
    else:
        hdr = f"  {'Metric':<30} {'PX4 SITL':>10} {'Unit'}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    n_pass = n_fail = 0
    for r in comparison_rows:
        sitl_str = f"{r['px4_sitl']:.4f}" if isinstance(r['px4_sitl'], float) else str(r['px4_sitl'])
        if has_ref:
            ref_str = f"{r['iris_ma6']:.4f}" if r['iris_ma6'] is not None else "N/A"
            diff_str = f"{r['diff_pct']:.1f}%" if r['diff_pct'] is not None else "N/A"
            flag = r['flag']
            if flag == "PASS":
                n_pass += 1
            elif flag == "FAIL":
                n_fail += 1
            print(f"  {r['metric']:<30} {ref_str:>10} {sitl_str:>10} {diff_str:>8} {flag:>5} {r['unit']}")
        else:
            print(f"  {r['metric']:<30} {sitl_str:>10} {r['unit']}")

    if has_ref:
        n_compared = n_pass + n_fail
        print()
        print(f"  Threshold: ±{threshold*100:.0f}%")
        print(f"  Compared: {n_compared}, PASS: {n_pass}, FAIL: {n_fail}")
        if n_fail > 0:
            print(f"  >>> MISMATCH DETECTED — {n_fail} metrics exceed ±{threshold*100:.0f}% threshold")
        else:
            print(f"  >>> ALL METRICS WITHIN ±{threshold*100:.0f}% — proceed with deployment")

    # Save as JSON
    json_path = output_dir / "sysid_metrics.json"
    json_out = {"sitl_metrics": {k: asdict(v) for k, v in all_metrics.items()}}
    if ref_metrics:
        json_out["reference_metrics"] = ref_metrics
        json_out["comparison"] = comparison_rows
        json_out["threshold"] = threshold
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults saved to {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
