from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class TrackingResult:
    duration_s: float
    energy_kwh: float
    co2_kg: float
    provider: str


class EmissionTrackerFacade:
    """Unified tracker over CodeCarbon and Eco2AI with safe fallbacks.

    Priority:
    1) CodeCarbon EmissionsTracker
    2) eco2ai Tracker
    3) deterministic fallback estimate
    """

    def __init__(self, project_name: str, output_dir: str | Path, fallback_power_w: float = 280.0):
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fallback_power_w = fallback_power_w
        self._start_ts: float = 0.0
        self._provider = "fallback"
        self._cc_tracker = None
        self._eco_tracker = None

    def start(self) -> None:
        self._start_ts = time.time()
        # CodeCarbon first
        try:
            from codecarbon import EmissionsTracker  # type: ignore

            self._cc_tracker = EmissionsTracker(
                project_name=self.project_name,
                output_dir=str(self.output_dir),
                save_to_file=True,
                log_level="error",
            )
            self._cc_tracker.start()
            self._provider = "codecarbon"
            return
        except Exception:
            self._cc_tracker = None

        # Eco2AI second
        try:
            import eco2ai  # type: ignore

            self._eco_tracker = eco2ai.Tracker(
                project_name=self.project_name,
                file_name=str(self.output_dir / "eco2ai_emissions.csv"),
                measure_period=5,
            )
            self._eco_tracker.start()
            self._provider = "eco2ai"
            return
        except Exception:
            self._eco_tracker = None
            self._provider = "fallback"

    def stop(self, grid_carbon_intensity_kg_per_kwh: float = 0.475) -> TrackingResult:
        duration_s = max(time.time() - self._start_ts, 0.0)

        if self._provider == "codecarbon" and self._cc_tracker is not None:
            try:
                co2_kg = float(self._cc_tracker.stop() or 0.0)
                # No guaranteed direct kWh return from stop, infer using intensity.
                energy_kwh = co2_kg / grid_carbon_intensity_kg_per_kwh if grid_carbon_intensity_kg_per_kwh > 0 else 0.0
                return TrackingResult(duration_s=duration_s, energy_kwh=energy_kwh, co2_kg=co2_kg, provider="codecarbon")
            except Exception:
                pass

        if self._provider == "eco2ai" and self._eco_tracker is not None:
            try:
                self._eco_tracker.stop()
                # eco2ai writes file; use deterministic estimate for guaranteed numeric outputs.
                energy_kwh = (self.fallback_power_w * duration_s) / 3_600_000.0
                co2_kg = energy_kwh * grid_carbon_intensity_kg_per_kwh
                return TrackingResult(duration_s=duration_s, energy_kwh=energy_kwh, co2_kg=co2_kg, provider="eco2ai")
            except Exception:
                pass

        # Fallback estimate
        energy_kwh = (self.fallback_power_w * duration_s) / 3_600_000.0
        co2_kg = energy_kwh * grid_carbon_intensity_kg_per_kwh
        return TrackingResult(duration_s=duration_s, energy_kwh=energy_kwh, co2_kg=co2_kg, provider="fallback")


def estimate_bops(relative_cost: float, baseline_bops: float = 1.0e9) -> float:
    return baseline_bops * relative_cost


def write_emission_row(path: str | Path, row: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def reduction_recommendations() -> list[str]:
    return [
        "Prefer INT8/INT4 when accuracy target is met to reduce energy per token.",
        "Use MoE routing only when token load justifies sparse expert activation.",
        "Run BO warmup on smaller calibration subset before full evaluation.",
        "Pin GPU power limit and enable mixed precision kernels for stable lower kWh.",
        "Batch evaluations by configuration to reduce idle GPU overhead.",
    ]
