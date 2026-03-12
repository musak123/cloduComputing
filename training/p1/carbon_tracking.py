from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CarbonResult:
    tracker: str
    duration_s: float
    energy_kwh: float
    emissions_kg_co2eq: float
    source: str


class CarbonTracker:
    """Unified wrapper for CodeCarbon and Eco2AI with safe fallback.

    Preference order:
    1) codecarbon
    2) eco2ai
    3) internal estimate fallback
    """

    def __init__(self, project_name: str, output_dir: str) -> None:
        self.project_name = project_name
        self.output_dir = output_dir
        self._backend: str = "fallback"
        self._codecarbon = None
        self._eco = None
        self._start = 0.0
        self._result: Optional[CarbonResult] = None

        try:
            from codecarbon import EmissionsTracker  # type: ignore

            self._backend = "codecarbon"
            self._codecarbon = EmissionsTracker(
                project_name=project_name,
                output_dir=output_dir,
                output_file=f"{project_name}_codecarbon.csv",
                save_to_file=True,
                log_level="error",
            )
            return
        except Exception:
            pass

        try:
            import eco2ai  # type: ignore

            self._backend = "eco2ai"
            self._eco = eco2ai.Tracker(
                project_name=project_name,
                file_name=f"{project_name}_eco2ai.csv",
                measure_period=10,
            )
        except Exception:
            self._backend = "fallback"

    def start(self) -> None:
        self._start = time.time()
        if self._backend == "codecarbon" and self._codecarbon is not None:
            self._codecarbon.start()
        elif self._backend == "eco2ai" and self._eco is not None:
            self._eco.start()

    def stop(self) -> CarbonResult:
        duration = max(time.time() - self._start, 0.0)

        if self._backend == "codecarbon" and self._codecarbon is not None:
            emissions_kg = float(self._codecarbon.stop() or 0.0)
            # Fallback estimate for kWh if raw consumption is unavailable.
            energy_kwh = emissions_kg / 0.475 if emissions_kg > 0 else duration * 0.35 / 3600
            self._result = CarbonResult(
                tracker="codecarbon",
                duration_s=round(duration, 6),
                energy_kwh=round(energy_kwh, 8),
                emissions_kg_co2eq=round(emissions_kg, 8),
                source="measured",
            )
            return self._result

        if self._backend == "eco2ai" and self._eco is not None:
            self._eco.stop()
            # Eco2AI stores values internally/file; use conservative estimate in-process.
            energy_kwh = duration * 0.35 / 3600
            emissions_kg = energy_kwh * 0.475
            self._result = CarbonResult(
                tracker="eco2ai",
                duration_s=round(duration, 6),
                energy_kwh=round(energy_kwh, 8),
                emissions_kg_co2eq=round(emissions_kg, 8),
                source="estimated_from_runtime",
            )
            return self._result

        # Fallback model (kept explicit for reproducibility in no-dependency environments).
        energy_kwh = duration * 0.35 / 3600
        emissions_kg = energy_kwh * 0.475
        self._result = CarbonResult(
            tracker="fallback",
            duration_s=round(duration, 6),
            energy_kwh=round(energy_kwh, 8),
            emissions_kg_co2eq=round(emissions_kg, 8),
            source="estimated_from_runtime",
        )
        return self._result

    def as_dict(self) -> Dict[str, object]:
        if self._result is None:
            return {}
        return {
            "tracker": self._result.tracker,
            "duration_s": self._result.duration_s,
            "energy_kwh": self._result.energy_kwh,
            "emissions_kg_co2eq": self._result.emissions_kg_co2eq,
            "source": self._result.source,
        }
