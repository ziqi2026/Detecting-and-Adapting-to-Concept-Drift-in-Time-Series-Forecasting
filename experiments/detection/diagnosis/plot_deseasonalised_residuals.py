"""
Plot the actual deseasonalised demand residual values
for a 14-day section of the AEMO TRAIN data.

Residual:
    Actual demand - Expected seasonal demand

Unlike an ACF plot, this figure shows actual consecutive
observations from the TRAIN dataset.

Outputs:
    results/figures/diagnosis/
        SA1_deseasonalised_residual_14days.png
        NSW1_deseasonalised_residual_14days.png
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from drift_lab.data_loader import load_and_split_aemo
from drift_lab.deseasonalise import deseasonalise_demand_frame


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

DATA_PATH = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "diagnosis"
)

REGIONS = (
    "SA1",
    "NSW1",
)

OBSERVATIONS_PER_DAY = 48

# Actual consecutive 14-day section
DAYS_TO_PLOT = 14
N_OBSERVATIONS = (
    DAYS_TO_PLOT
    * OBSERVATIONS_PER_DAY
)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for region in REGIONS:

        (
            _,
            train,
            calibration,
            test,
        ) = load_and_split_aemo(
            region=region,
            data_path=DATA_PATH,
        )

        # ---------------------------------------------------------
        # Deseasonalise TRAIN demand
        # ---------------------------------------------------------

        train_ds = deseasonalise_demand_frame(
            data=train,
            reference=train,
        )

        # Actual demand
        actual = (
            train["TOTALDEMAND"]
            .to_numpy(dtype=float)
        )

        # Deseasonalised residual
        residual = (
            train_ds["TOTALDEMAND"]
            .to_numpy(dtype=float)
        )

        # ---------------------------------------------------------
        # Select actual first 14 consecutive days from TRAIN
        # ---------------------------------------------------------

        actual_plot = actual[
            :N_OBSERVATIONS
        ]

        residual_plot = residual[
            :N_OBSERVATIONS
        ]

        x = np.arange(
            N_OBSERVATIONS
        )

        # ---------------------------------------------------------
        # Plot residual values
        # ---------------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(16, 6)
        )

        ax.plot(
            x,
            residual_plot,
            linewidth=1.2,
            label="Deseasonalised demand residual",
        )

        # Zero residual reference
        ax.axhline(
            0,
            linewidth=1.0,
            linestyle="--",
            label="Zero residual",
        )

        # ---------------------------------------------------------
        # Daily boundaries
        # ---------------------------------------------------------

        for day in range(
            1,
            DAYS_TO_PLOT
        ):

            ax.axvline(
                day * OBSERVATIONS_PER_DAY,
                linestyle=":",
                alpha=0.25,
            )

        # ---------------------------------------------------------
        # X-axis
        # ---------------------------------------------------------

        day_positions = np.arange(
            0,
            N_OBSERVATIONS + 1,
            OBSERVATIONS_PER_DAY,
        )

        day_labels = [
            f"Day {day}"
            for day in range(
                1,
                DAYS_TO_PLOT + 2,
            )
        ]

        ax.set_xticks(
            day_positions
        )

        ax.set_xticklabels(
            day_labels,
            rotation=45,
            ha="right",
        )

        ax.set_title(
            f"{region} — Deseasonalised Demand Residual "
            "(14 Consecutive TRAIN Days)"
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            "Demand residual (MW)"
        )

        ax.set_xlim(
            0,
            N_OBSERVATIONS - 1,
        )

        ax.grid(
            alpha=0.20
        )

        ax.legend()

        fig.tight_layout()

        # ---------------------------------------------------------
        # Save
        # ---------------------------------------------------------

        output_path = (
            OUTPUT_DIR
            / f"{region}_deseasonalised_residual_14days.png"
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        # ---------------------------------------------------------
        # Print residual summary
        # ---------------------------------------------------------

        print(f"\n{region}")

        print(
            f"Residual mean: "
            f"{np.mean(residual):.4f} MW"
        )

        print(
            f"Residual std:  "
            f"{np.std(residual):.4f} MW"
        )

        print(
            f"Residual min:  "
            f"{np.min(residual):.4f} MW"
        )

        print(
            f"Residual max:  "
            f"{np.max(residual):.4f} MW"
        )

        print(
            f"Saved: {output_path}"
        )


if __name__ == "__main__":
    main()