"""
Plot ACF of AEMO TRAIN demand before and after deseasonalisation.

The ACF is calculated from the full TRAIN demand series.
The plot shows lag distances up to 14 days:

    48 observations  = 1-day lag
    336 observations = 7-day lag
    672 observations = 14-day lag

These are lag distances, not specific days or weeks selected
from the TRAIN dataset.

Outputs:
    results/figures/diagnosis/
        SA1_acf_before_after_deseasonalisation_14day_lag.png
        NSW1_acf_before_after_deseasonalisation_14day_lag.png
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf


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

# Maximum lag distance:
# 672 half-hour observations = 14 days
MAX_LAG = 672


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
        # Original full TRAIN demand
        # ---------------------------------------------------------

        before = (
            train["TOTALDEMAND"]
            .to_numpy(dtype=float)
        )

        # ---------------------------------------------------------
        # TRAIN-fitted deseasonalisation
        # ---------------------------------------------------------

        train_ds = deseasonalise_demand_frame(
            data=train,
            reference=train,
        )

        after = (
            train_ds["TOTALDEMAND"]
            .to_numpy(dtype=float)
        )

        # ---------------------------------------------------------
        # ACF
        #
        # Calculated using the full TRAIN series.
        # MAX_LAG only controls the maximum lag distance shown.
        # ---------------------------------------------------------

        before_acf = acf(
            before,
            nlags=MAX_LAG,
            fft=True,
        )

        after_acf = acf(
            after,
            nlags=MAX_LAG,
            fft=True,
        )

        lags = np.arange(
            MAX_LAG + 1
        )

        # ---------------------------------------------------------
        # Plot
        # ---------------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(16, 6)
        )

        ax.plot(
            lags,
            before_acf,
            label="Before deseasonalisation",
            linewidth=1.5,
        )

        ax.plot(
            lags,
            after_acf,
            label="After deseasonalisation",
            linewidth=1.5,
        )

        # ---------------------------------------------------------
        # Daily lag reference lines
        #
        # 48  = values 1 day apart
        # 96  = values 2 days apart
        # ...
        # 672 = values 14 days apart
        # ---------------------------------------------------------

        for day in range(1, 15):

            lag = (
                day
                * OBSERVATIONS_PER_DAY
            )

            # 7-day and 14-day lags are highlighted separately
            if lag not in (
                336,
                672,
            ):

                ax.axvline(
                    lag,
                    linestyle=":",
                    alpha=0.25,
                )

        # ---------------------------------------------------------
        # 7-day lag
        # ---------------------------------------------------------

        ax.axvline(
            336,
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label="7-day lag (336)",
        )

        # ---------------------------------------------------------
        # 14-day lag
        # ---------------------------------------------------------

        ax.axvline(
            672,
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label="14-day lag (672)",
        )

        # ---------------------------------------------------------
        # Zero autocorrelation reference
        # ---------------------------------------------------------

        ax.axhline(
            0,
            linewidth=0.8,
        )

        # ---------------------------------------------------------
        # X-axis shows lag distance in days
        # ---------------------------------------------------------

        day_lags = np.arange(
            0,
            MAX_LAG + 1,
            OBSERVATIONS_PER_DAY,
        )

        day_labels = [
            "0"
        ] + [
            f"{day}d"
            for day in range(1, 15)
        ]

        ax.set_xticks(
            day_lags
        )

        ax.set_xticklabels(
            day_labels,
            rotation=45,
            ha="right",
        )

        ax.set_title(
            f"{region} — Demand ACF Before vs "
            "After Deseasonalisation "
            "(Lag Distance up to 14 Days)"
        )

        ax.set_xlabel(
            "Lag distance (days)"
        )

        ax.set_ylabel(
            "Autocorrelation"
        )

        ax.set_xlim(
            0,
            MAX_LAG,
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
            / f"{region}_acf_before_after_deseasonalisation_14day_lag.png"
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        # ---------------------------------------------------------
        # Print important lag distances
        # ---------------------------------------------------------

        print(f"\n{region}")

        print(
            f"Lag 1   (30 min apart): "
            f"{before_acf[1]:.4f} -> "
            f"{after_acf[1]:.4f}"
        )

        print(
            f"Lag 48  (1 day apart):  "
            f"{before_acf[48]:.4f} -> "
            f"{after_acf[48]:.4f}"
        )

        print(
            f"Lag 336 (7 days apart): "
            f"{before_acf[336]:.4f} -> "
            f"{after_acf[336]:.4f}"
        )

        print(
            f"Lag 672 (14 days apart): "
            f"{before_acf[672]:.4f} -> "
            f"{after_acf[672]:.4f}"
        )

        print(
            f"Saved: {output_path}"
        )


if __name__ == "__main__":
    main()