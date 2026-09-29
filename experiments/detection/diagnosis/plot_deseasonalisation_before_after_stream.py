"""
Plot the AEMO TEST demand stream before and after deseasonalisation.

IMPORTANT:
    The seasonal profile is fitted using TRAIN only.
    It is then applied unchanged to TEST.

Top panel:
    Original TEST TOTALDEMAND.

Bottom panel:
    TEST TOTALDEMAND after applying the seasonal profile
    learned from TRAIN (deseasonalised residual).

The figure displays the first 365 days of TEST for
visualisation only. The seasonal profile itself is still
learned from the complete TRAIN period.

Outputs:
    results/figures/diagnosis/
        SA1_test_deseasonalisation_before_after_stream.png
        NSW1_test_deseasonalisation_before_after_stream.png
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

# Display one year of TEST data
DAYS_TO_PLOT = 365

N_OBSERVATIONS = (
    OBSERVATIONS_PER_DAY
    * DAYS_TO_PLOT
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
        # Apply TRAIN-fitted seasonal profile to TEST
        #
        # data=test:
        #     TEST is the stream being transformed.
        #
        # reference=train:
        #     Seasonal profile is learned ONLY from TRAIN.
        #
        # TEST is never used to fit the seasonal profile.
        # ---------------------------------------------------------

        test_ds = deseasonalise_demand_frame(
            data=test,
            reference=train,
        )

        # ---------------------------------------------------------
        # Select the same TEST observations before and after
        # ---------------------------------------------------------

        n_plot = min(
            N_OBSERVATIONS,
            len(test),
        )

        before = (
            test["TOTALDEMAND"]
            .iloc[:n_plot]
            .to_numpy(dtype=float)
        )

        after = (
            test_ds["TOTALDEMAND"]
            .iloc[:n_plot]
            .to_numpy(dtype=float)
        )

        # ---------------------------------------------------------
        # Get TEST timestamps
        # ---------------------------------------------------------

        if "SETTLEMENTDATE" in test.columns:

            timestamps = pd.to_datetime(
                test["SETTLEMENTDATE"]
                .iloc[:n_plot]
            )

        else:

            timestamps = pd.to_datetime(
                test.index[:n_plot]
            )

        # ---------------------------------------------------------
        # Plot
        # ---------------------------------------------------------

        fig, (
            ax_before,
            ax_after,
        ) = plt.subplots(
            2,
            1,
            figsize=(18, 9),
            sharex=True,
        )

        # ---------------------------------------------------------
        # BEFORE:
        # Original TEST demand
        # ---------------------------------------------------------

        ax_before.plot(
            timestamps,
            before,
            linewidth=0.8,
        )

        ax_before.set_title(
            f"{region} — TEST Before Deseasonalisation"
        )

        ax_before.set_ylabel(
            "Demand (MW)"
        )

        ax_before.grid(
            alpha=0.20
        )

        # ---------------------------------------------------------
        # AFTER:
        # TEST residual after TRAIN-fitted deseasonalisation
        # ---------------------------------------------------------

        ax_after.plot(
            timestamps,
            after,
            linewidth=0.8,
        )

        ax_after.axhline(
            0,
            linestyle="--",
            linewidth=1.0,
            label="Zero residual",
        )

        ax_after.set_title(
            f"{region} — TEST After Deseasonalisation "
            "(Residual Stream)"
        )

        ax_after.set_xlabel(
            "TEST date"
        )

        ax_after.set_ylabel(
            "Demand residual (MW)"
        )

        ax_after.grid(
            alpha=0.20
        )

        ax_after.legend()

        # ---------------------------------------------------------
        # Overall title
        # ---------------------------------------------------------

        fig.suptitle(
            f"{region} — TEST Demand Before vs "
            "After TRAIN-Fitted Deseasonalisation",
            fontsize=14,
        )

        fig.tight_layout()

        # ---------------------------------------------------------
        # Save
        # ---------------------------------------------------------

        output_path = (
            OUTPUT_DIR
            / f"{region}_test_deseasonalisation_before_after_stream.png"
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        # ---------------------------------------------------------
        # Print information
        # ---------------------------------------------------------

        start_time = (
            timestamps.iloc[0]
            if hasattr(timestamps, "iloc")
            else timestamps[0]
        )

        end_time = (
            timestamps.iloc[-1]
            if hasattr(timestamps, "iloc")
            else timestamps[-1]
        )

        print(f"\n{region}")

        print(
            "Seasonal profile fitted on TRAIN only."
        )

        print(
            "Seasonal profile applied to TEST."
        )

        print(
            f"Displayed TEST period: "
            f"{start_time} to {end_time}"
        )

        print(
            f"Displayed observations: "
            f"{n_plot:,}"
        )

        print(
            f"Before mean: "
            f"{np.mean(before):.2f} MW"
        )

        print(
            f"After mean:  "
            f"{np.mean(after):.2f} MW"
        )

        print(
            f"Saved: {output_path}"
        )


if __name__ == "__main__":
    main()