# Shared Experimental Decisions — Teams 41 and 42

This document records the settings agreed between Teams 41 and 42 following the supervisor's feedback. These settings will be kept consistent so that the main experimental difference remains:

- **Team 41:** detection on the demand distribution.
- **Team 42:** detection on the forecast-error stream.

## 1. Data and Evaluation Period

- **Regions:** SA1 and NSW1.
- **TRAIN:** 1 Jan 2018 – 31 Dec 2019.
- **Calibration:** 1 Jan 2020 – 29 Feb 2020.
- **TEST:** from 1 Mar 2020.
- Both teams will use the same periods.

## 2. Event Catalogue

- Both teams will use the same `events.csv`.
- The same Tier-1/Tier-2 definitions and event dates will be fixed before the final detector runs.
- **COVID-19 event:** 16 Mar 2020 – 17 May 2020.
- **Five-Minute Settlement (1 Oct 2021):** removed from the target event catalogue because it is a price-market rule change and should not count as a successful demand-drift match.

## 3. Event-Matching Rule

- A detection must occur **at or after the event start**.
- A detection may match until **event end + 7 days**.
- Therefore, the agreed evaluation grace period is **7 days**.
- This 7-day grace period is only for event matching; it is separate from the refractory period.

## 4. Deseasonalisation

- Deseasonalisation parameters will be fitted using **TRAIN only**.
- Both teams will use the same seasonal structure and fitting period.
- The implementation will use:
  - a smoothed **day-of-year (DOY)** component; and
  - a **day-of-week × half-hour (DOW × TOD)** component.
- Detection will operate on the resulting deseasonalised representation.

## 5. Standardisation

- Detector inputs will be **Z-score standardised using TRAIN only**.
- Team 41 will standardise its demand-based detector input.
- Team 42 will standardise its forecast-error detector input.
- No Calibration or TEST statistics will be used to fit the standardisation.

## 6. Synthetic Calibration

- Detector configurations will be selected on the **synthetic benchmark before AEMO evaluation**.
- Agreed false-alarm budget: **≤2 false alarms per year**.
- The no-drift synthetic control must also be satisfied.
- Both teams will use the same **five seeds: 1–5**.
- Once selected, detector parameters will be **frozen** and applied to AEMO unchanged.

## 7. Refractory Period

- Agreed refractory period: **14 days**.
- It applies after **every accepted detection**, not only detections that match documented events.
- After one accepted detection, alarms occurring during the following 14 days are suppressed.
- The purpose is to reduce repeated detections of the same sustained change and prevent excessive retraining in Arms C/D.
- The **14-day refractory period and 7-day event-matching grace period are independent**:
  - **7 days:** evaluation/event matching.
  - **14 days:** suppression of repeated detections.

## 8. Detector Warm-Up

- Detectors must be warmed before TEST rather than starting cold on 1 Mar 2020.
- The supervisor originally specified warming on TRAIN.
- The proposed shared implementation is to use **Jan–Feb 2020 Calibration** as the immediate pre-TEST warm-up period.
- During warm-up:
  - detector state is updated chronologically;
  - alarms are not recorded;
  - detector state is retained;
  - recording begins on **1 Mar 2020**.
- The exact warm-up period remains to be confirmed between both teams before the final runs.

## 9. T2 Reporting

- Keep the existing Tier-1 event columns.
- Keep unmatched detections.
- Report **Tier-1 and Tier-2 precision separately**.
- Add the supervisor's **chance-matching baseline**:

  `K × [1 − (1 − w/T)^N]`

  where:
  - `N` = number of detections;
  - `K` = number of events;
  - `w` = matching window in days;
  - `T` = total evaluation duration in days.

## 10. Shared Evaluation and Logging

- Both teams will use `drift_lab.evaluation`.
- Every run will write a row to `runs.csv` through the shared logging harness.

## 11. Detection Before Adaptation

- Detector configurations must be calibrated and agreed before running **Arms C and D**.
- The detector configuration carried forward to adaptation should produce an operationally plausible retraining count: **a handful per year, not hundreds**.
- Arms C and D should only be run after the shared detector configuration has been finalised.

## 12. Adaptation Settings

- **Arm A — Never Retrain:** use the initial model throughout the TEST period with no retraining.
- **Arm B — Periodic Retraining:** retrain at a fixed **30-day interval**.
- **Arm C — Drift-Triggered Full History:** retrain when an accepted drift is detected, using all available historical data up to that point.
- **Arm D — Drift-Triggered Recent Window:** retrain when an accepted drift is detected, using only recent historical data. Evaluate training windows of **30, 60, 120, and 180 days**.
- For Arm D, the window determines **how much recent data is used at each drift-triggered retraining**, not how frequently retraining occurs.

## Shared Comparison Principle

All applicable settings above should be identical between the teams.

The intended experimental difference is only:

- **Team 41 = demand-based detection**
- **Team 42 = forecast-error-based detection**