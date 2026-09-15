# Diagnostic T2 — Aggregated AEMO detections vs documented events

| resolution | detector | region | COVID delay | solar/mild-weather delay | 5-minute settlement delay | 2022 suspension delay | security-directions delay | tier1_unmatched | tier2_contextual | unmatched | precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3h_MAE | adwin | NSW1 | not detected | 12.979 | not detected | not detected | not detected | 10 | 2 | 8 | 27.27% |
| 3h_MAE | kswin | NSW1 | not detected | 25.604 | not detected | not detected | not detected | 4 | 0 | 4 | 20.00% |
| 3h_MAE | page_hinkley | NSW1 | not detected | not detected | not detected | not detected | not detected | 1 | 0 | 1 | 0.00% |
| 3h_MAE | adwin | SA1 | not detected | not detected | not detected | not detected | not detected | 9 | 2 | 7 | 22.22% |
| 3h_MAE | kswin | SA1 | not detected | not detected | not detected | not detected | not detected | 3 | 0 | 3 | 0.00% |
| 3h_MAE | page_hinkley | SA1 | not detected | not detected | not detected | not detected | not detected | 3 | 0 | 3 | 0.00% |
| 6h_MAE | adwin | NSW1 | not detected | 4.979 | not detected | not detected | not detected | 5 | 1 | 4 | 33.33% |
| 6h_MAE | kswin | NSW1 | not detected | not detected | not detected | not detected | not detected | 7 | 2 | 5 | 28.57% |
| 6h_MAE | page_hinkley | NSW1 | not detected | not detected | not detected | not detected | not detected | 0 | 0 | 0 | N/A |
| 6h_MAE | adwin | SA1 | not detected | not detected | not detected | not detected | not detected | 9 | 0 | 9 | 0.00% |
| 6h_MAE | kswin | SA1 | not detected | not detected | not detected | not detected | not detected | 6 | 0 | 6 | 0.00% |
| 6h_MAE | page_hinkley | SA1 | not detected | not detected | not detected | not detected | not detected | 1 | 1 | 0 | 100.00% |

This diagnostic table applies the same AEMO event-matching implementation and protocol as the official T2 evaluation, but to the frozen 3-hour and 6-hour aggregated detector outputs. Delays are fractional days from the frozen reference date, rounded to 3 decimals. Tier 1 unmatched counts detections that did not match a primary Tier 1 event. Tier 2 contextual reports how many of those detections were subsequently matched to Tier 2 events. Unmatched counts detections matched to neither tier. Precision is the proportion of detections matched to either Tier 1 or Tier 2. Historical events are contextual references; unmatched detections are not automatically false positives.
