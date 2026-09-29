# T2 — Detection on AEMO vs documented events

| detector | region | COVID delay | solar/mild-weather delay | 2022 suspension delay | security-directions delay | detections | tier1_matched | tier1_precision | tier2_matched | tier2_precision | unmatched | tier1_chance_matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adwin | SA1 | 58.312 | 14.979 | not detected | not detected | 13 | 2 | 15.38% | 1 | 7.69% | 10 | 0.252 |
| page_hinkley | SA1 | 16.146 | not detected | not detected | not detected | 5 | 1 | 20.00% | 0 | 0.00% | 4 | 0.099 |
| kswin | SA1 | not detected | 15.917 | 14.75 | not detected | 17 | 2 | 11.76% | 2 | 11.76% | 13 | 0.327 |
| adwin | NSW1 | not detected | not detected | not detected | not detected | 16 | 0 | 0.00% | 2 | 12.50% | 14 | 0.308 |
| page_hinkley | NSW1 | not detected | not detected | not detected | not detected | 5 | 0 | 0.00% | 0 | 0.00% | 5 | 0.099 |
| kswin | NSW1 | 21.438 | not detected | not detected | not detected | 25 | 1 | 4.00% | 3 | 12.00% | 21 | 0.471 |

Delays are fractional days from the frozen reference date, rounded to 3 decimals. Tier 1 precision is the proportion of detections matched to Tier 1 events. Tier 2 precision is the proportion of detections matched to Tier 2 events. Unmatched counts detections matched to neither tier. Tier 1 chance matches is the expected number of Tier 1 event matches under randomly placed detections. Historical events are contextual references; unmatched detections are not automatically false positives. See docs/aemo_matching_protocol.md and T2_manifest.json.
