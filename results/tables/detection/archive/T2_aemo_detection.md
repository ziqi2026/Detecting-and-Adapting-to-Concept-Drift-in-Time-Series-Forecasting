# T2 — Detection on AEMO vs documented events

| detector | region | COVID delay | solar/mild-weather delay | 2022 suspension delay | security-directions delay | detections | tier1_matched | tier1_precision | tier2_matched | tier2_precision | unmatched | tier1_chance_matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adwin | SA1 | 62.979 | 14.979 | not detected | 27.646 | 16 | 3 | 18.75% | 3 | 18.75% | 10 | 0.308 |
| page_hinkley | SA1 | not detected | not detected | not detected | 33.542 | 9 | 1 | 11.11% | 2 | 22.22% | 6 | 0.176 |
| kswin | SA1 | not detected | 15.812 | not detected | 33.646 | 28 | 2 | 7.14% | 3 | 10.71% | 23 | 0.523 |
| adwin | NSW1 | not detected | 4.312 | not detected | not detected | 22 | 1 | 4.55% | 2 | 9.09% | 19 | 0.417 |
| page_hinkley | NSW1 | not detected | not detected | not detected | not detected | 10 | 0 | 0.00% | 1 | 10.00% | 9 | 0.195 |
| kswin | NSW1 | 21.542 | 25.688 | not detected | not detected | 30 | 2 | 6.67% | 4 | 13.33% | 24 | 0.558 |

Delays are fractional days from the frozen reference date, rounded to 3 decimals. Tier 1 precision is the proportion of detections matched to Tier 1 events. Tier 2 precision is the proportion of detections matched to Tier 2 events. Unmatched counts detections matched to neither tier. Tier 1 chance matches is the expected number of Tier 1 event matches under randomly placed detections. Historical events are contextual references; unmatched detections are not automatically false positives. See docs/aemo_matching_protocol.md and T2_manifest.json.
