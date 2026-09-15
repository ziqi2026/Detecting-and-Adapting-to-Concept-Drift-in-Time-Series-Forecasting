# T2 — Detection on AEMO vs documented events

| detector | region | COVID delay | solar/mild-weather delay | 5-minute settlement delay | 2022 suspension delay | security-directions delay | unmatched | precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adwin | SA1 | 14.312 | 3.646 | not detected | 4.646 | 2.979 | 123 | 10.22% |
| page_hinkley | SA1 | not detected | not detected | not detected | not detected | not detected | 13 | 18.75% |
| kswin | SA1 | 0.958 | 1.688 | 1.188 | 2.104 | 0.625 | 359 | 6.75% |
| adwin | NSW1 | 4.979 | 2.979 | 5.646 | not detected | 11.646 | 120 | 9.77% |
| page_hinkley | NSW1 | not detected | 7.146 | not detected | not detected | not detected | 11 | 21.43% |
| kswin | NSW1 | 0.938 | 0.083 | 0.083 | 1.708 | 1.646 | 356 | 5.07% |

Delays are fractional days from the frozen reference date, rounded to 3 decimals. Precision is the proportion of detections matched to either tier. Historical events are contextual references; unmatched detections are not automatically false positives. See docs/aemo_matching_protocol.md and T2_manifest.json.
