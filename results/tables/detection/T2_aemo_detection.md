# T2 — Detection on AEMO vs documented events

| detector | region | COVID delay | solar/mild-weather delay | 5-minute settlement delay | 2022 suspension delay | security-directions delay | tier1_unmatched | tier2_contextual | unmatched | precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adwin | SA1 | 0.312 | 3.646 | not detected | 4.646 | 5.646 | 139 | 14 | 125 | 12.59% |
| page_hinkley | SA1 | not detected | 14.771 | not detected | not detected | not detected | 10 | 2 | 8 | 27.27% |
| kswin | SA1 | 0.208 | 2.271 | not detected | not detected | 2.271 | 294 | 17 | 277 | 6.73% |
| adwin | NSW1 | 4.312 | 0.312 | 3.646 | 5.979 | 10.312 | 156 | 7 | 149 | 7.45% |
| page_hinkley | NSW1 | not detected | 27.896 | not detected | not detected | not detected | 3 | 0 | 3 | 25.00% |
| kswin | NSW1 | 2.396 | 0.25 | 2.875 | 1.458 | 3.292 | 287 | 15 | 272 | 6.85% |

Delays are fractional days from the frozen reference date, rounded to 3 decimals. Tier 1 unmatched counts detections that did not match a primary Tier 1 event. Tier 2 contextual reports how many of those detections were subsequently matched to Tier 2 events. Unmatched counts detections matched to neither tier. Precision is the proportion of detections matched to either Tier 1 or Tier 2. Historical events are contextual references; unmatched detections are not automatically false positives. See docs/aemo_matching_protocol.md and T2_manifest.json.
