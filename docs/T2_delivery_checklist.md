# T2 — 验收与文件位置 / Acceptance and file locations

Current protocol: `aemo_t1_7day_one_to_one_v2`.

The current implementation follows the user's instruction to reuse the
project's T1 seven-day matching rule. Tier assignments and the stored Task 120
inputs remain byte-identical to their committed versions.

当前实现按用户要求复用 T1 的七天匹配规则。冻结事件目录、分层标准和 Task 120
输入均与已提交版本字节一致。所有验收项已通过；本清单不表示已提交或推送 Git。

## Acceptance / 验收

| Requirement / 要求 | Status / 结果 | Evidence / 依据 |
| --- | --- | --- |
| Match Task 120 detections using the specified rule | PASS | Six stored runs; all 1,060 indices, timestamps, detector configurations and normalized errors validated. T1 tolerance and first-unused matching are shared with T2. |
| Include all five Tier 1 reference columns | PASS | The final table contains COVID, solar/mild-weather, Five-Minute Settlement, suspension and security-directions columns. |
| Report delays in days | PASS | Full-precision fractional days in the event detail and log; three decimal places in the display table. |
| Report `not detected` where no match exists | PASS | Explicit statuses retained for every applicable Tier 1 event; zero-detection behavior is tested. |
| Check Tier 2 before classifying as unmatched | PASS | Complete Tier 1 allocation first, then match unused detections against Tier 2. Each event and detection is used at most once. |
| Calculate unmatched and precision consistently | PASS | 19 Tier 1 + 59 Tier 2 + 982 unmatched = 1,060. Precision is matched detections in either tier divided by all detections for that detector-region run. |
| Implement reusable logic in evaluation.py | PASS | `evaluate_aemo_detections`, catalogue validation, event windows and shared T1/T2 matching primitive. |
| Log metrics through run_logger.py with three regimes | PASS | 3,141 current-version rows in `results/runs.csv`; only pre-drift, drift and post-drift regime labels. |
| Produce T2 for SA1 and NSW1 | PASS | Six final rows: ADWIN, Page-Hinkley and KSWIN for each region. |

Events remain contextual historical references. Unmatched detections are not
automatically false positives; the precision column is a catalogue-match
proportion under the specified one-to-one rule.

历史事件作为背景参考；unmatched 不直接等同于 false positive。Precision 为既定
一对一规则下的目录匹配比例，不代表已知 ground truth 上的分类准确性。

## File locations / 文件位置

All paths below are relative to the repository root.

| Location / 位置 | Purpose / 用途 |
| --- | --- |
| `src/drift_lab/evaluation.py` | Shared T1/T2 evaluation functions / 共享评估逻辑 |
| `src/drift_lab/run_logger.py` | Existing logger reused without schema changes / 复用现有日志组件 |
| `experiments/detection/run_aemo_t2_evaluation.py` | Evaluation and reconstruction runner / 评估与表格重建脚本 |
| `test/detection/test_aemo_t2_evaluation.py` | Fourteen regression and integration tests / 14 项测试 |
| `data/events.csv` | Frozen input, unchanged / 冻结事件目录，未改动 |
| `results/tables/detection/aemo_detection_changepoints.csv` | Task 120 input, unchanged / 已保存的检测输入，未改动 |
| `results/tables/detection/T2_aemo_detection.csv` | Current final T2 / 当前最终表格 |
| `results/tables/detection/T2_aemo_detection.md` | Readable final T2 / 可阅读版表格 |
| `results/tables/detection/T2_detection_matches.csv` | One audit row per detection / 逐检测匹配明细 |
| `results/tables/detection/T2_tier1_event_details.csv` | Thirty detector-region-event status/delay rows / 五事件明细 |
| `results/tables/detection/T2_detection_summary.csv` | Counts and precision for each run / 每组匹配计数与 precision |
| `results/tables/detection/T2_manifest.json` | Input/code hashes and run identities / 哈希与运行配置 |
| `results/runs.csv` | Required shared long-format metric log / 要求指定的长格式日志 |
| `docs/aemo_matching_protocol.json` | Executable matching configuration / 可执行匹配配置 |
| `docs/aemo_matching_protocol.md` | Rule interpretation and version record / 匹配规则与版本说明 |
| `docs/T2_process_zh_en.md` | Bilingual method and results / 中英双语过程说明 |
| `results/tables/detection/archive/aemo_context_v1/` | Superseded snapshots, historical only / 旧版归档，不作为当前交付 |

## Logging and reproduction / 日志与复现

The log retains 3,141 superseded rows and 3,141 current rows (6,282 total).
The current split ID is
`aemo_fixed_train2018_2019_test202003_task120_t2_t1rule_v2`.
The reconstruction code also selects the configuration hashes in the current
manifest. Do not aggregate versions together or sum overlapping event windows.
Current T2 counts are reconstructed from the disjoint `t2.global.*` metrics.

日志中保留旧版和新版各 3,141 行。正式 T2 只使用上述新版 split_id 和 manifest
中的配置哈希。逐事件 regime 沿用 F1 的前 7 天、完整事件期、后 7 天；事件窗口
重叠，不能直接相加。全期计数使用互斥的 `t2.global.*` 指标。

Run from the repository root:

```sh
python3 experiments/detection/run_aemo_t2_evaluation.py
python3 experiments/detection/run_aemo_t2_evaluation.py --from-logs
PYTHONPATH=src python3 -m unittest discover -s test/detection -p test_aemo_t2_evaluation.py -v
PYTHONPATH=src/drift_lab python3 test/detection/test_evaluation_detection.py
```

Verified: 14 tests and the original synthetic checks pass; all six real runs
match T1's assignment behavior; rerunning does not append duplicate metrics;
reconstruction produces a byte-identical T2 CSV; current manifest hashes match.

已验证：14 项测试及原有检查通过；六组真实结果与 T1 匹配行为一致；重复执行不会
重复记账；日志重建的 T2 CSV 字节一致；manifest 的文件哈希匹配。
