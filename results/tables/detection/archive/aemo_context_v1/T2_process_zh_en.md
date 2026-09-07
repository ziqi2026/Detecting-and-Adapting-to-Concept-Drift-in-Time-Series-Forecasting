# T2 — 中英过程说明 / Bilingual process and results

## 1. 输入核实 / Verify the inputs

中文：本次直接使用 Task 120 已保存的
`results/tables/detection/aemo_detection_changepoints.csv`。共 1,060 个检测，
覆盖 ADWIN、Page-Hinkley、KSWIN 与 SA1、NSW1 六种组合。逐项核对了检测索引与
`load_and_split_aemo()` 返回的测试序列时间戳、检测配置和标准化绝对误差。
两地区均为训练 35,039 行、校准 2,880 行、测试 67,249 行；测试时间为
2020-03-01 00:00 至 2024-01-01 00:00，与原加载器的区间结束时间口径一致。
本次没有重新训练预测模型或重新运行检测器。

English: This evaluation uses the stored Task 120 file directly. Its 1,060
detections cover three detectors and two regions. Every index was checked
against the test timestamps returned by `load_and_split_aemo()`. Detector
configurations and the calibration-normalized absolute errors also reproduced.
Each region has 35,039 training, 2,880 calibration, and 67,249 test observations.
The test stream runs from 2020-03-01 00:00 to 2024-01-01 00:00, following the
loader's interval-ending timestamps. No forecasting or detection was rerun.

## 2. 固定目录和时间规则 / Fix the catalogue and temporal rule

中文：保留原 `data/events.csv` 全部 34 个事件，即 5 个 Tier 1 和 29 个 Tier 2。
用户提供的分层标准规定了优先级，但没有指定时间容差和重复检测的计数方式。
因此，在计算本次真实匹配前，先保存了
`docs/aemo_matching_protocol.json` 和 `docs/aemo_matching_protocol.md`，
明确标注这些时间及计数规则是本次新增的评估约定；没有根据匹配结果调整窗口或 tier。

English: All 34 frozen events are retained: five Tier 1 and 29 Tier 2 events.
The supplied criteria specify tier precedence, but leave temporal tolerance
and repeated-detection accounting unspecified. Before calculating real-data
matches, this evaluation recorded those choices in the protocol JSON and
Markdown files. These are explicitly identified as new evaluation conventions,
not previously existing project rules. Neither windows nor tiers were tuned
after inspecting matches.

| Tier 1 ID | Reference / 参考事件 | Start / 起日 | End / 止日 | Matching interpretation / 匹配类型 |
| --- | --- | --- | --- | --- |
| EVT-2020-02 | COVID-19 demand change | 2020-04-01 | 2020-05-17 | Interval / 区间 |
| EVT-2020-07 | Solar and mild-weather demand erosion | 2020-12-01 | 2020-12-31 | Month / 月份 |
| EVT-2021-10 | Five-Minute Settlement | 2021-10-01 | 2021-10-01 | Point / 点事件 |
| EVT-2022-03 | NEM spot market suspension | 2022-06-15 | 2022-06-24 | Point at commencement; end is duration context / 开始日点事件 |
| EVT-2023-14 | SA and NSW security directions | 2023-10-27 | 2023-11-25 | Interval / 区间 |

中文：无时区时间按固定 AEST（UTC+10）解释。点事件窗口为
`[起日, 起日 + 7天)`；区间和月份事件为
`[起日, 结束日次日 + 7天)`。左端包含、右端不包含。2022 暂停事件遵循目录
`date_precision=day` 和开始日说明，不使用整个暂停持续期扩大点事件匹配窗口。

English: Naive timestamps are interpreted as fixed AEST (UTC+10). Point
events use `[start, start + 7 days)`; interval and month events use
`[start, day after end + 7 days)`. The left boundary is included and the
right boundary excluded. The 2022 suspension follows the catalogue's
`date_precision=day` and commencement note; its duration does not extend
the point-event matching window.

## 3. 分层匹配 / Assign detections by tier

中文：每个 detector × region 独立评估，每个检测先检查适用的 Tier 1；没有
Tier 1 候选时再检查 Tier 2，最后才标为 unmatched。同级窗口重叠时，先选起日
较早的事件，再按 event_id 排序。每个检测只归属一个事件；一个事件可匹配多个
检测。匹配明细保留所有候选 ID。本次 21 个检测存在多个候选，均按同一规则处理。

English: Each detector-region run is evaluated independently. For every
detection, applicable Tier 1 windows are checked first, followed by Tier 2.
Only detections matching neither tier are unmatched. Overlaps within a tier
are resolved by earliest start and then event ID. Each detection has one
assignment, while an event may explain multiple detections. All candidate
IDs are retained in the audit; 21 detections had multiple candidates.

## 4. 延迟和 precision / Calculate delays and precision

中文：每个 Tier 1 延迟为最早归属检测时间减去该事件起日，除以 24 小时，保留
小数天。无匹配时填 `not detected`。Precision 是
`(Tier 1匹配检测数 + Tier 2匹配检测数) / 检测总数`，分子不是事件数量。
例如 ADWIN/SA1 为 `(10 + 18) / 137 = 20.44%`，unmatched 为 109。

English: Tier 1 delay is the elapsed time from the reference start to the
earliest assigned detection, expressed in fractional days. An event with
no assigned detection is reported as `not detected`. Precision is
`(Tier 1 matched detections + Tier 2 matched detections) / all detections`.
The numerator counts detections, not unique events. For ADWIN/SA1 this is
`(10 + 18) / 137 = 20.44%`, with 109 unmatched detections.

| Detector | Region | All detections | Tier 1 matched | Tier 2 matched | Unmatched | Precision |
| --- | --- | --- | --- | --- | --- | --- |
| adwin | SA1 | 137 | 10 | 18 | 109 | 20.44% |
| page_hinkley | SA1 | 16 | 0 | 3 | 13 | 18.75% |
| kswin | SA1 | 385 | 39 | 47 | 299 | 22.34% |
| adwin | NSW1 | 133 | 15 | 15 | 103 | 22.56% |
| page_hinkley | NSW1 | 14 | 1 | 2 | 11 | 21.43% |
| kswin | NSW1 | 375 | 40 | 35 | 300 | 20.00% |
| Total | Both | 1060 | 105 | 120 | 835 | — |

## 5. Regime 日志 / Log metrics by regime

中文：沿用 F1 的逐事件定义：pre-drift 为事件前 7 天，drift 为完整事件持续期，
post-drift 为结束后的 7 天。通过现有 `run_logger.log_run()` 写入
`results/runs.csv`，本次共 3,141 行；method 保留真实 detector 名，事件 ID 写入
metric_name。日志还记录每个窗口实际覆盖的观测数量，避免把测试期外的空窗口
误读为已经观察到的零检测期。事件窗口可能重叠，不能相加得到全期总数。

English: Event-relative regimes follow F1: seven days before the event,
the full documented event duration, and seven days after it. The existing
`run_logger.log_run()` wrote 3,141 rows to `results/runs.csv`, retaining
the actual detector name and encoding the event ID in the metric name.
Observation exposure is logged for each window, including truncated or
unobserved windows. Overlapping event-window counts must not be summed
to obtain full-stream totals.

中文：为使最终 T2 完全由日志重建，另用 `t2.global.*` 记录互斥的全测试期计数。
具体优先级为：在任一事件持续期内则 drift；否则在任一事件前 7 天则 pre-drift；
否则若已过任一事件起日则 post-drift；其余为 pre-drift。这一额外计数口径在匹配
前已记录，用于保留事件之间的检测，不能与逐事件窗口混加。T2 将这些互斥计数
相加后计算 precision，而不是平均各 regime 的 precision。

English: A separate `t2.global.*` layer partitions the entire test stream
for reproducible accounting. Event-duration membership takes precedence
(`drift`), followed by membership in an event's preceding seven days
(`pre-drift`); otherwise timestamps after any applicable event start are
`post-drift`, and earlier timestamps are `pre-drift`. This additional
accounting convention was documented before matching. It retains detections
between events and must not be mixed with overlapping event-window counts.
T2 sums the disjoint counts and computes a ratio; it does not average regime
precision values.

## 6. 结果解释 / Interpret the results

中文：KSWIN 在 SA1 和 NSW1 均匹配到 5/5 个 Tier 1；ADWIN 在两地区均为 4/5。
ADWIN/SA1 未匹配到 Five-Minute Settlement，ADWIN/NSW1 未匹配到 2022 暂停。
Page-Hinkley 在 SA1 为 0/5，在 NSW1 为 1/5（太阳能／温和天气事件）。
其 Tier 2 匹配仍计入 precision。

English: KSWIN matched all five Tier 1 references in both regions. ADWIN
matched four in each: Five-Minute Settlement was not detected in SA1 and
the 2022 suspension was not detected in NSW1. Page-Hinkley matched none
of the five in SA1 and one in NSW1 (solar/mild-weather erosion). Its Tier 2
matches still contribute to precision.

中文：这些是历史背景事件，不是 ground-truth changepoints。Precision 应解释为
“目录匹配比例”，不代表真实分类准确性。宽时间窗口和重复检测会影响匹配比例；
不能只凭这张表判断检测器优劣，也不能把 unmatched 直接称为 false positives。
日期精度有限，小数延迟仅表示相对目录起日的时间差，不能理解为精确的物理响应时间。
这里只评估已保存的 Task 120 运行；KSWIN 使用其记录的 seed 12，没有虚构多 seed
实验或 mean ± SD。

English: These are contextual historical events, not ground-truth
changepoints. Precision is a catalogue-match proportion, not ground-truth
classification accuracy. Broad windows and repeated detections affect the
proportion, so this table alone cannot establish detector superiority.
Unmatched detections are not automatically false positives. Fractional
delays measure offsets from catalogue dates rather than exact physical
response times. This evaluates the stored Task 120 runs, including KSWIN
seed 12; no additional seeds or mean ± SD results are invented.

## 7. 验证与复现 / Verify and reproduce

中文：11 项新测试和原有 synthetic 检测检查通过。全部 1,060 个输入检测均在
匹配明细中出现且只出现一次；105 + 120 + 835 的计数平衡。事件目录和 Task 120
输入未改动。重复运行后日志字节完全一致；`--from-logs` 重建的 T2 CSV 与原输出
字节完全一致。Manifest 保存输入、代码和规则的 SHA-256 及实际运行配置。

English: Eleven new tests and the existing synthetic checks passed. Every
input detection appears exactly once in the audit, with counts balancing
as 105 + 120 + 835 = 1,060. The catalogue and Task 120 inputs are unchanged.
A repeated evaluation leaves the log byte-identical, and `--from-logs`
reconstructs a byte-identical T2 CSV. The manifest records SHA-256 hashes
for inputs, code and protocol, together with the actual run configurations.

From the repository root / 在仓库根目录运行：

```sh
python3 experiments/detection/run_aemo_t2_evaluation.py
python3 experiments/detection/run_aemo_t2_evaluation.py --from-logs
PYTHONPATH=src python3 -m unittest discover -s test/detection -p test_aemo_t2_evaluation.py -v
```

## Files / 文件

- [Final T2 table / 最终表格](../results/tables/detection/T2_aemo_detection.md)
- [T2 CSV](../results/tables/detection/T2_aemo_detection.csv)
- [Per-detection audit / 逐检测匹配明细](../results/tables/detection/T2_detection_matches.csv)
- [Tier 1 details / 五事件延迟明细](../results/tables/detection/T2_tier1_event_details.csv)
- [Counts / 匹配计数](../results/tables/detection/T2_detection_summary.csv)
- [Manifest / 来源与配置](../results/tables/detection/T2_manifest.json)
- [Run log / 长格式指标日志](../results/runs.csv)
- [Matching protocol / 匹配规则](aemo_matching_protocol.md)

The historical `results/tables/runs.csv` was preserved. T2 writes to
`results/runs.csv`, the path specified in this deliverable.

历史 `results/tables/runs.csv` 已保留；T2 写入本交付要求指定的 `results/runs.csv`。
