# Failure-detection validation report

Computed against 500 SWE-agent traces (250 failures / 250 successes by `target`).

Each row is the confusion matrix produced when the named signal (or `any_signal` for the aggregate) is treated as the positive prediction and `has_error = 1` as the positive label.

| Signal | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hit_max_turns` | 99 | 27 | 151 | 223 | 0.79 | 0.40 | 0.53 |
| `repeated_tool_call_loop` | 42 | 8 | 208 | 242 | 0.84 | 0.17 | 0.28 |
| `ended_on_error` | 85 | 15 | 165 | 235 | 0.85 | 0.34 | 0.49 |
| `abandonment` | 1 | 0 | 249 | 250 | 1.00 | 0.00 | 0.01 |
| `incomplete_session` | 0 | 0 | 250 | 250 | n/a | 0.00 | n/a |
| `any_signal` | 121 | 34 | 129 | 216 | 0.78 | 0.48 | 0.60 |

**Reading the table.** High precision + low recall means the rule is correct when it fires but misses many failures (the LLM judge picks those up in the second pass). High recall + low precision means the rule over-triggers and we'd need to tighten it. The aggregate `any_signal` row is what gates whether a trace is sent to the LLM judge: anything firing here gets a label.
