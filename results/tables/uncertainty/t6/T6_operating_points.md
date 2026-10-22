# T6 — Escalation operating points

Percentages are seed means ± sample SD in percentage points. Partial reachability is explicit. Stored thresholds only; no interpolation. Budgets are hard caps. Coverage is on retained forecasts, not human-corrected system coverage. All is an auxiliary whole-stream row.

## NSW1 / A / adaptive_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | 1.667% ± 3.727 pp      | 95.105% ± 0.181 pp | 95.301% ± 0.126 pp | 95.356% ± 0.199 pp |
| drift      | partial: 4/5 seeds achieved; 1/5 unreachable | unreachable            | 90.049% ± 0.036 pp | 90.049% ± 0.036 pp | 90.049% ± 0.036 pp |
| post-drift | unreachable                                  | unreachable            | 89.940% ± 0.013 pp | 89.940% ± 0.013 pp | 89.940% ± 0.013 pp |
| all        | partial: 2/5 seeds achieved; 3/5 unreachable | unreachable            | 90.000% ± 0.001 pp | 90.000% ± 0.001 pp | 90.000% ± 0.001 pp |

## NSW1 / A / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp |
| drift      | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.689% ± 0.178 pp | 96.689% ± 0.178 pp | 96.689% ± 0.178 pp |
| post-drift | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.989% ± 0.196 pp | 96.989% ± 0.196 pp | 96.989% ± 0.196 pp |
| all        | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.899% ± 0.192 pp | 96.899% ± 0.192 pp | 96.899% ± 0.192 pp |

## NSW1 / B / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 1.667% ± 3.727 pp      | 95.105% ± 0.181 pp | 95.301% ± 0.126 pp | 95.356% ± 0.199 pp |
| drift      | unreachable            | unreachable            | 89.914% ± 0.026 pp | 89.914% ± 0.026 pp | 89.914% ± 0.026 pp |
| post-drift | 0.000% ± 0.000 pp      | unreachable            | 90.010% ± 0.011 pp | 90.010% ± 0.011 pp | 90.010% ± 0.011 pp |
| all        | 0.000% ± 0.000 pp      | unreachable            | 90.011% ± 0.002 pp | 90.011% ± 0.002 pp | 90.011% ± 0.002 pp |

## NSW1 / B / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp |
| drift      | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 97.830% ± 0.132 pp | 97.830% ± 0.132 pp | 97.830% ± 0.132 pp |
| post-drift | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 97.668% ± 0.145 pp | 97.668% ± 0.145 pp | 97.668% ± 0.145 pp |
| all        | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 97.707% ± 0.143 pp | 97.707% ± 0.143 pp | 97.707% ± 0.143 pp |

## NSW1 / C_adwin / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 1.667% ± 3.727 pp      | 95.105% ± 0.181 pp | 95.301% ± 0.126 pp | 95.356% ± 0.199 pp |
| drift      | unreachable            | unreachable            | 89.877% ± 0.045 pp | 89.877% ± 0.045 pp | 89.877% ± 0.045 pp |
| post-drift | 0.000% ± 0.000 pp      | unreachable            | 90.025% ± 0.017 pp | 90.025% ± 0.017 pp | 90.025% ± 0.017 pp |
| all        | 0.000% ± 0.000 pp      | unreachable            | 90.012% ± 0.001 pp | 90.012% ± 0.001 pp | 90.012% ± 0.001 pp |

## NSW1 / C_adwin / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp |
| drift      | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 97.854% ± 0.114 pp | 97.854% ± 0.114 pp | 97.854% ± 0.114 pp |
| post-drift | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 97.635% ± 0.171 pp | 97.635% ± 0.171 pp | 97.635% ± 0.171 pp |
| all        | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 97.690% ± 0.152 pp | 97.690% ± 0.152 pp | 97.690% ± 0.152 pp |

## NSW1 / C_kswin / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 1.667% ± 3.727 pp      | 95.105% ± 0.181 pp | 95.301% ± 0.126 pp | 95.356% ± 0.199 pp |
| drift      | unreachable            | unreachable            | 89.879% ± 0.048 pp | 89.879% ± 0.048 pp | 89.879% ± 0.048 pp |
| post-drift | 0.000% ± 0.000 pp      | unreachable            | 90.029% ± 0.020 pp | 90.029% ± 0.020 pp | 90.029% ± 0.020 pp |
| all        | 0.000% ± 0.000 pp      | unreachable            | 90.014% ± 0.002 pp | 90.014% ± 0.002 pp | 90.014% ± 0.002 pp |

## NSW1 / C_kswin / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp |
| drift      | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 97.708% ± 0.158 pp | 97.708% ± 0.158 pp | 97.708% ± 0.158 pp |
| post-drift | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 97.622% ± 0.151 pp | 97.622% ± 0.151 pp | 97.622% ± 0.151 pp |
| all        | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 97.639% ± 0.153 pp | 97.639% ± 0.153 pp | 97.639% ± 0.153 pp |

## NSW1 / C_page_hinkley / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 1.667% ± 3.727 pp      | 95.105% ± 0.181 pp | 95.301% ± 0.126 pp | 95.356% ± 0.199 pp |
| drift      | unreachable            | unreachable            | 89.922% ± 0.029 pp | 89.922% ± 0.029 pp | 89.922% ± 0.029 pp |
| post-drift | 0.000% ± 0.000 pp      | unreachable            | 90.008% ± 0.010 pp | 90.008% ± 0.010 pp | 90.008% ± 0.010 pp |
| all        | 0.000% ± 0.000 pp      | unreachable            | 90.012% ± 0.001 pp | 90.012% ± 0.001 pp | 90.012% ± 0.001 pp |

## NSW1 / C_page_hinkley / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp |
| drift      | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 97.719% ± 0.135 pp | 97.719% ± 0.135 pp | 97.719% ± 0.135 pp |
| post-drift | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 97.604% ± 0.223 pp | 97.604% ± 0.223 pp | 97.604% ± 0.223 pp |
| all        | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 97.630% ± 0.198 pp | 97.630% ± 0.198 pp | 97.630% ± 0.198 pp |

## NSW1 / D_120d_adwin / adaptive_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | 1.667% ± 3.727 pp      | 95.105% ± 0.181 pp | 95.301% ± 0.126 pp | 95.356% ± 0.199 pp |
| drift      | 71.267% ± 1.100 pp                           | unreachable            | 89.721% ± 0.021 pp | 89.721% ± 0.021 pp | 89.721% ± 0.021 pp |
| post-drift | 0.000% ± 0.000 pp                            | unreachable            | 90.070% ± 0.009 pp | 90.070% ± 0.009 pp | 90.070% ± 0.009 pp |
| all        | partial: 2/5 seeds achieved; 3/5 unreachable | unreachable            | 89.998% ± 0.009 pp | 89.998% ± 0.009 pp | 89.998% ± 0.009 pp |

## NSW1 / D_120d_adwin / static_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | 0.000% ± 0.000 pp      | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp |
| drift      | unreachable                                  | unreachable            | 86.303% ± 0.492 pp | 86.303% ± 0.492 pp | 86.303% ± 0.492 pp |
| post-drift | partial: 3/5 seeds achieved; 2/5 unreachable | unreachable            | 90.192% ± 0.406 pp | 90.192% ± 0.406 pp | 90.192% ± 0.406 pp |
| all        | unreachable                                  | unreachable            | 89.108% ± 0.431 pp | 89.108% ± 0.431 pp | 89.108% ± 0.431 pp |

## NSW1 / D_120d_kswin / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 1.667% ± 3.727 pp      | 95.105% ± 0.181 pp | 95.301% ± 0.126 pp | 95.356% ± 0.199 pp |
| drift      | unreachable            | unreachable            | 89.510% ± 0.014 pp | 89.510% ± 0.014 pp | 89.510% ± 0.014 pp |
| post-drift | 0.000% ± 0.000 pp      | unreachable            | 90.122% ± 0.008 pp | 90.122% ± 0.008 pp | 90.122% ± 0.008 pp |
| all        | unreachable            | unreachable            | 89.974% ± 0.006 pp | 89.974% ± 0.006 pp | 89.974% ± 0.006 pp |

## NSW1 / D_120d_kswin / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp |
| drift      | unreachable            | unreachable            | 88.483% ± 0.456 pp | 88.483% ± 0.456 pp | 88.483% ± 0.456 pp |
| post-drift | 0.000% ± 0.000 pp      | unreachable            | 92.881% ± 0.380 pp | 92.881% ± 0.380 pp | 92.881% ± 0.380 pp |
| all        | 0.000% ± 0.000 pp      | unreachable            | 91.636% ± 0.402 pp | 91.636% ± 0.402 pp | 91.636% ± 0.402 pp |

## NSW1 / D_120d_page_hinkley / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 1.667% ± 3.727 pp      | 95.105% ± 0.181 pp | 95.301% ± 0.126 pp | 95.356% ± 0.199 pp |
| drift      | 0.000% ± 0.000 pp      | unreachable            | 90.115% ± 0.019 pp | 90.121% ± 0.016 pp | 90.124% ± 0.018 pp |
| post-drift | unreachable            | unreachable            | 89.904% ± 0.003 pp | 89.904% ± 0.003 pp | 89.904% ± 0.003 pp |
| all        | unreachable            | unreachable            | 89.994% ± 0.003 pp | 89.994% ± 0.003 pp | 89.994% ± 0.003 pp |

## NSW1 / D_120d_page_hinkley / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp |
| drift      | 0.000% ± 0.000 pp      | unreachable            | 92.354% ± 0.393 pp | 92.354% ± 0.393 pp | 92.354% ± 0.393 pp |
| post-drift | 0.000% ± 0.000 pp      | unreachable            | 94.123% ± 0.304 pp | 94.123% ± 0.304 pp | 94.123% ± 0.304 pp |
| all        | 0.000% ± 0.000 pp      | unreachable            | 93.627% ± 0.330 pp | 93.627% ± 0.330 pp | 93.627% ± 0.330 pp |

## NSW1 / D_180d_adwin / adaptive_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | 1.667% ± 3.727 pp      | 95.105% ± 0.181 pp | 95.301% ± 0.126 pp | 95.356% ± 0.199 pp |
| drift      | 42.591% ± 1.781 pp                           | unreachable            | 89.836% ± 0.022 pp | 89.836% ± 0.022 pp | 89.836% ± 0.022 pp |
| post-drift | 0.000% ± 0.000 pp                            | unreachable            | 90.026% ± 0.004 pp | 90.026% ± 0.004 pp | 90.026% ± 0.004 pp |
| all        | partial: 3/5 seeds achieved; 2/5 unreachable | unreachable            | 90.000% ± 0.004 pp | 90.000% ± 0.004 pp | 90.000% ± 0.004 pp |

## NSW1 / D_180d_adwin / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp |
| drift      | unreachable            | unreachable            | 87.540% ± 0.555 pp | 87.540% ± 0.555 pp | 87.540% ± 0.555 pp |
| post-drift | 0.000% ± 0.000 pp      | unreachable            | 91.663% ± 0.361 pp | 91.663% ± 0.361 pp | 91.663% ± 0.361 pp |
| all        | 0.000% ± 0.000 pp      | unreachable            | 90.504% ± 0.417 pp | 90.504% ± 0.417 pp | 90.504% ± 0.417 pp |

## NSW1 / D_180d_kswin / adaptive_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | 1.667% ± 3.727 pp      | 95.105% ± 0.181 pp | 95.301% ± 0.126 pp | 95.356% ± 0.199 pp |
| drift      | unreachable                                  | unreachable            | 89.849% ± 0.025 pp | 89.849% ± 0.025 pp | 89.850% ± 0.024 pp |
| post-drift | partial: 3/5 seeds achieved; 2/5 unreachable | unreachable            | 89.997% ± 0.007 pp | 89.997% ± 0.007 pp | 89.997% ± 0.007 pp |
| all        | unreachable                                  | unreachable            | 89.983% ± 0.004 pp | 89.983% ± 0.004 pp | 89.983% ± 0.004 pp |

## NSW1 / D_180d_kswin / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp |
| drift      | 0.000% ± 0.000 pp      | unreachable            | 90.682% ± 0.462 pp | 90.682% ± 0.462 pp | 90.682% ± 0.462 pp |
| post-drift | 0.000% ± 0.000 pp      | unreachable            | 94.281% ± 0.325 pp | 94.281% ± 0.325 pp | 94.281% ± 0.325 pp |
| all        | 0.000% ± 0.000 pp      | unreachable            | 93.258% ± 0.365 pp | 93.258% ± 0.365 pp | 93.258% ± 0.365 pp |

## NSW1 / D_180d_page_hinkley / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 1.667% ± 3.727 pp      | 95.105% ± 0.181 pp | 95.301% ± 0.126 pp | 95.356% ± 0.199 pp |
| drift      | 0.000% ± 0.000 pp      | unreachable            | 90.094% ± 0.017 pp | 90.100% ± 0.020 pp | 90.103% ± 0.020 pp |
| post-drift | unreachable            | unreachable            | 89.913% ± 0.005 pp | 89.913% ± 0.005 pp | 89.913% ± 0.005 pp |
| all        | unreachable            | unreachable            | 89.994% ± 0.003 pp | 89.994% ± 0.003 pp | 89.994% ± 0.003 pp |

## NSW1 / D_180d_page_hinkley / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95                         | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:---------------------------------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp                            | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp |
| drift      | 0.000% ± 0.000 pp      | unreachable                                  | 94.472% ± 0.251 pp | 94.472% ± 0.251 pp | 94.472% ± 0.251 pp |
| post-drift | 0.000% ± 0.000 pp      | partial: 2/5 seeds achieved; 3/5 unreachable | 94.935% ± 0.244 pp | 94.935% ± 0.244 pp | 94.935% ± 0.244 pp |
| all        | 0.000% ± 0.000 pp      | partial: 1/5 seeds achieved; 4/5 unreachable | 94.810% ± 0.247 pp | 94.810% ± 0.247 pp | 94.810% ± 0.247 pp |

## NSW1 / D_30d_adwin / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 1.667% ± 3.727 pp      | 95.105% ± 0.181 pp | 95.301% ± 0.126 pp | 95.356% ± 0.199 pp |
| drift      | 46.401% ± 1.608 pp     | unreachable            | 89.810% ± 0.020 pp | 89.810% ± 0.020 pp | 89.810% ± 0.020 pp |
| post-drift | 0.116% ± 0.017 pp      | unreachable            | 90.020% ± 0.008 pp | 90.020% ± 0.008 pp | 90.020% ± 0.008 pp |
| all        | unreachable            | unreachable            | 89.988% ± 0.010 pp | 89.988% ± 0.010 pp | 89.988% ± 0.010 pp |

## NSW1 / D_30d_adwin / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp |
| drift      | unreachable            | unreachable            | 83.904% ± 0.461 pp | 83.904% ± 0.461 pp | 83.904% ± 0.461 pp |
| post-drift | unreachable            | unreachable            | 89.072% ± 0.409 pp | 89.072% ± 0.409 pp | 89.072% ± 0.409 pp |
| all        | unreachable            | unreachable            | 87.627% ± 0.424 pp | 87.627% ± 0.424 pp | 87.627% ± 0.424 pp |

## NSW1 / D_30d_kswin / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 1.667% ± 3.727 pp      | 95.105% ± 0.181 pp | 95.301% ± 0.126 pp | 95.356% ± 0.199 pp |
| drift      | unreachable            | unreachable            | 89.326% ± 0.014 pp | 89.326% ± 0.014 pp | 89.326% ± 0.014 pp |
| post-drift | 0.000% ± 0.000 pp      | unreachable            | 90.164% ± 0.006 pp | 90.164% ± 0.006 pp | 90.164% ± 0.006 pp |
| all        | unreachable            | unreachable            | 89.951% ± 0.007 pp | 89.951% ± 0.007 pp | 89.951% ± 0.007 pp |

## NSW1 / D_30d_kswin / static_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | 0.000% ± 0.000 pp      | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp |
| drift      | unreachable                                  | unreachable            | 85.477% ± 0.476 pp | 85.477% ± 0.476 pp | 85.477% ± 0.476 pp |
| post-drift | 0.000% ± 0.000 pp                            | unreachable            | 92.017% ± 0.400 pp | 92.017% ± 0.400 pp | 92.017% ± 0.400 pp |
| all        | partial: 3/5 seeds achieved; 2/5 unreachable | unreachable            | 90.160% ± 0.422 pp | 90.160% ± 0.422 pp | 90.160% ± 0.422 pp |

## NSW1 / D_30d_page_hinkley / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 1.667% ± 3.727 pp      | 95.105% ± 0.181 pp | 95.301% ± 0.126 pp | 95.356% ± 0.199 pp |
| drift      | 0.000% ± 0.000 pp      | unreachable            | 90.086% ± 0.020 pp | 90.086% ± 0.020 pp | 90.089% ± 0.018 pp |
| post-drift | unreachable            | unreachable            | 89.883% ± 0.005 pp | 89.883% ± 0.005 pp | 89.883% ± 0.005 pp |
| all        | unreachable            | unreachable            | 89.970% ± 0.004 pp | 89.970% ± 0.004 pp | 89.970% ± 0.004 pp |

## NSW1 / D_30d_page_hinkley / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp |
| drift      | 0.000% ± 0.000 pp      | unreachable            | 91.748% ± 0.346 pp | 91.748% ± 0.346 pp | 91.748% ± 0.346 pp |
| post-drift | 0.000% ± 0.000 pp      | unreachable            | 90.782% ± 0.430 pp | 90.782% ± 0.430 pp | 90.782% ± 0.430 pp |
| all        | 0.000% ± 0.000 pp      | unreachable            | 91.092% ± 0.406 pp | 91.092% ± 0.406 pp | 91.092% ± 0.406 pp |

## NSW1 / D_60d_adwin / adaptive_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | 1.667% ± 3.727 pp      | 95.105% ± 0.181 pp | 95.301% ± 0.126 pp | 95.356% ± 0.199 pp |
| drift      | 60.272% ± 9.241 pp                           | unreachable            | 89.676% ± 0.021 pp | 89.676% ± 0.021 pp | 89.676% ± 0.021 pp |
| post-drift | 0.000% ± 0.000 pp                            | unreachable            | 90.085% ± 0.005 pp | 90.085% ± 0.005 pp | 90.085% ± 0.005 pp |
| all        | partial: 1/5 seeds achieved; 4/5 unreachable | unreachable            | 89.994% ± 0.006 pp | 89.994% ± 0.006 pp | 89.994% ± 0.006 pp |

## NSW1 / D_60d_adwin / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp |
| drift      | unreachable            | unreachable            | 84.654% ± 0.518 pp | 84.654% ± 0.518 pp | 84.654% ± 0.518 pp |
| post-drift | unreachable            | unreachable            | 88.998% ± 0.426 pp | 88.998% ± 0.426 pp | 88.998% ± 0.426 pp |
| all        | unreachable            | unreachable            | 87.790% ± 0.453 pp | 87.790% ± 0.453 pp | 87.790% ± 0.453 pp |

## NSW1 / D_60d_kswin / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 1.667% ± 3.727 pp      | 95.105% ± 0.181 pp | 95.301% ± 0.126 pp | 95.356% ± 0.199 pp |
| drift      | unreachable            | unreachable            | 89.440% ± 0.030 pp | 89.440% ± 0.030 pp | 89.440% ± 0.030 pp |
| post-drift | 0.000% ± 0.000 pp      | unreachable            | 90.136% ± 0.005 pp | 90.136% ± 0.005 pp | 90.136% ± 0.005 pp |
| all        | unreachable            | unreachable            | 89.963% ± 0.006 pp | 89.963% ± 0.006 pp | 89.963% ± 0.006 pp |

## NSW1 / D_60d_kswin / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp |
| drift      | unreachable            | unreachable            | 87.533% ± 0.480 pp | 87.533% ± 0.480 pp | 87.533% ± 0.480 pp |
| post-drift | 0.000% ± 0.000 pp      | unreachable            | 92.488% ± 0.412 pp | 92.488% ± 0.412 pp | 92.488% ± 0.412 pp |
| all        | 0.000% ± 0.000 pp      | unreachable            | 91.085% ± 0.431 pp | 91.085% ± 0.431 pp | 91.085% ± 0.431 pp |

## NSW1 / D_60d_page_hinkley / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 1.667% ± 3.727 pp      | 95.105% ± 0.181 pp | 95.301% ± 0.126 pp | 95.356% ± 0.199 pp |
| drift      | 0.000% ± 0.000 pp      | unreachable            | 90.250% ± 0.025 pp | 90.278% ± 0.024 pp | 90.300% ± 0.031 pp |
| post-drift | unreachable            | unreachable            | 89.815% ± 0.002 pp | 89.815% ± 0.002 pp | 89.815% ± 0.002 pp |
| all        | unreachable            | unreachable            | 89.970% ± 0.006 pp | 89.970% ± 0.006 pp | 89.970% ± 0.006 pp |

## NSW1 / D_60d_page_hinkley / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp | 96.354% ± 0.521 pp |
| drift      | 0.000% ± 0.000 pp      | unreachable            | 91.802% ± 0.375 pp | 91.802% ± 0.375 pp | 91.802% ± 0.375 pp |
| post-drift | 0.000% ± 0.000 pp      | unreachable            | 90.586% ± 0.365 pp | 90.586% ± 0.365 pp | 90.586% ± 0.365 pp |
| all        | 0.000% ± 0.000 pp      | unreachable            | 90.969% ± 0.368 pp | 90.969% ± 0.368 pp | 90.969% ± 0.368 pp |

## SA1 / A / adaptive_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | unreachable            | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp |
| drift      | partial: 2/5 seeds achieved; 3/5 unreachable | unreachable            | 89.980% ± 0.038 pp | 89.980% ± 0.038 pp | 89.980% ± 0.038 pp |
| post-drift | partial: 1/5 seeds achieved; 4/5 unreachable | unreachable            | 89.986% ± 0.019 pp | 89.986% ± 0.019 pp | 89.986% ± 0.019 pp |
| all        | unreachable                                  | unreachable            | 89.992% ± 0.001 pp | 89.992% ± 0.001 pp | 89.992% ± 0.001 pp |

## SA1 / A / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp |
| drift      | 0.000% ± 0.000 pp      | unreachable            | 93.202% ± 0.519 pp | 93.202% ± 0.519 pp | 93.202% ± 0.519 pp |
| post-drift | 0.000% ± 0.000 pp      | unreachable            | 94.462% ± 0.330 pp | 94.462% ± 0.330 pp | 94.462% ± 0.330 pp |
| all        | 0.000% ± 0.000 pp      | unreachable            | 94.043% ± 0.392 pp | 94.043% ± 0.392 pp | 94.043% ± 0.392 pp |

## SA1 / B / adaptive_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | unreachable            | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp |
| drift      | unreachable                                  | unreachable            | 89.909% ± 0.027 pp | 89.909% ± 0.027 pp | 89.909% ± 0.027 pp |
| post-drift | 0.007% ± 0.015 pp                            | unreachable            | 90.035% ± 0.015 pp | 90.035% ± 0.015 pp | 90.035% ± 0.015 pp |
| all        | partial: 1/5 seeds achieved; 4/5 unreachable | unreachable            | 89.999% ± 0.001 pp | 89.999% ± 0.001 pp | 89.999% ± 0.001 pp |

## SA1 / B / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95                         | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:---------------------------------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp                            | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp |
| drift      | 0.000% ± 0.000 pp      | unreachable                                  | 94.548% ± 0.424 pp | 94.548% ± 0.424 pp | 94.548% ± 0.424 pp |
| post-drift | 0.000% ± 0.000 pp      | partial: 3/5 seeds achieved; 2/5 unreachable | 95.236% ± 0.381 pp | 95.236% ± 0.381 pp | 95.236% ± 0.381 pp |
| all        | 0.000% ± 0.000 pp      | partial: 3/5 seeds achieved; 2/5 unreachable | 95.007% ± 0.397 pp | 95.007% ± 0.397 pp | 95.007% ± 0.397 pp |

## SA1 / C_adwin / adaptive_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | unreachable            | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp |
| drift      | unreachable                                  | unreachable            | 89.946% ± 0.020 pp | 89.946% ± 0.020 pp | 89.946% ± 0.020 pp |
| post-drift | 0.010% ± 0.014 pp                            | unreachable            | 90.018% ± 0.006 pp | 90.018% ± 0.006 pp | 90.018% ± 0.006 pp |
| all        | partial: 3/5 seeds achieved; 2/5 unreachable | unreachable            | 90.001% ± 0.002 pp | 90.001% ± 0.002 pp | 90.001% ± 0.002 pp |

## SA1 / C_adwin / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95                         | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:---------------------------------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp                            | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp |
| drift      | 0.000% ± 0.000 pp      | unreachable                                  | 94.407% ± 0.435 pp | 94.407% ± 0.435 pp | 94.407% ± 0.435 pp |
| post-drift | 0.000% ± 0.000 pp      | partial: 4/5 seeds achieved; 1/5 unreachable | 95.327% ± 0.344 pp | 95.327% ± 0.344 pp | 95.327% ± 0.344 pp |
| all        | 0.000% ± 0.000 pp      | partial: 3/5 seeds achieved; 2/5 unreachable | 95.019% ± 0.377 pp | 95.019% ± 0.377 pp | 95.019% ± 0.377 pp |

## SA1 / C_kswin / adaptive_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | unreachable            | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp |
| drift      | unreachable                                  | unreachable            | 89.945% ± 0.038 pp | 89.945% ± 0.038 pp | 89.945% ± 0.038 pp |
| post-drift | partial: 3/5 seeds achieved; 2/5 unreachable | unreachable            | 90.013% ± 0.023 pp | 90.013% ± 0.023 pp | 90.013% ± 0.023 pp |
| all        | unreachable                                  | unreachable            | 89.997% ± 0.001 pp | 89.997% ± 0.001 pp | 89.997% ± 0.001 pp |

## SA1 / C_kswin / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95                         | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:---------------------------------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp                            | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp |
| drift      | 0.000% ± 0.000 pp      | unreachable                                  | 94.417% ± 0.425 pp | 94.417% ± 0.425 pp | 94.417% ± 0.425 pp |
| post-drift | 0.000% ± 0.000 pp      | partial: 4/5 seeds achieved; 1/5 unreachable | 95.311% ± 0.341 pp | 95.311% ± 0.341 pp | 95.311% ± 0.341 pp |
| all        | 0.000% ± 0.000 pp      | partial: 3/5 seeds achieved; 2/5 unreachable | 95.012% ± 0.369 pp | 95.012% ± 0.369 pp | 95.012% ± 0.369 pp |

## SA1 / C_page_hinkley / adaptive_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | unreachable            | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp |
| drift      | unreachable                                  | unreachable            | 89.931% ± 0.041 pp | 89.931% ± 0.041 pp | 89.931% ± 0.041 pp |
| post-drift | 0.000% ± 0.000 pp                            | unreachable            | 90.029% ± 0.017 pp | 90.029% ± 0.017 pp | 90.029% ± 0.017 pp |
| all        | partial: 4/5 seeds achieved; 1/5 unreachable | unreachable            | 90.003% ± 0.003 pp | 90.003% ± 0.003 pp | 90.003% ± 0.003 pp |

## SA1 / C_page_hinkley / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95                         | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:---------------------------------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp                            | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp |
| drift      | 0.000% ± 0.000 pp      | unreachable                                  | 94.335% ± 0.445 pp | 94.335% ± 0.445 pp | 94.335% ± 0.445 pp |
| post-drift | 0.000% ± 0.000 pp      | partial: 4/5 seeds achieved; 1/5 unreachable | 95.318% ± 0.313 pp | 95.318% ± 0.313 pp | 95.318% ± 0.313 pp |
| all        | 0.000% ± 0.000 pp      | partial: 3/5 seeds achieved; 2/5 unreachable | 94.988% ± 0.360 pp | 94.988% ± 0.360 pp | 94.988% ± 0.360 pp |

## SA1 / D_120d_adwin / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | unreachable            | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp |
| drift      | 0.275% ± 0.030 pp      | unreachable            | 90.102% ± 0.016 pp | 90.102% ± 0.016 pp | 90.102% ± 0.016 pp |
| post-drift | unreachable            | unreachable            | 89.914% ± 0.009 pp | 89.914% ± 0.009 pp | 89.914% ± 0.009 pp |
| all        | unreachable            | unreachable            | 89.986% ± 0.004 pp | 89.986% ± 0.004 pp | 89.986% ± 0.004 pp |

## SA1 / D_120d_adwin / static_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | 0.000% ± 0.000 pp      | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp |
| drift      | unreachable                                  | unreachable            | 87.209% ± 0.601 pp | 87.209% ± 0.601 pp | 87.209% ± 0.601 pp |
| post-drift | partial: 3/5 seeds achieved; 2/5 unreachable | unreachable            | 90.046% ± 0.445 pp | 90.046% ± 0.445 pp | 90.046% ± 0.445 pp |
| all        | unreachable                                  | unreachable            | 89.114% ± 0.501 pp | 89.114% ± 0.501 pp | 89.114% ± 0.501 pp |

## SA1 / D_120d_kswin / adaptive_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | unreachable            | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp |
| drift      | partial: 4/5 seeds achieved; 1/5 unreachable | unreachable            | 90.014% ± 0.018 pp | 90.014% ± 0.018 pp | 90.014% ± 0.018 pp |
| post-drift | unreachable                                  | unreachable            | 89.960% ± 0.011 pp | 89.960% ± 0.011 pp | 89.960% ± 0.011 pp |
| all        | unreachable                                  | unreachable            | 89.985% ± 0.004 pp | 89.985% ± 0.004 pp | 89.985% ± 0.004 pp |

## SA1 / D_120d_kswin / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp |
| drift      | unreachable            | unreachable            | 86.497% ± 0.644 pp | 86.497% ± 0.644 pp | 86.497% ± 0.644 pp |
| post-drift | unreachable            | unreachable            | 88.909% ± 0.513 pp | 88.909% ± 0.513 pp | 88.909% ± 0.513 pp |
| all        | unreachable            | unreachable            | 88.129% ± 0.559 pp | 88.129% ± 0.559 pp | 88.129% ± 0.559 pp |

## SA1 / D_120d_page_hinkley / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | unreachable            | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp |
| drift      | unreachable            | unreachable            | 89.628% ± 0.015 pp | 89.628% ± 0.015 pp | 89.628% ± 0.015 pp |
| post-drift | 0.000% ± 0.000 pp      | unreachable            | 90.147% ± 0.003 pp | 90.147% ± 0.003 pp | 90.147% ± 0.003 pp |
| all        | unreachable            | unreachable            | 89.977% ± 0.001 pp | 89.977% ± 0.001 pp | 89.977% ± 0.001 pp |

## SA1 / D_120d_page_hinkley / static_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | 0.000% ± 0.000 pp      | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp |
| drift      | partial: 3/5 seeds achieved; 2/5 unreachable | unreachable            | 90.087% ± 0.596 pp | 90.087% ± 0.596 pp | 90.087% ± 0.596 pp |
| post-drift | 0.000% ± 0.000 pp                            | unreachable            | 91.676% ± 0.510 pp | 91.676% ± 0.510 pp | 91.676% ± 0.510 pp |
| all        | 0.000% ± 0.000 pp                            | unreachable            | 91.160% ± 0.541 pp | 91.160% ± 0.541 pp | 91.160% ± 0.541 pp |

## SA1 / D_180d_adwin / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | unreachable            | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp |
| drift      | 66.759% ± 0.923 pp     | unreachable            | 89.875% ± 0.015 pp | 89.875% ± 0.015 pp | 89.875% ± 0.015 pp |
| post-drift | 0.066% ± 0.061 pp      | unreachable            | 90.036% ± 0.007 pp | 90.036% ± 0.007 pp | 90.036% ± 0.007 pp |
| all        | unreachable            | unreachable            | 89.989% ± 0.002 pp | 89.989% ± 0.002 pp | 89.989% ± 0.002 pp |

## SA1 / D_180d_adwin / static_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | 0.000% ± 0.000 pp      | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp |
| drift      | unreachable                                  | unreachable            | 85.996% ± 0.646 pp | 85.996% ± 0.646 pp | 85.996% ± 0.646 pp |
| post-drift | partial: 2/5 seeds achieved; 3/5 unreachable | unreachable            | 89.643% ± 0.536 pp | 89.643% ± 0.536 pp | 89.643% ± 0.536 pp |
| all        | unreachable                                  | unreachable            | 88.437% ± 0.575 pp | 88.437% ± 0.575 pp | 88.437% ± 0.575 pp |

## SA1 / D_180d_kswin / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | unreachable            | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp |
| drift      | unreachable            | unreachable            | 89.728% ± 0.011 pp | 89.728% ± 0.011 pp | 89.728% ± 0.011 pp |
| post-drift | 0.000% ± 0.000 pp      | unreachable            | 90.121% ± 0.008 pp | 90.121% ± 0.008 pp | 90.121% ± 0.008 pp |
| all        | unreachable            | unreachable            | 89.994% ± 0.002 pp | 89.994% ± 0.002 pp | 89.994% ± 0.002 pp |

## SA1 / D_180d_kswin / static_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | 0.000% ± 0.000 pp      | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp |
| drift      | unreachable                                  | unreachable            | 88.573% ± 0.604 pp | 88.573% ± 0.604 pp | 88.573% ± 0.604 pp |
| post-drift | 0.000% ± 0.000 pp                            | unreachable            | 91.439% ± 0.437 pp | 91.439% ± 0.437 pp | 91.439% ± 0.437 pp |
| all        | partial: 4/5 seeds achieved; 1/5 unreachable | unreachable            | 90.489% ± 0.496 pp | 90.489% ± 0.496 pp | 90.489% ± 0.496 pp |

## SA1 / D_180d_page_hinkley / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | unreachable            | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp |
| drift      | unreachable            | unreachable            | 89.700% ± 0.016 pp | 89.700% ± 0.016 pp | 89.700% ± 0.016 pp |
| post-drift | 0.000% ± 0.000 pp      | unreachable            | 90.109% ± 0.006 pp | 90.109% ± 0.006 pp | 90.109% ± 0.006 pp |
| all        | unreachable            | unreachable            | 89.977% ± 0.001 pp | 89.977% ± 0.001 pp | 89.977% ± 0.001 pp |

## SA1 / D_180d_page_hinkley / static_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | 0.000% ± 0.000 pp      | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp |
| drift      | partial: 3/5 seeds achieved; 2/5 unreachable | unreachable            | 89.926% ± 0.474 pp | 89.926% ± 0.474 pp | 89.926% ± 0.474 pp |
| post-drift | partial: 3/5 seeds achieved; 2/5 unreachable | unreachable            | 90.184% ± 0.515 pp | 90.184% ± 0.515 pp | 90.184% ± 0.515 pp |
| all        | partial: 3/5 seeds achieved; 2/5 unreachable | unreachable            | 90.131% ± 0.502 pp | 90.131% ± 0.502 pp | 90.131% ± 0.502 pp |

## SA1 / D_30d_adwin / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | unreachable            | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp |
| drift      | 0.756% ± 0.084 pp      | unreachable            | 90.041% ± 0.018 pp | 90.041% ± 0.018 pp | 90.041% ± 0.018 pp |
| post-drift | unreachable            | unreachable            | 89.904% ± 0.008 pp | 89.904% ± 0.008 pp | 89.904% ± 0.008 pp |
| all        | unreachable            | unreachable            | 89.958% ± 0.006 pp | 89.958% ± 0.006 pp | 89.958% ± 0.006 pp |

## SA1 / D_30d_adwin / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp |
| drift      | unreachable            | unreachable            | 84.041% ± 0.671 pp | 84.041% ± 0.671 pp | 84.041% ± 0.671 pp |
| post-drift | unreachable            | unreachable            | 87.874% ± 0.555 pp | 87.874% ± 0.555 pp | 87.874% ± 0.555 pp |
| all        | unreachable            | unreachable            | 86.615% ± 0.596 pp | 86.615% ± 0.596 pp | 86.615% ± 0.596 pp |

## SA1 / D_30d_kswin / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | unreachable            | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp |
| drift      | 77.365% ± 4.462 pp     | unreachable            | 89.900% ± 0.013 pp | 89.900% ± 0.013 pp | 89.900% ± 0.013 pp |
| post-drift | unreachable            | unreachable            | 89.980% ± 0.004 pp | 89.980% ± 0.004 pp | 89.980% ± 0.004 pp |
| all        | unreachable            | unreachable            | 89.958% ± 0.003 pp | 89.958% ± 0.003 pp | 89.958% ± 0.003 pp |

## SA1 / D_30d_kswin / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp |
| drift      | unreachable            | unreachable            | 85.028% ± 0.754 pp | 85.028% ± 0.754 pp | 85.028% ± 0.754 pp |
| post-drift | unreachable            | unreachable            | 87.173% ± 0.656 pp | 87.173% ± 0.656 pp | 87.173% ± 0.656 pp |
| all        | unreachable            | unreachable            | 86.493% ± 0.690 pp | 86.493% ± 0.690 pp | 86.493% ± 0.690 pp |

## SA1 / D_30d_page_hinkley / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | unreachable            | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp |
| drift      | 0.054% ± 0.050 pp      | unreachable            | 90.052% ± 0.015 pp | 90.052% ± 0.015 pp | 90.052% ± 0.015 pp |
| post-drift | 99.477% ± 0.197 pp     | unreachable            | 89.932% ± 0.005 pp | 89.932% ± 0.005 pp | 89.932% ± 0.005 pp |
| all        | unreachable            | unreachable            | 89.981% ± 0.001 pp | 89.981% ± 0.001 pp | 89.981% ± 0.001 pp |

## SA1 / D_30d_page_hinkley / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp |
| drift      | unreachable            | unreachable            | 88.736% ± 0.564 pp | 88.736% ± 0.564 pp | 88.736% ± 0.564 pp |
| post-drift | unreachable            | unreachable            | 86.432% ± 0.721 pp | 86.432% ± 0.721 pp | 86.432% ± 0.721 pp |
| all        | unreachable            | unreachable            | 87.274% ± 0.668 pp | 87.274% ± 0.668 pp | 87.274% ± 0.668 pp |

## SA1 / D_60d_adwin / adaptive_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | unreachable            | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp |
| drift      | partial: 3/5 seeds achieved; 2/5 unreachable | unreachable            | 89.992% ± 0.019 pp | 89.992% ± 0.019 pp | 89.992% ± 0.019 pp |
| post-drift | unreachable                                  | unreachable            | 89.950% ± 0.007 pp | 89.950% ± 0.007 pp | 89.950% ± 0.007 pp |
| all        | unreachable                                  | unreachable            | 89.972% ± 0.003 pp | 89.972% ± 0.003 pp | 89.972% ± 0.003 pp |

## SA1 / D_60d_adwin / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp |
| drift      | unreachable            | unreachable            | 85.182% ± 0.661 pp | 85.182% ± 0.661 pp | 85.182% ± 0.661 pp |
| post-drift | unreachable            | unreachable            | 88.633% ± 0.604 pp | 88.633% ± 0.604 pp | 88.633% ± 0.604 pp |
| all        | unreachable            | unreachable            | 87.499% ± 0.625 pp | 87.499% ± 0.625 pp | 87.499% ± 0.625 pp |

## SA1 / D_60d_kswin / adaptive_conformal

| regime     | review rate for 0.90                         | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:---------------------------------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp                            | unreachable            | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp |
| drift      | unreachable                                  | unreachable            | 89.899% ± 0.012 pp | 89.899% ± 0.012 pp | 89.899% ± 0.012 pp |
| post-drift | partial: 4/5 seeds achieved; 1/5 unreachable | unreachable            | 90.006% ± 0.007 pp | 90.006% ± 0.007 pp | 90.006% ± 0.007 pp |
| all        | unreachable                                  | unreachable            | 89.976% ± 0.002 pp | 89.976% ± 0.002 pp | 89.976% ± 0.002 pp |

## SA1 / D_60d_kswin / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp |
| drift      | unreachable            | unreachable            | 84.981% ± 0.667 pp | 84.981% ± 0.667 pp | 84.981% ± 0.667 pp |
| post-drift | unreachable            | unreachable            | 86.581% ± 0.661 pp | 86.581% ± 0.661 pp | 86.581% ± 0.661 pp |
| all        | unreachable            | unreachable            | 86.090% ± 0.664 pp | 86.090% ± 0.664 pp | 86.090% ± 0.664 pp |

## SA1 / D_60d_page_hinkley / adaptive_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | unreachable            | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp | 91.458% ± 0.622 pp |
| drift      | 0.000% ± 0.000 pp      | unreachable            | 90.084% ± 0.019 pp | 90.084% ± 0.019 pp | 90.084% ± 0.019 pp |
| post-drift | 98.670% ± 0.683 pp     | unreachable            | 89.911% ± 0.007 pp | 89.911% ± 0.007 pp | 89.911% ± 0.007 pp |
| all        | unreachable            | unreachable            | 89.979% ± 0.001 pp | 89.979% ± 0.001 pp | 89.979% ± 0.001 pp |

## SA1 / D_60d_page_hinkley / static_conformal

| regime     | review rate for 0.90   | review rate for 0.95   | coverage @ 5%      | coverage @ 10%     | coverage @ 20%     |
|:-----------|:-----------------------|:-----------------------|:-------------------|:-------------------|:-------------------|
| pre-drift  | 0.000% ± 0.000 pp      | 0.000% ± 0.000 pp      | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp | 96.302% ± 1.213 pp |
| drift      | unreachable            | unreachable            | 88.960% ± 0.599 pp | 88.960% ± 0.599 pp | 88.960% ± 0.599 pp |
| post-drift | unreachable            | unreachable            | 88.245% ± 0.655 pp | 88.245% ± 0.655 pp | 88.245% ± 0.655 pp |
| all        | unreachable            | unreachable            | 88.535% ± 0.637 pp | 88.535% ± 0.637 pp | 88.535% ± 0.637 pp |
