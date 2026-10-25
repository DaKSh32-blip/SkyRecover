# Delay Forecaster Results

Trained on 109,289 Delta flights (2024-01-01 to 2024-02-15), tested on 82,427 flights it had never seen (2024-03-01 to 2024-03-31). 16% of test flights left 15+ minutes late.

| Model | Avg error (min) | AUC | Precision | Recall |
|---|---|---|---|---|
| Always guess on time | 10.94 | 0.5 | 0.0 | 0.0 |
| Airport + hour typical delay | 10.95 | 0.5977 | 0.222 | 0.001 |
| Ripple formula only (no ML) | 9.77 | 0.6137 | 0.97 | 0.156 |
| SkyRecover, planning (schedule only) | 10.51 | 0.6608 | 0.715 | 0.09 |
| SkyRecover, day-of (schedule + live) | 8.12 | 0.8224 | 0.79 | 0.427 |

**Avg error**: average minutes the predicted delay is off by (lower is better).
**AUC**: how well it ranks late flights above on-time ones (0.5 = coin flip, 1.0 = perfect).
**Precision**: of flights it flags as late, the share that really were.
**Recall**: of flights that really were late, the share it caught.

Top signals, SkyRecover, planning (schedule only):

- slack_minutes: 23.6%
- origin: 19.8%
- dest: 15.6%
- dep_hour: 12.1%
- leg_of_day: 11.7%
- day_of_week: 6.8%

Top signals, SkyRecover, day-of (schedule + live):

- inbound_arr_delay_known: 36.0%
- inbound_pushed_known: 9.9%
- inbound_dep_delay_known: 9.9%
- origin_recent_delay: 8.3%
- leg_of_day: 7.4%
- origin: 7.3%
