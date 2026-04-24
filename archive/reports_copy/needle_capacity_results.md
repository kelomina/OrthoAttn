# Needle Capacity Results

| Context | Forward Only | Forward Peak Mem (MB) | Train Step | Train Peak Mem (MB) |
|---:|:---:|---:|:---:|---:|
| 16384 | PASS | 31.45 | PASS | 1315.02 |
| 32768 | PASS | 46.81 | PASS | 2610.98 |
| 65536 | PASS | 84.81 | PASS | 5202.11 |
| 131072 | PASS | 157.81 | PASS | 10340.00 |
| 262144 | PASS | 305.81 | PASS | 20734.86 |
| 524288 | PASS | 601.81 | OOM | - |
| 1048576 | PASS | 1193.81 | OOM | - |
| 2097152 | PASS | 2377.81 | OOM | - |
