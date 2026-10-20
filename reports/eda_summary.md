# SentinelAI — HDFS_v1 Exploratory Data Analysis Summary

_Generated 2026-10-19 from the real Loghub HDFS_v1 dataset (575,061 blocks) in `data/raw/hdfs/`. All figures under `reports/figures/`. No models were trained._

## 1. Dataset overview

- blocks (traces): **575,061**
- event features: **29** (E1..E29)
- template IDs match event columns: **True** (29 templates)
- occurrence matrix: 575,061 x 32
- traces: 575,061 x 6 (Features, TimeInterval, Latency)
- cross-check: mean sequence length from Features = 19.43 vs mean occurrences per row from matrix = 19.43 (consistent: True)

## 2. Class distribution and imbalance

| class | count | percent |
|---|---:|---:|
| Normal | 558,223 | 97.07% |
| Anomaly | 16,838 | 2.93% |

- anomaly share: **2.93%** of blocks
- imbalance ratio: **33.2** normal blocks per anomalous block

## 3. Missing values

| file | column | missing | percent |
|---|---|---:|---:|
| occurrence_matrix | Type | 558,223 | 97.07% |
| traces | Type | 558,223 | 97.07% |

- `Type` is empty on **558,223** matrix rows and **558,223** trace rows — exactly the normal rows; Type is populated only on anomalous rows (failure-type code) and must be excluded from model features.

## 4. Event frequency (top 10)

| event | total occurrences | blocks with event | mean per block | template |
|---|---:|---:|---:|---|
| E5 | 1,723,232 | 575,061 | 2.997 | [*]Receiving block[*]src:[*]dest:[*] |
| E26 | 1,719,741 | 568,854 | 2.991 | [*]BLOCK* NameSystem[*]addStoredBlock: blockMap upda... |
| E11 | 1,706,679 | 568,880 | 2.968 | [*]PacketResponder[*]for block[*]terminating[*] |
| E9 | 1,706,514 | 568,854 | 2.968 | [*]Received block[*]of size[*]from[*] |
| E21 | 1,402,047 | 466,647 | 2.438 | [*]Deleting block[*]file[*] |
| E23 | 1,396,174 | 465,015 | 2.428 | [*]BLOCK* NameSystem[*]delete:[*]is added to invalid... |
| E22 | 575,061 | 575,061 | 1 | [*]BLOCK* NameSystem[*]allocateBlock:[*] |
| E3 | 428,726 | 137,434 | 0.7455 | [*]Served block[*]to[*] |
| E4 | 356,207 | 129,295 | 0.6194 | [*]Got exception while serving[*]to[*] |
| E2 | 120,036 | 82,705 | 0.2087 | [*]Verification succeeded for[*] |

- total event occurrences across all blocks: **11,175,629**
- top-1 event share: **15.4%** of all occurrences
- top-5 events share: **73.9%** of all occurrences

## 5. Event occurrence distribution per trace

| stat | Normal | Anomaly |
|---|---:|---:|
| mean occurrences per trace | 19.5 | 17.12 |
| median occurrences per trace | 19 | 20 |
| max occurrences in one trace | 298 | 284 |
| mean distinct event types | 7.255 | 7.159 |
| max distinct event types | 14 | 20 |

- global: min 2, mean 19.43, max 298 occurrences per trace

## 6. Discriminative events (top 10 by |point-biserial r|)

| event | normal rate | anomaly rate | ratio anom/norm | point-biserial r | p-value |
|---|---:|---:|---:|---:|---:|
| E9 | 3 | 1.891 | 0.63 | -0.6026 | 0.00e+00 |
| E11 | 3 | 1.901 | 0.63 | -0.5978 | 0.00e+00 |
| E26 | 3.009 | 2.391 | 0.79 | -0.2549 | 0.00e+00 |
| E23 | 2.444 | 1.896 | 0.78 | -0.0781 | 0.00e+00 |
| E21 | 2.448 | 2.107 | 0.86 | -0.0486 | 1.63e-297 |
| E20 | 0.0003977 | 0.3161 | 794.92 | +0.5256 | 0.00e+00 |
| E5 | 3.005 | 2.729 | 0.91 | -0.2107 | 0.00e+00 |
| E6 | 0.004688 | 0.2661 | 56.75 | +0.2999 | 0.00e+00 |
| E18 | 0.004556 | 0.2648 | 58.13 | +0.3123 | 0.00e+00 |
| E25 | 0.004556 | 0.2648 | 58.13 | +0.3123 | 0.00e+00 |

- constant events (zero variance, no discriminative information): E22 (constant count 1 in every block)
- rare events (occur in < 1,000 blocks): E1 (5 blocks); E8 (46 blocks); E10 (67 blocks); E12 (26 blocks); E13 (488 blocks); E14 (63 blocks); E15 (38 blocks); E17 (9 blocks); E19 (4 blocks); E24 (4 blocks); E27 (974 blocks); E29 (46 blocks)

## 7. Extreme event counts

| event | max count in one block |
|---|---:|
| E3 | 203 |
| E4 | 41 |
| E5 | 13 |
| E26 | 13 |
| E21 | 13 |

| top blocks | total occurrences | class |
|---|---:|---|
| blk_-4145674605155741075 | 298 | Normal |
| blk_-2891794341254261063 | 284 | Anomaly |
| blk_2813981518546746323 | 280 | Normal |
| blk_-2825351351457839825 | 278 | Anomaly |
| blk_9014620365357651780 | 277 | Normal |

## 8. Correlations

- event-count Pearson correlation (29 x 29): max |r| off-diagonal = **1.000**
- strongest pairs: E18–E25 (+1.000); E16–E25 (+0.997); E16–E18 (+0.997); E9–E11 (+0.996); E21–E23 (+0.994)
- top pair E18–E25: identical count vectors in 575,061 blocks (100.0%) — redundant features
- verdict: heatmap generated (`correlation_heatmap.png`).
- trace length vs latency: Pearson r = +0.619 (p = 0.00e+00), Spearman rho = +0.831 (p = 0.00e+00)

## 9. Figures

- `reports/figures/class_distribution.png`
- `reports/figures/top_event_frequency.png`
- `reports/figures/normal_vs_anomaly_events.png`
- `reports/figures/event_count_distribution.png`
- `reports/figures/correlation_heatmap.png`

