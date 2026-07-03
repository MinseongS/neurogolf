# int8 QLinearConv ranking-only scan (S10 handoff item)

**Date:** 2026-07-03  ·  **Source:** `reports/dtype_overpay_scan.json` PRODUCER_BOUND bucket  ·  **Recipe:** S10 tasks 264/184/365/191

## Verdict: DRY WELL — 0 of 32 RANKING_ONLY Conv candidates is a net win.

The S10 QLinearConv lever is **bit-exact but net-loss** for every PRODUCER_BOUND Conv, because of a structural mismatch the naive `dtype_overpay_scan` savings miss:

- Every one of the 32 RANKING_ONLY Convs reads the **free 10-channel fp32 `input[1,10,30,30]` directly** (these are 1×1 / dilated colour-index Convs: `color_f = Σ_k k·input_k`).
- `QLinearConv` requires a **quantized (uint8) input**. Quantizing the 10-channel input materializes a NEW counted `[1,10,30,30]` uint8 plane = **9000 B**.
- That 9000 B dwarfs the ≤2700 B freed on the (now uint8) single-channel output plane. **Net = save − 9000 < 0 for all 32.**

The S10 wins (264/184/365) fed QLinearConv a **single-channel already-integer plane** (gray-mask / Sign output), so their quantized-input copy was ~256–512 B, not 9000 B. PRODUCER_BOUND = *reads the raw 10-channel input* = exactly the case where the 10× channel fan-in kills the lever. This is the same wall documented in `reports/candidates/task041_signed.py` for fp16 recast.

**Amortization also fails.** For tasks with several PRODUCER_BOUND Convs reading the same input (e.g. 138/177/224/233), one shared uint8 input copy could serve all of them. But the largest per-task aggregate Conv-output saving is only **5052 B (task233)** — still far below the single 9000 B input plane. No task recovers the cost even when the quantized input is shared.

## Class counts (65 Conv-produced PRODUCER_BOUND tensors)

| class | count | note |
|---|---:|---|
| RANKING_ONLY | 32 | consumers order-only (TopK/ArgMax/threshold); int8 bit-exact — but net-loss |
| MIXED | 23 | some paths hit value-sensitive arithmetic |
| VALUE_SENSITIVE | 10 | feeds arithmetic where exact values matter |

(764 total PRODUCER_BOUND tensors; only 65 are Conv-produced = QLinearConv-eligible. The other 699 are Slice/Einsum/ReduceSum/ReduceMax/GatherND/MaxPool/etc — noted but skipped, QLinearConv N/A.)

## Top-10 RANKING_ONLY by naive (output-only) est delta — ALL refuted by input-quant cost

| # | task | tensor | out bytes freed | +uint8 input cost | **honest net** | naive Δpts | honest Δpts | consumers | owned |
|--:|--:|---|--:|--:|--:|--:|--:|---|:--:|
| 1 | 74 | `color_f` | 2700 | +9000 | **-6300** | +0.3543 | 0.0 | Cast |  |
| 2 | 383 | `color_f` | 1728 | +9000 | **-7272** | +0.335 | 0.0 | Cast |  |
| 3 | 80 | `colf32` | 2700 | +9000 | **-6300** | +0.2952 | 0.0 | Cast |  |
| 4 | 79 | `color_f` | 588 | +9000 | **-8412** | +0.2071 | 0.0 | Cast |  |
| 5 | 138 | `cgf32` | 1950 | +9000 | **-7050** | +0.167 | 0.0 | Cast |  |
| 6 | 201 | `labels_f` | 507 | +9000 | **-8493** | +0.1638 | 0.0 | Cast |  |
| 7 | 322 | `color_index` | 27 | +9000 | **-8973** | +0.145 | 0.0 | Cast |  |
| 8 | 185 | `p3_grid_f32` | 243 | +9000 | **-8757** | +0.1323 | 0.0 | Cast |  |
| 9 | 178 | `col0_labels_wide` | 90 | +9000 | **-8910** | +0.1255 | 0.0 | Cast |  |
| 10 | 132 | `scalar14x15` | 420 | +9000 | **-8580** | +0.1084 | 0.0 | TopK |  |

**Naive headline (output-plane-only) total est delta across all 32 = +3.030 pts.**  
**Honest total after input-quantization cost = 0.000 pts (0 positive-net candidates).**

## Empirical builds (top-3 by naive est delta, none owned)

Built + gated to convert the theoretical refutation into measured evidence. All three: mechanism is bit-exact (fresh 0 divergence, bundled fail=0) but cost rises exactly as predicted.

| task | candidate | inc mem | cand mem | Δmem | inc pts | cand pts | Δpts | bundled | fresh div |
|--:|---|--:|--:|--:|--:|--:|--:|:--:|:--:|
| 74 | `reports/candidates/task074_qconv.py` | 9000 | 15300 | +6300 | 15.889 | 15.360 | **−0.529** | 267/267 | 0/2000 |
| 80 | `reports/candidates/task080_qconv.py` | 10109 | 16409 | +6300 | 15.735 | 15.266 | **−0.469** | 231/231 | 0/2000 |
| 383 | `reports/candidates/task383_qconv.py` | 5974 | 13246 | +7272 | 16.289 | 15.498 | **−0.791** | 266/266 | 0/2000 |

Δmem matches prediction exactly: +9000 (uint8 input plane) − output-plane saving (2700 for 74/80; 1728 for 383) = +6300 / +6300 / +7272. task383 also folds the +11 bias as int32 (QLinearConv input 8, scale=1) — bit-exact.

## Recommendation

- **Do NOT adopt.** Close the 'Conv→ranking-only int8 전수 스캔' queue item as a DRY WELL. The PRODUCER_BOUND bucket is QLinearConv-immune for the same reason it is fp16-recast-immune: any dtype narrowing of a Conv that reads the raw 10-channel input must first narrow that 10-channel input, and the 10× fan-in always exceeds the single-channel output saving.

- The S10 QLinearConv lever remains valid ONLY where the Conv's **data input is already a small (1–2 channel) integer plane** (detection/hash/occupancy Convs fed by a Sign/gray mask). Those are NOT in the PRODUCER_BOUND bucket — they are interior planes. A future scan should re-target `producer_op==Conv` tensors whose **conv input is ≤2 channels**, not the free-input readers.
