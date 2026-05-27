# Precision Validation for LoopTF + PonderNet Copy Experiments

本文档说明本项目中新加入的 **precision 验证实验** 代码，包括实验目的、新增脚本、如何运行、如何汇总结果，以及最终输出文件的位置。

---

## 1. 实验目的

本组实验用于验证：

> 降低模型参数 precision 是否会促使 Looped Transformer + PonderNet 学到更加 length-uniform、可 extrapolate 的 copy 解。

具体来说，我们在 copy ponder 设置下，对不同 positional encoding 和不同参数精度进行对照实验：

- Positional Encoding:
  - `nope`
  - `rope`

- Precision / Quantization:
  - `32-bit`: 原始 fp32 模型，相当于不做额外 precision 投影
  - `16-bit`: 将模型参数投影到 16-bit precision
  - `8-bit`: 将模型参数投影到 8-bit precision

- Seeds:
  - 支持多 seed，例如 `0, 1, 2, 3, 42`

核心关注点包括：

1. NoPE 和 RoPE 在不同 precision 下是否仍然呈现不同的 extrapolation 行为。
2. 降低 precision 是否能改善 RoPE 的 OOD extrapolation。
3. NoPE 的 long-range extrapolation 是否存在明显 seed variance。
4. 8-bit precision 是否能够稳定完成 ID training。

---

## 2. 新增 / 使用的主要脚本

### 2.1 `run_precision_ponder_one.sh`

该脚本用于运行单个 precision ponder 实验。

调用格式：

```bash
bash run_precision_ponder_one.sh <pe> <bits> <seed> <train_steps> <eval_every>
```

参数含义：

| 参数 | 含义 | 示例 |
|---|---|---|
| `pe` | positional encoding 类型 | `nope`, `rope` |
| `bits` | 参数 precision bit 数 | `32`, `16`, `8` |
| `seed` | 随机种子 | `0`, `1`, `2`, `42` |
| `train_steps` | 训练步数 | `100001` |
| `eval_every` | evaluation 间隔 | `1000` |

示例：

```bash
bash run_precision_ponder_one.sh nope 32 2 100001 1000
```

表示运行：

```text
PE = nope
precision = 32-bit
seed = 2
train_steps = 100001
eval_every = 1000
```

---

### 2.2 `run_precision_ponder_grid.sh`

该脚本用于批量运行一组 precision ponder 实验。

它会循环遍历：

```text
PE_LIST
BITS_LIST
SEEDS_LIST
STEPS_LIST
```

并对每个组合调用：

```bash
bash run_precision_ponder_one.sh "$pe" "$bits" "$seed" "$steps" "$EVAL_EVERY"
```

默认设置通常类似：

```bash
PE_LIST="${PE_LIST:-rope nope}"
BITS_LIST="${BITS_LIST:-32 16}"
SEEDS_LIST="${SEEDS_LIST:-1}"
STEPS_LIST="${STEPS_LIST:-100001}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
FAIL_FAST="${FAIL_FAST:-false}"
```

其中：

| 环境变量 | 含义 |
|---|---|
| `PE_LIST` | 要运行的 positional encoding 列表 |
| `BITS_LIST` | 要运行的 precision bit 列表 |
| `SEEDS_LIST` | 要运行的 seed 列表 |
| `STEPS_LIST` | 训练步数列表 |
| `EVAL_EVERY` | evaluation 间隔 |
| `FAIL_FAST` | 某个 run 失败后是否立即停止 |

---

### 2.3 `summarize_precision_results.py`

该脚本用于扫描 precision 实验目录，收集所有 run 的 diagnostics 结果，并生成统一的 summary CSV 和 run index CSV。

常用运行方式：

```bash
python summarize_precision_results.py \
  --root ../models/precision_ponder \
  --out ../models/precision_ponder/precision_ponder_summary_after_seed2.csv \
  --index-out ../models/precision_ponder/precision_ponder_run_index_after_seed2.csv
```

输出文件：

```text
../models/precision_ponder/precision_ponder_summary_after_seed2.csv
../models/precision_ponder/precision_ponder_run_index_after_seed2.csv
```

其中：

| 文件 | 作用 |
|---|---|
| `precision_ponder_summary_after_seed2.csv` | 汇总所有 diagnostics 结果 |
| `precision_ponder_run_index_after_seed2.csv` | 记录每个 run 的配置、run_id、summary_csv 路径等信息 |

建议每次补完一个新的 seed 后，都生成一个带 seed 标记的新 summary 文件，避免覆盖之前结果。

---

### 2.4 `export_precision_ponder_txts.py`

该脚本用于根据 `precision_ponder_run_index_after_seedX.csv` 导出指定 seed 的实验结果 txt 文件。

运行示例：

```bash
mkdir -p ../reports/precision_ponder_seed2

python export_precision_ponder_txts.py \
  --index ../models/precision_ponder/precision_ponder_run_index_after_seed2.csv \
  --outdir ../reports/precision_ponder_seed2 \
  --seed 2 \
  --pes nope rope \
  --bits 32 16 8
```

输出文件：

```text
../reports/precision_ponder_seed2/nope_copy_ponder_bits32_seed2.txt
../reports/precision_ponder_seed2/nope_copy_ponder_bits16_seed2.txt
../reports/precision_ponder_seed2/nope_copy_ponder_bits8_seed2.txt
../reports/precision_ponder_seed2/rope_copy_ponder_bits32_seed2.txt
../reports/precision_ponder_seed2/rope_copy_ponder_bits16_seed2.txt
../reports/precision_ponder_seed2/rope_copy_ponder_bits8_seed2.txt
```

如果某些配置没有对应 run，脚本会跳过并提示 `no matching run`。

如果导出文件已经存在，默认不会覆盖。若确认要覆盖当前 seed 的 txt，可以加：

```bash
--overwrite
```

例如：

```bash
python export_precision_ponder_txts.py \
  --index ../models/precision_ponder/precision_ponder_run_index_after_seed2.csv \
  --outdir ../reports/precision_ponder_seed2 \
  --seed 2 \
  --pes nope rope \
  --bits 32 16 8 \
  --overwrite
```

这只会覆盖：

```text
../reports/precision_ponder_seed2/
```

目录下的 seed2 文件，不会影响 seed0 或 seed1 的结果。

---

## 3. 如何运行实验

以下命令默认在项目的 `src` 目录下运行：

```bash
cd /data/diaozb/looped-tf-length-generalization/src
```

---

## 3.1 单个实验运行方式

例如运行 NoPE、32-bit、seed=2：

```bash
bash run_precision_ponder_one.sh nope 32 2 100001 1000
```

运行 RoPE、16-bit、seed=2：

```bash
bash run_precision_ponder_one.sh rope 16 2 100001 1000
```

运行 NoPE、8-bit、seed=2：

```bash
bash run_precision_ponder_one.sh nope 8 2 100001 1000
```

---

## 3.2 批量运行一个 seed 的完整实验

例如 seed=2，一共运行：

```text
nope bits32 seed2
nope bits16 seed2
nope bits8 seed2
rope bits32 seed2
rope bits16 seed2
rope bits8 seed2
```

可以运行：

```bash
mkdir -p ../logs/precision_ponder_grid

PE_LIST="rope nope" \
BITS_LIST="32 16 8" \
SEEDS_LIST="2" \
STEPS_LIST="100001" \
EVAL_EVERY="1000" \
FAIL_FAST="false" \
bash run_precision_ponder_grid.sh 2>&1 | tee ../logs/precision_ponder_grid/grid_seed2_$(date +%Y%m%d_%H%M%S).log
```

日志会保存到：

```text
../logs/precision_ponder_grid/grid_seed2_YYYYMMDD_HHMMSS.log
```

---

## 3.3 批量运行多个 seed

例如同时运行 seed=1,2,3：

```bash
mkdir -p ../logs/precision_ponder_grid

PE_LIST="rope nope" \
BITS_LIST="32 16" \
SEEDS_LIST="1 2 3" \
STEPS_LIST="100001" \
EVAL_EVERY="1000" \
FAIL_FAST="false" \
bash run_precision_ponder_grid.sh 2>&1 | tee ../logs/precision_ponder_grid/grid_seed1_2_3_$(date +%Y%m%d_%H%M%S).log
```

这会运行：

```text
2 PE × 2 precision × 3 seeds = 12 runs
```

如果要包含 8-bit，则改成：

```bash
BITS_LIST="32 16 8"
```

---

## 3.4 推荐使用 tmux 运行

为了避免 SSH 断连导致实验中断，推荐使用 `tmux`。

创建 tmux session：

```bash
tmux new -s precision_seed2
```

进入后运行实验：

```bash
cd /data/diaozb/looped-tf-length-generalization/src

mkdir -p ../logs/precision_ponder_grid

PE_LIST="rope nope" \
BITS_LIST="32 16 8" \
SEEDS_LIST="2" \
STEPS_LIST="100001" \
EVAL_EVERY="1000" \
FAIL_FAST="false" \
bash run_precision_ponder_grid.sh 2>&1 | tee ../logs/precision_ponder_grid/grid_seed2_$(date +%Y%m%d_%H%M%S).log
```

后台挂起 tmux：

```text
Ctrl-b d
```

重新进入：

```bash
tmux attach -t precision_seed2
```

---

## 4. 实验结果保存位置

### 4.1 模型和 diagnostics 输出

每个 run 的模型、config、diagnostics 会保存在：

```text
../models/precision_ponder/
```

其中每个具体 run 目录下通常会包含：

```text
config.yaml
diagnostics_summary.csv
```

可以用下面命令查看所有 diagnostics：

```bash
find ../models/precision_ponder -name "diagnostics_summary.csv" | sort
```

---

### 4.2 grid 运行日志

grid 脚本的运行日志保存在：

```text
../logs/precision_ponder_grid/
```

例如：

```text
../logs/precision_ponder_grid/grid_seed2_20260527_153000.log
```

检查是否有失败：

```bash
grep -R "Traceback\|ERROR\|failed\|Run failed" \
  ../logs/precision_ponder \
  ../logs/precision_ponder_grid
```

---

### 4.3 汇总 CSV 输出

运行：

```bash
python summarize_precision_results.py \
  --root ../models/precision_ponder \
  --out ../models/precision_ponder/precision_ponder_summary_after_seed2.csv \
  --index-out ../models/precision_ponder/precision_ponder_run_index_after_seed2.csv
```

会生成：

```text
../models/precision_ponder/precision_ponder_summary_after_seed2.csv
../models/precision_ponder/precision_ponder_run_index_after_seed2.csv
```

建议不同 seed 或不同阶段使用不同文件名，例如：

```text
precision_ponder_summary_after_seed1.csv
precision_ponder_run_index_after_seed1.csv

precision_ponder_summary_after_seed2.csv
precision_ponder_run_index_after_seed2.csv

precision_ponder_summary_after_seed42.csv
precision_ponder_run_index_after_seed42.csv
```

这样不会覆盖之前的结果。

---

### 4.4 导出的 txt 结果

运行：

```bash
mkdir -p ../reports/precision_ponder_seed2

python export_precision_ponder_txts.py \
  --index ../models/precision_ponder/precision_ponder_run_index_after_seed2.csv \
  --outdir ../reports/precision_ponder_seed2 \
  --seed 2 \
  --pes nope rope \
  --bits 32 16 8
```

会生成：

```text
../reports/precision_ponder_seed2/nope_copy_ponder_bits32_seed2.txt
../reports/precision_ponder_seed2/nope_copy_ponder_bits16_seed2.txt
../reports/precision_ponder_seed2/nope_copy_ponder_bits8_seed2.txt
../reports/precision_ponder_seed2/rope_copy_ponder_bits32_seed2.txt
../reports/precision_ponder_seed2/rope_copy_ponder_bits16_seed2.txt
../reports/precision_ponder_seed2/rope_copy_ponder_bits8_seed2.txt
```

查看导出结果：

```bash
ls -lh ../reports/precision_ponder_seed2

head -40 ../reports/precision_ponder_seed2/nope_copy_ponder_bits32_seed2.txt
```

---

## 5. 完整 seed=2 流程示例

下面是从运行实验到导出 txt 的完整流程。

### Step 1: 进入项目目录

```bash
cd /data/diaozb/looped-tf-length-generalization/src
```

### Step 2: 确认脚本可执行

```bash
chmod +x run_precision_ponder_grid.sh
chmod +x run_precision_ponder_one.sh
```

### Step 3: 运行 seed=2 的 6 个实验

```bash
mkdir -p ../logs/precision_ponder_grid

PE_LIST="rope nope" \
BITS_LIST="32 16 8" \
SEEDS_LIST="2" \
STEPS_LIST="100001" \
EVAL_EVERY="1000" \
FAIL_FAST="false" \
bash run_precision_ponder_grid.sh 2>&1 | tee ../logs/precision_ponder_grid/grid_seed2_$(date +%Y%m%d_%H%M%S).log
```

### Step 4: 检查是否有失败

```bash
grep -R "Traceback\|ERROR\|failed\|Run failed" \
  ../logs/precision_ponder \
  ../logs/precision_ponder_grid
```

### Step 5: 汇总结果

```bash
python summarize_precision_results.py \
  --root ../models/precision_ponder \
  --out ../models/precision_ponder/precision_ponder_summary_after_seed2.csv \
  --index-out ../models/precision_ponder/precision_ponder_run_index_after_seed2.csv
```

### Step 6: 导出 seed=2 的 txt

```bash
mkdir -p ../reports/precision_ponder_seed2

python export_precision_ponder_txts.py \
  --index ../models/precision_ponder/precision_ponder_run_index_after_seed2.csv \
  --outdir ../reports/precision_ponder_seed2 \
  --seed 2 \
  --pes nope rope \
  --bits 32 16 8
```

### Step 7: 查看结果

```bash
ls -lh ../reports/precision_ponder_seed2

head -40 ../reports/precision_ponder_seed2/nope_copy_ponder_bits32_seed2.txt
head -40 ../reports/precision_ponder_seed2/rope_copy_ponder_bits32_seed2.txt
```

---

## 6. 如何确认 seed 是否被正确识别

运行：

```bash
python - <<'PY'
import pandas as pd

idx = pd.read_csv("../models/precision_ponder/precision_ponder_run_index_after_seed2.csv")

cols = ["pe", "quant_tag", "weight_bits", "seed", "run_id", "summary_csv"]
cols = [c for c in cols if c in idx.columns]

print(
    idx[idx["seed"].astype(str).str.replace(".0", "", regex=False) == "2"][cols]
    .sort_values(["pe", "weight_bits"])
    .to_string(index=False)
)
PY
```

正常情况下，应该看到 6 行：

```text
nope bits32 seed2
nope bits16 seed2
nope bits8 seed2
rope bits32 seed2
rope bits16 seed2
rope bits8 seed2
```

---

## 7. 结果文件字段说明

导出的 txt 和 summary CSV 中常见字段包括：

| 字段 | 含义 |
|---|---|
| `split` | ID / OOD split |
| `length` | 测试序列长度 |
| `auto_answer_acc` | PonderNet 自动 halting 下的 answer accuracy |
| `auto_code_acc` | PonderNet 自动 halting 下的 code accuracy |
| `expected_exit_step` | 期望退出步数 |
| `argmax_exit_step_mean` | argmax halting step 的平均值 |
| `argmax_exit_step_mode` | argmax halting step 的众数 |
| `best_forced_answer_acc` | forced loop step 搜索下的最佳 answer accuracy |
| `best_forced_answer_step` | 达到最佳 answer accuracy 的 forced step |
| `min_answer_step_loss` | 最小 answer loss |
| `min_answer_loss_step` | 达到最小 answer loss 的 step |

重点关注：

```text
auto_answer_acc
best_forced_answer_acc
best_forced_answer_step
expected_exit_step
argmax_exit_step_mean
```

---

## 8. 推荐分析方式

### 8.1 对比 NoPE vs RoPE

重点看：

```text
NoPE bits32 / bits16 是否有 OOD extrapolation
RoPE bits32 / bits16 是否 OOD 从 length 21 开始失败
```

如果 RoPE 在不同 seed 和不同 precision 下都没有 OOD extrapolation，说明 RoPE 的 failure 比较稳定。

---

### 8.2 对比 32-bit vs 16-bit

重点看：

```text
16-bit 是否改变 qualitative pattern
```

如果 16-bit 只是使 accuracy 略微下降，但没有让 RoPE extrapolate，也没有改变 NoPE/RoPE 的主体差异，则说明：

```text
降低 precision 本身不足以带来 length extrapolation
```

---

### 8.3 检查 8-bit

如果 8-bit 在 ID length 上已经失败，则不应将其作为 extrapolation 对照。

此时 8-bit 更适合作为 training stability negative case：

```text
8-bit all-parameter precision projection may be too aggressive and can break ID training.
```

---

### 8.4 检查 seed variance

对每个 seed 分别导出：

```text
../reports/precision_ponder_seed0/
../reports/precision_ponder_seed1/
../reports/precision_ponder_seed2/
../reports/precision_ponder_seed42/
```

然后比较：

```text
nope_copy_ponder_bits32_seedX.txt
nope_copy_ponder_bits16_seedX.txt
rope_copy_ponder_bits32_seedX.txt
rope_copy_ponder_bits16_seedX.txt
```

重点看 NoPE 的 far-OOD lengths，例如：

```text
length = 40
length = 60
```

如果不同 seed 下 NoPE 的 long-range OOD accuracy 差异明显，则说明：

```text
NoPE increases the chance of learning a length-uniform computation, but does not guarantee it.
```

---

## 9. 当前阶段可以支持的结论

根据目前 precision 验证实验，比较稳妥的阶段性结论是：

```text
Lowering numerical precision alone is not sufficient for length extrapolation.
```

更具体地说：

1. `32-bit` 和 `16-bit` 通常保持相同的 qualitative pattern：
   - NoPE 仍然比 RoPE 更容易 extrapolate。
   - RoPE 仍然在 OOD length 上失败。
2. `16-bit` 更像是 fp32 模型的轻微劣化版本，而不是一种新的可 extrapolate solution。
3. `8-bit` 在当前 all-parameter precision projection 设置下容易导致 ID training 崩溃，因此不能作为有效的 extrapolation evidence。
4. NoPE 的 far-OOD extrapolation 可能存在 seed sensitivity，需要多 seed 统计确认。
5. 因此，precision alone 不是关键因素；更关键的是模型是否学到了 semantic length-uniform computation，以及 positional encoding 是否诱导了 position-specific shortcut。

---

## 10. 常见问题

### Q1: 如何避免覆盖之前的结果？

汇总时使用带 seed 的文件名：

```bash
--out ../models/precision_ponder/precision_ponder_summary_after_seed2.csv
--index-out ../models/precision_ponder/precision_ponder_run_index_after_seed2.csv
```

导出 txt 时使用带 seed 的目录：

```bash
--outdir ../reports/precision_ponder_seed2
--seed 2
```

这样不会覆盖：

```text
../reports/precision_ponder_seed0/
../reports/precision_ponder_seed1/
```

---

### Q2: 如果导出 txt 时提示 already exists 怎么办？

说明该 seed 的 txt 已经导出过。

如果不想覆盖，什么都不用做。

如果确认要重新导出，可以加：

```bash
--overwrite
```

---

### Q3: 如果某些配置显示 no matching run 怎么办？

说明 `precision_ponder_run_index_after_seedX.csv` 中没有找到对应配置。

需要检查：

1. 该实验是否真的跑完。
2. 对应 run 目录下是否有 `diagnostics_summary.csv`。
3. `seed` 是否正确。
4. `bits` 是否正确。
5. `pe` 是否是 `nope` 或 `rope`。
6. `summarize_precision_results.py` 是否在实验跑完后重新执行过。

可以用：

```bash
find ../models/precision_ponder -name "diagnostics_summary.csv" | sort
```

检查 diagnostics 是否存在。

---

### Q4: 8-bit ID 崩了还需要继续分析 OOD 吗？

一般不需要。

如果模型在 ID length 上都没有学会 copy，那么它的 OOD accuracy 没有太多 extrapolation 解释意义。

此时 8-bit 可以作为：

```text
training stability negative result
```

而不是作为：

```text
extrapolation comparison
```

---

## 11. 建议的最终目录结构

完成多个 seed 后，推荐结果结构如下：

```text
../models/precision_ponder/
  precision_ponder_summary_after_seed1.csv
  precision_ponder_run_index_after_seed1.csv
  precision_ponder_summary_after_seed2.csv
  precision_ponder_run_index_after_seed2.csv
  precision_ponder_summary_after_seed42.csv
  precision_ponder_run_index_after_seed42.csv

../reports/
  precision_ponder_seed1/
    nope_copy_ponder_bits32_seed1.txt
    nope_copy_ponder_bits16_seed1.txt
    rope_copy_ponder_bits32_seed1.txt
    rope_copy_ponder_bits16_seed1.txt

  precision_ponder_seed2/
    nope_copy_ponder_bits32_seed2.txt
    nope_copy_ponder_bits16_seed2.txt
    nope_copy_ponder_bits8_seed2.txt
    rope_copy_ponder_bits32_seed2.txt
    rope_copy_ponder_bits16_seed2.txt
    rope_copy_ponder_bits8_seed2.txt

../logs/
  precision_ponder_grid/
    grid_seed1_YYYYMMDD_HHMMSS.log
    grid_seed2_YYYYMMDD_HHMMSS.log
```
