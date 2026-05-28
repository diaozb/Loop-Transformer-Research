# Copy + Ponder Sharpness Experiment README

本文档说明当前新增的 **sharpness 测试实验**：它用于比较 `copy + PonderNet` 设置下，NoPE 和 RoPE checkpoint 在参数扰动后的稳定性差异。

当前实验不是训练新模型，而是对已经训练好的 checkpoint 做 post-training evaluation。

---

## 1. 实验目的

本实验要回答的问题是：

> Copy + Ponder 模型在 NoPE 和 RoPE 下学到的解，是否在参数空间附近同样稳定？

更具体地说，我们测试：

```text
已有 checkpoint θ
→ 对参数加随机扰动 θ + δ
→ 重新评估 Ponder objective / accuracy / expected exit step
→ 比较扰动前后变化
```

核心 sharpness 指标是：

```text
delta_objective = objective(θ + δ) - objective(θ)
```

其中 objective 是完整 Ponder objective：

```text
objective = reconstruction loss + beta * KL(halting distribution || geometric prior)
```

所以本实验不是只看 accuracy，也不是只看某个 fixed step 的 loss，而是看 Ponder 模型真实训练目标在参数扰动下的变化。

---

## 2. 当前正式实验对象

当前正式比较的是：

```text
Task: copy
Training: PonderNet / adaptive exit
PE comparison: NoPE vs RoPE
Checkpoint: best.pt
```

正式使用的 run 是：

```text
NoPE:
../models/nope_baselines/copy_ponder/62465a36-9f94-4157-8f93-edac50a7630d

RoPE:
../models/rope_baselines/copy_ponder/85c8ce6d-4abc-4432-a343-739683bc4bf7
```

注意：

```text
../models/nope_baselines/copy_ponder/deaab117-4677-4d40-85cb-14b8fc7d0506
```

不建议作为正式 NoPE 对照，因为它的 base accuracy 不够好。

```text
../models/nope_baselines/copy_ponder/5e292703-5b20-434e-96df-bdaed29f6fd0
```

主要用于 smoke test，不是正式结果。

---

## 3. 新增 / 使用的文件

### 3.1 新增核心脚本：`eval_ponder_sharpness.py`

位置：

```text
src/eval_ponder_sharpness.py
```

作用：

```text
加载一个 Copy + Ponder checkpoint
生成指定长度的 copy evaluation batch
计算原模型的 Ponder objective / accuracy / expected exit step
对模型参数加入随机扰动
重新计算 perturbed objective / accuracy / expected exit step
输出 raw CSV、summary CSV 和 sharpness plot
```

它是本实验最核心的新代码。

主要参数：

```text
--run-dir       一个完整 run 目录，里面应包含 best.pt 和 config.yaml
--checkpoint    checkpoint 文件名，通常是 best.pt
--out-dir       sharpness 输出目录
--lengths       要评估的长度，例如 1-20,21,22,40,60
--epsilons      参数扰动强度，例如 0,1e-4,3e-4,1e-3,3e-3,1e-2
--directions    每个 epsilon 下随机扰动方向数量
--n-batches     每个 length / epsilon / direction 下评估多少 batch
--batch-size    evaluation batch size
```

---

### 3.2 新增运行脚本：`run_copy_ponder_sharpness_formal.sh`

位置：

```text
src/run_copy_ponder_sharpness_formal.sh
```

作用：

```text
一键运行 NoPE 和 RoPE 的正式 sharpness evaluation
```

建议当前版本应使用下面两个 run：

```bash
NOPE_RUN_DIR="../models/nope_baselines/copy_ponder/62465a36-9f94-4157-8f93-edac50a7630d"
ROPE_RUN_DIR="../models/rope_baselines/copy_ponder/85c8ce6d-4abc-4432-a343-739683bc4bf7"
```

正式 sharpness 设置：

```bash
LENGTHS="1-20,21,22,40,60"
EPSILONS="0,1e-4,3e-4,1e-3,3e-3,1e-2"
DIRECTIONS=8
N_BATCHES=20
BATCH_SIZE=128
CHECKPOINT="best.pt"
OUT_NAME="sharpness_formal"
```

运行：

```bash
cd /data/diaozb/looped-tf-length-generalization/src

chmod +x run_copy_ponder_sharpness_formal.sh

./run_copy_ponder_sharpness_formal.sh
```

如果希望保留总日志：

```bash
./run_copy_ponder_sharpness_formal.sh 2>&1 | tee run_copy_ponder_sharpness_formal_$(date +%Y%m%d_%H%M%S).log
```

---

### 3.3 新增比较脚本：`compare_copy_ponder_sharpness_groups.sh`

位置：

```text
src/compare_copy_ponder_sharpness_groups.sh
```

作用：

```text
读取 NoPE 和 RoPE 的 sharpness_summary.csv
按 length group 汇总结果
生成 combined CSV、grouped CSV 和 comparison log
```

它把长度分成：

```text
ID_1_20          length 1-20
near_OOD_21_22   length 21,22
far_OOD_40       length 40
far_OOD_60       length 60
```

运行：

```bash
cd /data/diaozb/looped-tf-length-generalization/src

chmod +x compare_copy_ponder_sharpness_groups.sh

bash compare_copy_ponder_sharpness_groups.sh
```

---

### 3.4 可选检查脚本：`inspect_copy_ponder_runs.sh`

位置：

```text
src/inspect_copy_ponder_runs.sh
```

作用：

```text
扫描已有 copy_ponder run
打印 task / seed / train_steps / use_rope / use_wpe / checkpoint 是否存在
帮助确认哪个 run 是正式训练，哪个只是 smoke test
```

运行：

```bash
cd /data/diaozb/looped-tf-length-generalization/src

chmod +x inspect_copy_ponder_runs.sh

./inspect_copy_ponder_runs.sh | tee inspect_copy_ponder_runs.log
```

---

## 4. 原有文件的作用

### `train_ponder.py` / `train_ponder_nope.py`

作用：

```text
训练 Copy + PonderNet 模型
保存 best.pt、model.pt、config.yaml
训练完成后可生成 diagnostics
```

本 sharpness 实验不重新训练模型，只使用这些训练脚本已经生成的 checkpoint。

---

### `models.py`

作用：

```text
定义 GeneralTransformerModel
支持 NoPE / RoPE / WPE
提供 looped_forward 和 forward_single
```

Sharpness 脚本会通过 checkpoint 加载这个模型。

---

### `generate_training_data.py`

作用：

```text
生成 copy task 的 synthetic input / target / mask
```

Sharpness evaluation 复用它生成不同长度的 copy evaluation batch。

---

### `eval_ponder_diagnostics.py`

作用：

```text
评估 Ponder 模型的 accuracy、forced step、expected exit step、hidden convergence 等 diagnostics
```

它不是 sharpness 脚本，因为它不做参数扰动。

---

### `eval_copy_fixed_loop.py`

作用：

```text
评估 fixed-loop copy 模型
```

当前 sharpness 实验关注的是 Copy + Ponder，不是 fixed-loop，所以它不是当前主脚本。

---

## 5. 如何单独运行一个 sharpness evaluation

如果不想用 bash wrapper，也可以直接运行单个模型。

### NoPE

```bash
cd /data/diaozb/looped-tf-length-generalization/src

python eval_ponder_sharpness.py \
  --run-dir ../models/nope_baselines/copy_ponder/62465a36-9f94-4157-8f93-edac50a7630d \
  --checkpoint best.pt \
  --out-dir ../models/nope_baselines/copy_ponder/62465a36-9f94-4157-8f93-edac50a7630d/sharpness_formal \
  --lengths 1-20,21,22,40,60 \
  --n-batches 20 \
  --batch-size 128 \
  --epsilons 0,1e-4,3e-4,1e-3,3e-3,1e-2 \
  --directions 8
```

### RoPE

```bash
cd /data/diaozb/looped-tf-length-generalization/src

python eval_ponder_sharpness.py \
  --run-dir ../models/rope_baselines/copy_ponder/85c8ce6d-4abc-4432-a343-739683bc4bf7 \
  --checkpoint best.pt \
  --out-dir ../models/rope_baselines/copy_ponder/85c8ce6d-4abc-4432-a343-739683bc4bf7/sharpness_formal \
  --lengths 1-20,21,22,40,60 \
  --n-batches 20 \
  --batch-size 128 \
  --epsilons 0,1e-4,3e-4,1e-3,3e-3,1e-2 \
  --directions 8
```

---

## 6. 输出保存在哪里

### 6.1 NoPE sharpness 输出

```text
../models/nope_baselines/copy_ponder/62465a36-9f94-4157-8f93-edac50a7630d/sharpness_formal/
```

该目录下应包含：

```text
sharpness_raw.csv
sharpness_summary.csv
sharpness_delta_objective.png
```

---

### 6.2 RoPE sharpness 输出

```text
../models/rope_baselines/copy_ponder/85c8ce6d-4abc-4432-a343-739683bc4bf7/sharpness_formal/
```

该目录下应包含：

```text
sharpness_raw.csv
sharpness_summary.csv
sharpness_delta_objective.png
```

---

### 6.3 NoPE/RoPE 分组比较输出

```text
../eval/copy_ponder_sharpness_compare/
```

该目录下应包含：

```text
compare_copy_ponder_sharpness_<timestamp>.log
copy_ponder_sharpness_combined.csv
copy_ponder_sharpness_grouped.csv
```

其中：

```text
copy_ponder_sharpness_combined.csv
```

保存 NoPE 和 RoPE 合并后的逐 length / epsilon 结果。

```text
copy_ponder_sharpness_grouped.csv
```

保存按 length group 聚合后的结果，是最适合写报告和论文分析的文件。

---

## 7. 输出字段解释

### `sharpness_raw.csv`

保存更底层的结果，通常包含每个：

```text
length
epsilon
random direction
```

下的评估结果。

它适合用来重新计算 mean / std / max 等统计量。

---

### `sharpness_summary.csv`

保存按：

```text
length × epsilon
```

聚合后的结果。

当前正式 run 中应有：

```text
24 lengths × 6 epsilons = 144 rows
```

其中：

```text
lengths = 1,2,...,20,21,22,40,60
epsilons = 0, 0.0001, 0.0003, 0.001, 0.003, 0.01
```

重要字段：

| 字段                              | 含义                                    |
| ------------------------------- | ------------------------------------- |
| `length`                        | copy 序列长度                             |
| `epsilon`                       | 参数扰动强度                                |
| `objective_mean`                | 扰动后 Ponder objective 平均值              |
| `objective_std`                 | 不同随机扰动方向下 objective 标准差               |
| `delta_objective_mean`          | 扰动后 objective 相对 base objective 的平均变化 |
| `delta_objective_std`           | `delta_objective` 的标准差                |
| `rec_loss_mean`                 | reconstruction loss                   |
| `kl_loss_mean`                  | Ponder halting KL loss                |
| `auto_answer_acc_mean`          | auto halting 下 answer accuracy        |
| `delta_auto_answer_acc_mean`    | 扰动后 accuracy 相对 base 的变化              |
| `expected_exit_step_mean`       | expected halting step                 |
| `delta_expected_exit_step_mean` | 扰动后 expected exit step 相对 base 的变化    |

---

## 8. 当前正式结果摘要

### 8.1 Base performance: epsilon = 0

NoPE：

```text
ID 1-20 mean accuracy ≈ 0.99992
near OOD 21,22 accuracy ≈ 0.998
L40 accuracy ≈ 0.911
L60 accuracy ≈ 0.009
```

说明：

```text
NoPE 在训练长度内成功
NoPE 在 near OOD 也成功
NoPE 到 L40 仍能部分 extrapolate
NoPE 到 L60 基本失败
```

RoPE：

```text
ID 1-20 accuracy = 1.0
OOD 21,22,40,60 accuracy = 0.0
```

说明：

```text
RoPE 在训练长度内成功
但一超过训练长度就失败
```

---

### 8.2 Sharpness comparison: epsilon = 0.01

ID 1-20：

```text
NoPE delta_objective ≈ 3.32e-6
RoPE delta_objective ≈ 8.25e-7
```

解释：

```text
NoPE 和 RoPE 在 ID 上都很 flat / stable
ID sharpness 不是主要区别
```

near OOD 21-22：

```text
NoPE delta_objective ≈ 4.91e-5
RoPE delta_objective ≈ 8.04e-3
```

解释：

```text
RoPE 在 near OOD 的 objective sharpness 明显更大
NoPE 在 near OOD 仍然相对稳定
```

L40：

```text
NoPE delta_objective ≈ 0.001315
RoPE delta_objective ≈ 0.015221
```

解释：

```text
RoPE 在 L40 的扰动敏感性明显更强
NoPE 在 L40 仍有较好的 extrapolation 和较低 sharpness
```

L60：

```text
NoPE delta_objective ≈ 0.003640
RoPE delta_objective ≈ 0.003317
```

解释：

```text
L60 时 NoPE 和 RoPE 都基本失败
因此 L60 更像共同失败区间
不如 21/22 和 40 那么适合解释 NoPE/RoPE 差异
```

---

## 9. 当前可以写出的主要结论

当前 sharpness 结果支持以下结论：

```text
NoPE 和 RoPE 在 ID 长度上都很稳定，因此 ID sharpness 不能解释二者 extrapolation 差异。
真正的区别出现在训练长度之外。
NoPE 在 near-OOD 和 L40 上仍保持较低 objective 和较好的 accuracy；
RoPE 则从 L21 开始 accuracy 直接崩溃，并且在 near-OOD 和 L40 上显示更大的 objective sharpness。
这说明 RoPE 学到的是 ID-stable but OOD-fragile solution，
而 NoPE 更接近 length-uniform solution，至少在 moderate OOD length 上更稳定。
```

---

## 10. 注意事项

### 10.1 这是 random perturbation sharpness，不是 adversarial sharpness

当前实验使用随机参数扰动方向。

因此：

```text
delta_objective_mean < 0
```

偶尔出现是正常的，尤其在：

```text
epsilon 很小
directions 有限
eval batch 有采样噪声
模型不一定是该 eval distribution 上的严格局部最小点
```

所以不要过度解读很小 epsilon 下的正负号。

正式分析建议重点看：

```text
epsilon = 0.003
epsilon = 0.01
```

以及不同 length group 的相对趋势。

---

### 10.2 不要只看全长度平均

全长度平均会把：

```text
ID 1-20
near OOD 21-22
L40
L60
```

混在一起。

正式分析应该主要看：

```text
copy_ponder_sharpness_grouped.csv
```

而不是只看单个全局 mean。

---

### 10.3 单长度 group 的 std 可能是 NaN

在 grouped comparison 里，`far_OOD_40` 和 `far_OOD_60` 各自只有一个 length。

因此这些组的：

```text
delta_objective_std
```

如果显示为 `NaN`，这是正常的，不是实验错误。

---

### 10.4 目前还不是 multi-seed 结论

当前正式对照是：

```text
NoPE: one good 100001-step checkpoint
RoPE: one 100001-step checkpoint
```

所以当前结论可以写作：

```text
preliminary evidence
case study
single-seed / single-checkpoint comparison
```

如果要更稳，需要后续补 NoPE/RoPE 多 seed。

---

## 11. 推荐最终报告图表

建议从当前结果中画四类图：

```text
1. base auto_answer_acc_mean vs length
2. base objective_mean vs length
3. delta_objective_mean at epsilon=0.01 vs length
4. grouped comparison: ID / near OOD / L40 / L60
```

其中第 3 和第 4 个最适合展示 sharpness 主结论。

---

## 12. 最短复现实验流程

从 `src/` 目录开始：

```bash
cd /data/diaozb/looped-tf-length-generalization/src
```

检查 sharpness 脚本语法：

```bash
python -m py_compile eval_ponder_sharpness.py
```

运行正式 NoPE/RoPE sharpness：

```bash
bash run_copy_ponder_sharpness_formal.sh
```

运行分组比较：

```bash
bash compare_copy_ponder_sharpness_groups.sh
```

查看最终分组结果：

```bash
cat ../eval/copy_ponder_sharpness_compare/copy_ponder_sharpness_grouped.csv
```

或用 pandas 查看：

```bash
python - <<'PY'
import pandas as pd

path = "../eval/copy_ponder_sharpness_compare/copy_ponder_sharpness_grouped.csv"
df = pd.read_csv(path)

print(df.to_string(index=False))
PY
```
