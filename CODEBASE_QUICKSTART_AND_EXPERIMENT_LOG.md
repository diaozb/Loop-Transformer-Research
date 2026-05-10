# Codebase Quickstart and Experiment Log

## TL;DR

- 这是 ICLR 2025 论文 *Looped Transformers for Length Generalization* 的代码仓库，主线训练入口是 `src/train.py`，核心模型包装在 `src/models.py`。
- 当前仓库里最完整的原始结果证据在 `eval/` 下的 `json/csv/png`，`results_collection/` 里的 PDF 更像人工 memo/归档。
- 从已上传结果看，最强正面结果是：
  - `parity`：几乎完美的长度泛化，最佳 loop 基本严格对齐到输入长度。
  - `mod_add`：也非常强，长度 40 仍有 `0.9848` exact accuracy。
- 相对困难的任务是：
  - `copy`、`addition`、`sum_reverse`：训练外长度会明显掉。
  - `mod_add_digits`：几乎不外推。
- 后续附加实验里最值得写进投稿的是：
  - `copy` 对输入分布和位置编码非常敏感。
  - `parity` 的 random-loop / substep ablation 会破坏 “loop ~= length” 的对齐。
  - `parity_ponder` 有戏，`copy_ponder` 基本失败。

## 1. 重新上手时先记住什么

### 1.1 仓库结构

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `src/train.py` | 主训练入口 | 绝大多数基础任务都从这里进 |
| `src/models.py` | `GeneralTransformerModel` 包装 | 把 GPT-2 backbone 变成 looped transformer |
| `src/generate_training_data.py` | 任务数据生成 | parity / copy / addition / multi / sum_reverse / dict / mod_add / mod_add_digits |
| `src/test_func.py` | 训练期快速评测 helper | 不是统一 test runner，而是函数式评测 |
| `src/conf/*.yaml` | 任务配置 | 训练长度、测试长度、模型超参 |
| `src/eval/*.py` | dense eval / ablation 分析脚本 | 会把结果写到 `eval/<task>/<run>/` |
| `src/train_ponder.py` | PonderNet 风格训练 | 当前只看到 parity/copy 的结果 |
| `src/train_parity_random_loop.py` | parity 的 random loop supervision | ablation |
| `src/train_parity_step_supervision.py` | parity 的 substep / prefix supervision | ablation |
| `src/prob_parity.py` | parity hidden state probe | 分析 loop-depth / length 对齐 |
| `src/transformers/` | vendored Transformers | 本仓库对 GPT-2 做了定制修改 |
| `models/` | checkpoint 和 wandb 输出 | 目录层级和 eval 的 run id 基本对齐 |
| `eval/` | 已有实验结果 | 最重要的投稿素材来源 |
| `results_collection/` | PDF memo 归档 | 适合作为历史说明，不如 `eval/` 可复用 |

### 1.2 代码主线的 mental model

训练逻辑不是“输入一次 transformer，直接出答案”，而是：

1. 先生成一个离散序列任务样本。
2. 对大多数任务，把 token 转 one-hot，再送入 `GeneralTransformerModel`。
3. `looped_forward(xs, horizon)` 会重复执行同一个 transformer block stack，每一步都把原始输入重新注入。
4. 每个样本的 loss 不一定取最后一步，而是取“与任务计算深度对应”的那个 loop step。
5. curriculum 逐步拉长训练长度。

最关键的代码点：

- `src/models.py`
  - `looped_forward`: 每轮都做 `output = f(output + zs)`。
  - `forward_single`: 决定是否加 WPE，或者走 `forward_no_position`。
- `src/train.py`
  - 用 `batch_num` / `batch_num_1` 决定每个样本应该从哪个 loop step 取监督。
- `src/transformers/models/gpt2/modeling_gpt2.py`
  - 增加了 `forward_no_position`。
  - 在 attention 里加了 RoPE 分支。

### 1.3 一个很容易忘的坑

`train.py` 里调用数据生成器时，`max_num_digits=curriculum.n_points`，而生成函数内部是 `np.random.randint(min_num_digits, max_num_digits)`，右边界不包含。

这意味着：

- curriculum 的 `end=21`，实际最大训练长度是 `20`。
- curriculum 的 `end=20`，实际最大训练长度是 `19`。
- `mod_add_digits` 的 `end=5`，实际最大训练长度只有 `4`。

这个点在读结果时一定要一直记着，否则会把 train/OOD 边界看错一位。

## 2. 现在该从哪些文件开始看

建议阅读顺序：

1. `README.md`
2. `src/conf/parity.yaml` 和 `src/conf/copy.yaml`
3. `src/train.py`
4. `src/models.py`
5. `src/generate_training_data.py`
6. `src/eval/eval_parity_loops.py` 或 `src/eval/eval_copy_loops.py`
7. 再看 `src/train_ponder.py`、`src/prob_parity.py` 和 parity/copy 的 ablation 训练脚本

如果只是为了尽快恢复“怎么跑”：

```bash
conda activate ltf
cd src
python train.py --conf ./conf/parity.yaml
```

如果只是为了看现成结果：

- 先看 `eval/parity/`
- 再看 `eval/copy/`
- 然后看 `eval/parity_ponder/` 和 `eval/copy_ponder/`

## 3. 当前已有实验资产总览

### 3.1 有配置但没看到上传结果的任务

- `dict`
- `multi`（乘法）：配置在，README 也提到，但仓库里没有对应 `models/multi` 或 `eval/multi` 原始结果；只有 PDF memo。
- `reachability`：`models/reachability/` 下有 checkpoint，但当前 `src/` 没看到配套主线任务代码或评测结果，像是残留/旁支实验。

### 3.2 有原始结果的主线任务

| 任务 | 实际最大训练长度 | 主要结果目录 |
| --- | --- | --- |
| parity | 20 | `eval/parity/d9de8dd7-d283-4236-aa71-d02ce63ab40a/` |
| copy | 19 | `eval/copy/8fe7447f-9d64-4de4-bf04-29c50e860cd6/` |
| addition | 19 | `eval/addition/0ad72d9b-28ca-4db9-8b80-7968e2a3813a/dense_eval/` |
| sum_reverse | 19 | `eval/sum_reverse/90e27139-2c7d-4438-8da5-ede9bf789ea3/dense_eval/` |
| mod_add | 19 | `eval/mod_add/4471f21d-f6b4-4bd5-91f2-b648db3129a2/dense_eval/` |
| mod_add_digits | 4 | `eval/mod_add_digits/b8a5c387-c981-4e82-ac8a-9530ec222b70/dense_eval/` |

## 4. 主线结果总结

### 4.1 核心任务表现

下面的数字都来自 `eval/*/*/*_eval_results.json` 的“每个长度上取最优 loop 后的 exact accuracy”。

| 任务 | train max | best acc @20 | best acc @30 | best acc @40 | 最大长度仍 >=99% | 最大长度仍 >=95% | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| parity | 20 | `1.0000` | `1.0000` | `0.9998` | 40 | 40 | 最稳，几乎理想的 loop-length 对齐 |
| copy (`prob_one=0.5`) | 19 | `1.0000` | `0.9953` | `0.2010` | 30 | 32 | 训练内完美，OOD 断崖，且 exact-match 比 token acc 更脆弱 |
| addition | 19 | `1.0000` | `0.9888` | `0.1477` | 29 | 32 | OOD 明显掉，长度 40 几乎失败 |
| sum_reverse | 19 | `1.0000` | `0.9513` | `0.3888` | 27 | 30 | 比 copy/addition 稍好，但仍明显退化 |
| mod_add | 19 | `1.0000` | `0.9998` | `0.9848` | 38 | 40 | 第二强的正面结果，投稿里很适合当 positive control |
| mod_add_digits | 4 | `0.00025` @ len20 | N/A | N/A | 4 | 4 | 基本没有长度外推，说明十进制 digit interface 很难 |

### 4.2 loop 与长度的关系

- `parity`：最佳 loop 几乎严格等于输入长度，训练内外都成立。
- `sum_reverse`：最佳 loop 也基本等于长度。
- `mod_add`：同样基本是 `best_loop = length`。
- `copy`：训练外最佳 loop 通常更晚一些，长度 40 时最佳 loop 是 `39`。
- `addition`：训练外往往需要比长度更大的 loop，长度 40 时最佳 loop 是 `41`。

这说明：

- `parity` / `mod_add` 更像学到了“loop depth = algorithmic depth”的干净机制。
- `copy` / `addition` 更像学到了一个近似、但会在长程上退化的迭代过程。

### 4.3 推荐写进 paper 的主线叙事

最顺的主线应该是：

1. `parity` 展示几乎理想的 loop-depth 对齐和超长长度泛化。
2. `mod_add` 作为第二个正面例子，说明并不是只有 parity 有效。
3. `copy` / `addition` / `sum_reverse` 作为 harder tasks，展示 looped transformer 不是无条件成功，而是任务结构强相关。
4. `mod_add_digits` 作为“symbol interface matters”的反例。

## 5. 最重要的附加实验

### 5.1 Parity 的 positional / loop supervision ablation

| 变体 | OOD 平均最优 acc | best acc @40 | best loop @40 | 结论 |
| --- | --- | --- | --- | --- |
| baseline | `0.99994` | `0.99975` | 40 | 参考系 |
| rope | `0.99619` | `0.96400` | 40 | 单次 rope run 有些掉，但仍很强 |
| rope_best | `0.99946` | `0.99675` | 40 | 选 best 后接近 baseline |
| wpe | `0.99966` | `0.99525` | 40 | WPE 对 parity 也没造成明显问题 |
| wpe + fixed APE permutation | `0.99958` | `0.99750` | 40 | parity 对 APE permutation 很鲁棒 |
| wpe + resampled APE permutation | `0.99964` | `0.99800` | 40 | 同上 |
| random loop train 1..10 | `0.53079` | `0.51150` | 11 | 明显破坏 loop-length 对齐 |
| random loop train 2..20 | `0.75935` | `0.54975` | 16 | 比 1..10 好，但仍破坏 OOD 泛化 |
| substep supervision | `0.96968` | `0.84875` | 40 | 仍能工作，但不如标准训练 |

结论：

- `parity` 的好结果并不依赖某一种位置编码，`rope/wpe` 都能跑得很好。
- 真正关键的是监督方式：标准“在正确 loop step 上监督”最稳定。
- random-loop supervision 会让模型不再把正确 loop 深度绑定到输入长度。

### 5.2 Parity 的更长程泛化

额外结果目录：

- `eval/parity/.../out_of_distribution/`
- `eval/parity/.../test_head_influence/`
- `eval/prob/parity/`

可直接引用的数字：

| 长度 | best acc |
| --- | --- |
| 64 | `1.0000` |
| 128 | `0.9240` |
| 256 | `0.5480` |
| 512 | `0.6040` |

以及 `test_head_influence` 里：

| 长度 | best acc |
| --- | --- |
| 30 | `1.0000` |
| 40 | `1.0000` |
| 50 | `1.0000` |
| 60 | `0.99975` |
| 70 | `0.9980` |

如果投稿里要强调“learned iterative algorithm”的证据，这部分非常有价值。

### 5.3 Parity probe

`eval/prob/parity/` 里有三组 probe：

- `..._20_40_0`
  - 判断 hidden state 是否对应 `loop == length`
  - test overall accuracy `0.9006`
  - test overall AUC `0.9124`
- `..._20_40_-2`
  - 判断是否对应 `loop == length + 2`
  - test overall accuracy `0.1450`
  - test overall AUC `0.4168`
- `..._predict_loop`
  - 直接回归 loop index
  - test overall MAE `0.1816`

我会把它解释为：

- hidden states 明显编码了 “当前 loop 是否与所需计算深度对齐” 这个信号；
- 但对更远离正确 loop 的偏移（比如 `+2`）并没有同样强的线性可分结构。

### 5.4 Copy 的输入分布敏感性

同一个 checkpoint family，改 `prob_one` 后差异很大：

| 变体 | OOD 平均最优 acc | best acc @40 | best token acc @40 | 结论 |
| --- | --- | --- | --- | --- |
| `prob_one=0.0` | `1.0000` | `1.0000` | `1.0000` | 全零输入几乎是 trivial case |
| `prob_one=0.1` | `0.8999` | `0.5490` | `0.9829` | exact-match 降得比 token acc 快 |
| `prob_one=0.5` | `0.8364` | `0.2010` | `0.8367` | 最难，balanced input 让长程 copy 很脆弱 |
| `prob_one=0.8` | `0.8975` | `0.5142` | `0.9694` | 偏置分布比 0.5 好不少 |

结论：

- `copy` 对输入统计分布非常敏感。
- 很多 OOD 失败其实是“序列中某一位出错”，因为 token accuracy 仍明显高于 exact-match。

### 5.5 Copy 的位置编码对比

| 变体 | OOD 平均最优 acc | best acc @40 | best token acc @40 | 结论 |
| --- | --- | --- | --- | --- |
| baseline (`prob_one=0.5`) | `0.8364` | `0.2010` | `0.8367` | 参考系 |
| rope run 1 | `0.00004` | `0.0000` | `0.5139` | seq exact 几乎完全崩 |
| rope run 2 | `0.00939` | `0.0000` | `0.5667` | 同样很差 |
| wpe add once (first run) | `0.0711` | `0.0000` | `0.5490` | 方差很大，第一次 run 很差 |
| wpe add once (best run) | `0.9270` | `0.8080` | `0.9435` | 目前上传结果里最强 copy 方案 |
| wpe add all + fixed permutation | `0.00002` | `0.0000` | `0.3388` | 基本崩溃 |

结论：

- `copy` 对位置编码选择极其敏感。
- 目前看最值得保留的是 `wpe add once` 的 best run。
- `rope` 在 copy 上几乎完全失效，和 parity 的表现正好形成强对比。

### 5.6 Copy 的位置级准确率

结果在：

- `eval/copy/.../pos_acc_eval_0/`
- `eval/copy/.../pos_acc_eval_0.5/`

关键观察：

- `prob_one=0.0`
  - length 40 的最后一位，在 loop 3 就已经到 `>= 0.99`
  - 这基本说明全零 copy 过于简单，不适合作为主要证据
- `prob_one=0.5`
  - 最后一位在 loop 6 才超过 `0.5`
  - loop 8 才超过 `0.9`
  - loop 8 才接近 `0.99`

这个结果很适合支撑“信息逐位传播 / 每多几个 loop 才能把更靠后的位拷过去”的解释。

### 5.7 PonderNet 实验

| 变体 | 平均 auto-exit loops | 平均 auto-exit acc | 最长长度结果 | 结论 |
| --- | --- | --- | --- | --- |
| parity_ponder | `9.54` | `0.9969` | len40: acc `0.9920`, avg loops `14.62` | 成功，值得保留 |
| parity_ponder_mean_pool | `4.91` | `0.7142` | len40: acc `0.3990` | mean pooling 很差 |
| copy_ponder | `6.34` | `0.4750` | len40: seq acc `0.0`, token acc `0.3936` | 失败 |
| copy_ponder_no_reg | `1.00` | `0.4750` | len40: seq acc `0.0`, token acc `0.2097` | 更失败，基本立刻 halt |

结论：

- `parity` 的 adaptive halting 是成立的。
- `copy` 这边目前没有跑通，不适合写成正面主结论。
- 如果要保留 ponder 部分，建议只主推 parity，并把 copy 放到 appendix/negative results。

## 6. `results_collection/` 里的 PDF 应该怎么用

我能确认的 PDF 元信息如下：

| 文件 | 标题 | 页数 | 更像对应什么 |
| --- | --- | --- | --- |
| `results_collection/2026-02-04_loop_train_eval_multi_task_report.pdf` | `Loop Train & Eval 2026.2.1` | 6 | multi / multiplication 的历史报告 |
| `results_collection/2026-02-17_pondernet_experiment_memo.pdf` | `PonderNet` | 5 | `train_ponder.py` 与 `eval/parity_ponder`, `eval/copy_ponder` |
| `results_collection/2026-02-17_positional_encoding_experiment_memo.pdf` | `Positional Encoding` | 4 | parity/copy 的 `rope/wpe/APE permutation` |

注意：

- 这些 PDF 的文本层很弱，机器抽取不完整。
- 只要 `eval/` 里有 raw `json/csv`，我优先信 raw results。
- `multi` 目前只有 PDF，没有对应 raw checkpoint/eval，所以如果投稿要写 multi，最好补一轮可复现结果。

## 7. 我建议你投稿时优先使用的图

### 主文图候选

- `eval/parity/d9de8dd7-d283-4236-aa71-d02ce63ab40a/dense_eval/accuracy_heatmap.png`
- `eval/parity/d9de8dd7-d283-4236-aa71-d02ce63ab40a/out_of_distribution/accuracy_vs_loop.png`
- `eval/mod_add/4471f21d-f6b4-4bd5-91f2-b648db3129a2/dense_eval/accuracy_heatmap.png`
- `eval/copy/289ac6be-1382-47fe-ac9b-5d33ed9f9104/dense_eval_0.5_wpe_add_once_second_run_best_test/accuracy_heatmap.png`
- `eval/copy/8fe7447f-9d64-4de4-bf04-29c50e860cd6/pos_acc_eval_0.5/copy_position_accuracy_heatmap.png`
- `eval/parity_ponder/86494c70-cf32-48e0-a9cd-3e914bd41768/dense_eval_ponder/auto_exit_avg_loops_vs_length.png`
- `eval/parity_ponder/86494c70-cf32-48e0-a9cd-3e914bd41768/dense_eval_ponder/auto_exit_accuracy_vs_length.png`

### Appendix 图候选

- parity 的 `delta_l2_norm_mask_cycle_*_heatmap.png`
- copy 的 `token_accuracy_heatmap.png`
- parity probe 的 `train_accuracy_heatmap.png` / `test_accuracy_heatmap.png`
- 各种 `entropy_mask` / `answer_change_rate` heatmap

## 8. 现在如果你要继续做实验，我会怎么排优先级

1. 先把 `parity` baseline、`mod_add` baseline、`copy` best WPE run 的结论固定下来。
2. 如果投稿要包含 adaptive compute，就只保留 `parity_ponder`，把 `copy_ponder` 放 appendix 或 negative result。
3. 如果投稿要讲位置编码，就主讲：
   - parity 对位置编码鲁棒
   - copy 对位置编码高度敏感
4. 如果投稿要讲“loop 学到 algorithmic depth”，就一定要带上：
   - parity baseline
   - parity random-loop ablation
   - parity probe
   - copy position accuracy
5. 如果要保留 `multi`，建议补 raw rerun；当前只有 PDF，不够稳。

## 9. 当前仓库里我会额外提醒你的事

- 当前 worktree 是脏的，说明你最近还在改模型和 eval 脚本；正式复现实验前最好先整理一次分支。
- `src/eval/eval_copy_input_switch.py`、`src/eval/eval_parity_input_switch_after_first.py` 这些脚本在仓库里存在，但我没看到对应落盘结果目录；如果它们想进论文，需要补跑并保存。
- `dict` 和 `multi` 目前没有和主线一样完整的结果链。

