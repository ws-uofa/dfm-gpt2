# DFM for GPT-2 on WikiText-103

这是一个从实验仓库重新整理出的最小、独立实现。仓库只包含源代码、配置示例和
CPU 测试；不依赖原 `DeepFusionMem` 仓库，也不把模型、WikiText-103、FAISS
索引、memory 向量、checkpoint 或日志提交到 Git。

## 实现范围

固定支持以下闭环：

1. 最新已验证的 WikiText-103 建库协议：`cross_article` packing，1024-token
   block，16-token chunk，previous-chunk query，Top-16 key，并为每个 key 加入
   immediate continuation，最终共 32 memory slots；训练集检索排除当前 block。
2. GPT-2 traditional DFM：冻结 GPT-2，共享 memory projector，在层
   `[0,2,5,8,10,11]` 使用独立 memory attention 和 token-wise per-head gate。
3. GPT-2 transformer-only DFM：冻结 GPT-2，在相同六层加入独立的
   `causal-d256-l4-h8` continuous-prefix Transformer reader，并做 residual write。
4. 传统 CE loss，以及最新的 retrieved-vs-random margin loss：

   ```text
   L = NLL(retrieved) + weight * relu(margin + NLL(retrieved) - NLL(random))
   ```

刻意没有收录 Llama/Mistral、DDFM、NQ、旧消融、几十组 submission recipe 和
结果汇总脚本。这样核心实现保持在人可以完整阅读的规模。

## 目录

```text
dfm/
  config.py       两种架构和两种 loss 的显式配置
  data.py         prepared shards、datastore mmap、collator
  losses.py       next-token CE 与 margin 公式
  model.py        traditional / transformer-only GPT-2 DFM
  train.py        唯一训练程序
  evaluate.py     retrieved/random/off 对齐评估
scripts/
  build_wikitext103.py       分阶段建库程序
  build_random_negative.py   margin loss 的固定随机负样本
  build_latest_wikitext103.sh
  train_*.sh                 四个意图明确的训练入口
configs/paths.env.example    外部路径示例
tests/                       无网络、CPU 单元测试
```

## 外部数据与慢盘路径

以下内容是运行依赖，但不是仓库内容：

| 内容 | 当前已验证路径 | 用途 |
|---|---|---|
| GPT-2 small | `/plm-dhw/sunsiyuan/DEEPFUSIONMEM/models/MLPMemory/gpt2-small` | 冻结语言模型和 tokenizer |
| Qwen3-Embedding-0.6B | `/plm-dhw/sunsiyuan/DEEPFUSIONMEM/models/Qwen3-Embedding-0.6B` | key/query/continuation embedding |
| WikiText-103 raw-v1 | `/plm-dhw/sunsiyuan/DEEPFUSIONMEM/datasets/wikitext-103-raw-v1` | 原始语料 |
| 已完成 datastore | `/plm-dhw/sunsiyuan/DEEPFUSIONMEM/artifacts/wikitext103-db-ablation-v1/cross_article/datastore` | 7,482,592 个 train key/continuation pair 和 FAISS index |
| retrieved prepared data | `.../cross_article/prepared/exclude-block` | 116,915/245/280 个 train/validation/test block |
| margin random control | `.../cross_article/random-negative-seed42` | 与正样本逐行对齐、与 retrieved ID 不相交 |

省略号均指
`/plm-dhw/sunsiyuan/DEEPFUSIONMEM/artifacts/wikitext103-db-ablation-v1`。
这些路径可以迁移；代码只通过 CLI 参数或环境变量接收路径。禁止把慢盘上的大文件
复制进仓库。活动 checkpoint 建议写入
`/plm-shared/sunsiyuan/DEEPFUSIONMEM_FAST/runs/dfm-wikitext103`，确认有价值后再
同步到慢盘归档。

## 环境

当前机器已验证环境：

```bash
cd /plm-shared/sunsiyuan/dfm-wikitext103
source configs/paths.env.example
export PYTHONPATH=$PWD
/plm-shared/sunsiyuan/.venvs/dfm/bin/python -m pytest
```

若在其他机器安装，使用 `pip install -e '.[dev]'`。GPU 训练必须通过集群任务执行，
开发机只用于代码检查和 CPU 测试。

## 建库

完整建库是昂贵 GPU 数据任务。`build_wikitext103.py` 有四个可恢复阶段：

- `tokenize`：按行加 EOS 后连续拼接，生成 cross-article 1024-token blocks；
- `encode`：把 train stream 划分为相邻 16-token key/continuation pair 并编码；
- `index`：训练 IVF-PQ inner-product FAISS index；
- `prepare`：检索候选、训练集排除当前 block、保存 Top-16 prepared rows。

先用独立输出目录执行小规模 smoke，切勿覆盖已验证 artifact：

```bash
export WT103_ARTIFACT=/plm-shared/sunsiyuan/DEEPFUSIONMEM_FAST/smoke/dfm-wt103-build
$PYTHON_BIN scripts/build_wikitext103.py all \
  --dataset "$WIKITEXT103" --gpt2-tokenizer "$GPT2_MODEL" \
  --embedding-model "$EMBEDDING_MODEL" --output "$WT103_ARTIFACT" \
  --max-blocks 8 --nlist 32 --candidate-pool 64
```

正式运行可使用 `scripts/build_latest_wikitext103.sh`。它只负责容器内执行，不负责
申请 GPU；提交前应另外用平台的通用 ClusterX wrapper 做 dry-run。

margin loss 还需要一次性生成负样本：

```bash
$PYTHON_BIN scripts/build_random_negative.py \
  --positive "$WT103_ARTIFACT/prepared/exclude-block" \
  --datastore "$WT103_ARTIFACT/datastore" \
  --output "$WT103_ARTIFACT/random-negative-seed42" --seed 42
```

## 训练入口及意图

所有训练脚本接受额外 CLI 参数，例如 smoke 可追加 `--max-steps 20`。

| 脚本 | 意图 |
|---|---|
| `train_traditional_ce.sh` | 原始 gated memory-attention DFM + LM CE 基线 |
| `train_traditional_margin.sh` | 原始 DFM + retrieved-vs-random margin |
| `train_transformer_only_ce.sh` | 只有 Transformer reader 可训练的 CE 基线 |
| `train_transformer_only_margin.sh` | transformer-only + 最新 margin objective |

示例：

```bash
source configs/paths.env.example
export PYTHONPATH=$PWD
bash scripts/train_traditional_ce.sh --max-steps 20 --batch-size 1
```

默认学习率为 `1e-3`、weight decay 为 0、gradient clipping 为 1、seed 42。
GPT-2 所有参数都冻结并保持 eval mode；checkpoint 只保存新增 DFM 参数及
`run.json`，避免重复保存基础模型。

训练后使用同一批 test rows 做三条件评估；`--checkpoint` 指向具体的
`step-XXXXXXXX` 目录：

```bash
$PYTHON_BIN -m dfm.evaluate \
  --model "$GPT2_MODEL" --architecture traditional \
  --checkpoint "$DFM_RUNS/traditional-margin/step-00000020" \
  --prepared "$WT103_ARTIFACT/prepared/exclude-block" \
  --random-prepared "$WT103_ARTIFACT/random-negative-seed42" \
  --datastore "$WT103_ARTIFACT/datastore" \
  --output "$DFM_RUNS/traditional-margin/eval-test.json"
```

## 与历史实验的对应关系

- 建库协议对应 `wikitext103_database_construction_v1` 的推荐
  `cross_article/exclude-block` arm；历史完整测试 NLL 为 `2.935748`。
- traditional DFM 保留其六层 token-wise per-head gated memory attention 结构。
- transformer-only 保留最新 margin sweep 使用的
  `causal-d256-l4-h8-residual` 结构。
- 推荐 margin sweep 参数是 `(margin, weight)`：`(0.02,0.1)`、
  `(0.05,0.1)`、`(0.10,0.1)`、`(0.05,1.0)`；脚本默认 `(0.05,0.1)`。

本仓库重新实现了算法闭环，不声称新实现已经复现历史数值。正式结论前必须完成
full build receipt、冻结主干审计、同一 test rows 上的 retrieved/random/off 评估。
