# dfm-gpt2

这是一个从实验仓库重新整理出的最小、独立实现。仓库只包含源代码、配置示例和
CPU 测试；不依赖原 `DeepFusionMem` 仓库，也不把模型、WikiText-103、FAISS
索引、memory 向量、checkpoint 或日志提交到 Git。

## 实现范围

支持以下闭环；下列数值都是当前已验证 setting 的默认值，不是代码限制：

1. 最新已验证的 WikiText-103 默认建库协议：`cross_article` packing，1024-token
   block，16-token chunk，previous-chunk query，Top-16 key，并为每个 key 加入
   immediate continuation，最终共 32 memory slots；训练集检索排除当前 block。
2. GPT-2 traditional DFM：冻结 GPT-2，共享 memory projector，在层
   `[0,2,5,8,10,11]` 使用独立 memory attention 和 token-wise per-head gate。
3. GPT-2 transformer-only DFM：冻结 GPT-2，在相同六层加入独立的
   `causal-d256-l2-h8` continuous-prefix Transformer reader；默认 post-attn
   query 和 residual write。该点是近期宽度/深度扫描的性能—参数折中默认值。
4. 传统 CE loss，以及最新的 retrieved-vs-random margin loss：

   ```text
   L = NLL(retrieved) + weight * relu(margin + NLL(retrieved) - NLL(random))
   ```

所有常用 ablation 都通过 CLI 参数表达。完整默认值也集中记录在
[`configs/defaults.json`](configs/defaults.json)，方便审阅和生成实验 manifest。

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
  universal_eval.py  固定 280-row retrieved/random/off 严格评估
scripts/
  build_wikitext103.py       分阶段建库程序
  build_random_negative.py   margin loss 的固定随机负样本
  build_latest_wikitext103.sh
  train_*.sh                 四个意图明确的训练入口
  train_recent_protocol.sh   近期单实验可参数化入口
  evaluate_universal_test.sh universal test 入口
configs/experiments/         近期实验协议与完整 sweep 轴
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
`/plm-shared/sunsiyuan/DEEPFUSIONMEM_FAST/runs/dfm-gpt2`，确认有价值后再
同步到慢盘归档。

## 环境

运行依赖写在 `requirements.txt`，测试依赖写在 `requirements-dev.txt`。
`pyproject.toml` 使用相同约束，避免 editable install 和 requirements 安装得到
不同环境。当前机器复用持久共享环境 `/plm-shared/sunsiyuan/.venvs/dfm`，不在
仓库内创建 `.venv`。

首次使用时创建机器本地配置：

```bash
cd /plm-shared/sunsiyuan/dfm-gpt2
cp configs/paths.env.example configs/local.env
# 按机器修改 configs/local.env；该文件已被 Git 忽略。
source scripts/activate_local_env.sh
bash scripts/setup_environment.sh
```

本机已经整理好 `configs/local.env`。环境脚本默认只做项目依赖导入检查、CPU
测试和 shell 静态检查，不修改共享环境。确实需要安装或更新依赖时显式执行：

```bash
PYTHON_BIN=/plm-shared/sunsiyuan/.venvs/dfm/bin/python \
  bash scripts/setup_environment.sh --install
```

`--pip-check` 会额外检查共享环境中的所有包。当前共享环境内与本项目无关的
`autofaiss 2.17.0` 存在缺少可选依赖及旧版 NumPy/PyArrow 约束冲突，因此该检查
会给出警告，但不影响 dfm-gpt2 的依赖导入和测试。

也可以用 `$PYTHON_BIN scripts/check_environment.py` 输出机器可读的版本、路径与
CUDA 可见性报告。当前开发机没有可见 GPU 属于正常现象；GPU 训练必须通过集群
任务执行。近期训练入口按 `NUM_GPUS` 启动本机 Accelerate worker，并拒绝可见卡数
不一致的容器。

若在其他机器创建全新环境，可在个人持久快盘上执行：

```bash
python3 -m venv /plm-shared/sunsiyuan/.venvs/dfm-gpt2
/plm-shared/sunsiyuan/.venvs/dfm-gpt2/bin/python -m pip install -r requirements-dev.txt
```

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

### 建库消融参数

| CLI 参数 | 默认值 | 消融含义 |
|---|---:|---|
| `--packing-mode` | `cross_article` | 可选 `article_only`（`article_partial` 为历史别名）；后者仅让 training blocks/datastore pair 不跨文章并保留末尾 partial，validation/test target 仍固定 cross-article |
| `--block-size` | `1024` | GPT-2 LM block 长度 |
| `--chunk-size` | `16` | query/key/continuation chunk 长度，必须整除 block size |
| `--top-k` | `16` | 每个 query 保留的近邻 key 数 |
| `--continuation` | `append` | `none`=仅 key，`append`=key+continuation，`only`=仅 continuation |
| `--exclude-current-block` | true | 同时检查 candidate key 与 continuation 所属 block |
| `--exclude-current-article` | false | 同时检查 candidate key 与 continuation 所属 article |
| `--candidate-pool` | `2048` | exclusion 前的 ANN 候选数 |
| `--index-type` | `ivf_pq` | 可选 `flat`、`ivf_flat`、`ivf_pq` |
| `--metric` | `inner_product` | 可选 `inner_product`、`l2` |
| `--nlist/--nprobe` | `10942/32` | IVF 参数 |
| `--pq-code-size/--pq-nbits` | `64/8` | PQ 参数 |
| `--embedding-max-length` | `64` | Qwen embedding tokenizer 上限 |
| `--prepared-name` | `exclude-block` | 同一 datastore 下不同 prepared view 的目录名 |

布尔参数使用 Python BooleanOptionalAction，例如
`--no-exclude-current-block --exclude-current-article`。每一组改变 tokenization、
packing、chunk 或 continuation 的实验应使用新的 `WT103_ARTIFACT` 目录；脚本遇到
已有 stage 会拒绝覆盖。只改变 Top-K 或 exclusion 时可以复用 datastore，但应使用
不同的 `--prepared-name` 单独执行 `prepare` stage。

示例：article-only、chunk 32、仅 key、Top-32：

```bash
$PYTHON_BIN scripts/build_wikitext103.py all \
  --dataset "$WIKITEXT103" --gpt2-tokenizer "$GPT2_MODEL" \
  --embedding-model "$EMBEDDING_MODEL" --output "$NEW_ARTIFACT" \
  --packing-mode article_only --block-size 1024 --chunk-size 32 \
  --top-k 32 --continuation none --prepared-name article-key-top32
```

margin loss 还需要一次性生成负样本：

```bash
$PYTHON_BIN scripts/build_random_negative.py \
  --positive "$WT103_ARTIFACT/prepared/exclude-block" \
  --datastore "$WT103_ARTIFACT/datastore" \
  --output "$WT103_ARTIFACT/random-negative-seed42" --seed 42
```

## 训练入口及意图

所有训练脚本接受额外 CLI 参数；追加值会覆盖默认值。例如 smoke 可追加
`--max-steps 20`。

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
GPT-2 所有参数都冻结并保持 eval mode；checkpoint 只保存新增 DFM 参数和对应
`dfm-config.json`，run 根目录保存 `run.json` 与 `training-audit.json`，避免重复
保存基础模型，同时绑定数据元信息、初始化、冻结主干和最终 checkpoint 哈希。

### 模型与训练消融参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--fusion-layers` | `0,2,5,8,10,11` | memory 插入层；层数由列表长度决定 |
| `--fusion-timing` | traditional=`pre_attn`，t-only=`post_attn` | 精确选择 base self-attention residual 前后的 memory query/write 时机 |
| `--memory-dim` | `1024` | datastore vector 维度，启动时与 meta 校验 |
| `--chunk-size/--top-k` | `16/16` | 必须与 prepared artifact 兼容；允许读取已存 Top-K 的严格前缀 |
| `--memory-value-mode` | `key_plus_continuation` | `key`、`continuation` 或二者交错 |
| `--projector-hidden` | `768` | traditional DFM 共享 projector 中间宽度 |
| `--memory-attention-heads` | `12` | traditional DFM memory attention heads |
| `--gate-type` | `token_wise_per_head` | `none`、静态 `per_head`、token-wise 或 concat token-wise |
| `--gate-init` | `0.0` | sigmoid 前 logit；默认 gate=0.5 |
| `--memory-attention-dropout` | `0.1` | traditional memory attention dropout，匹配 GPT-2 `resid_pdrop` |
| `--reader-dim/layers/heads` | `256/2/8` | transformer-only 默认 reader 规模 |
| `--reader-ff-multiplier` | `4` | reader FFN expansion |
| `--reader-topology` | `causal` | `causal` 或 `bidirectional` memory prefix attention |
| `--reader-write` | `residual` | `residual` 或 `replace`；无有效 memory 时均严格 no-op |
| `--reader-sharing` | `independent` | 六层独立 reader 或一个 `shared` reader |
| `--loss` | `ce` | `ce` 或 `margin` |
| `--margin/--margin-weight` | `0.05/0.1` | hinge margin 及其 loss 权重 |
| `--preserve-negative-rng` | true | random-memory forward 后恢复 RNG，不扰动后续 positive path |
| `--learning-rate` | `1e-3` | AdamW learning rate |
| `--weight-decay` | `0` | AdamW weight decay |
| `--warmup-steps` | `0` | linear scheduler warmup |
| `--max-grad-norm` | `1` | gradient clipping |
| `--batch-size/--gradient-accumulation` | `2/1` | 每进程 batch 与累积步数 |
| `--epochs/--max-steps` | `1/-1` | 训练预算；正 max-steps 会截断 epoch |

例如只在 GPT-2 最后两层融合，并扫更强 margin：

```bash
bash scripts/train_traditional_margin.sh \
  --fusion-layers 10,11 --margin 0.05 --margin-weight 1.0
```

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
- transformer-only 默认采用近期 post-attn sweep 推荐的独立
  `causal-d256-l2-h8-residual` 结构；d128/192/256/512、L1/2/4 和
  independent/shared 均可通过参数严格重建。
- 推荐 margin sweep 参数是 `(margin, weight)`：`(0.02,0.1)`、
  `(0.05,0.1)`、`(0.10,0.1)`、`(0.05,1.0)`；脚本默认 `(0.05,0.1)`。

本仓库重新实现了算法闭环，不声称新实现已经复现历史数值。正式结论前必须完成
full build receipt、冻结主干审计、同一 test rows 上的 retrieved/random/off 评估。

## 近期统一实验协议

近期实验的机器可读定义在
`configs/experiments/recent_wikitext103.json`，来源是原实验 worktree commit
`7e18203`。核心约束如下：

- 无论训练 datastore 使用 `cross_article` 还是 `article_partial`，正式测试始终使用
  相应 datastore 上生成的同一套 280-row cross-article test blocks；
- 每个 checkpoint 同表报告 retrieved、seed-42 deterministic disjoint real-datastore
  random 和 true memory-bypass off；
- 输出 token-weighted NLL/PPL/top-1、sample-equal paired delta、10,000 次 paired
  bootstrap 95% CI、random-ID audit/hash 和输入/checkpoint SHA256；
- one epoch 固定 7,308 optimizer steps，five epochs 固定 36,540 steps，global
  batch 16，seed 42，LR 1e-3 linear decay，无 warmup/weight decay；
- checkpoint 训练审计必须证明 frozen GPT-2 fingerprint 不变，且所有可训练 tensor
  至少获得一次 gradient 和 nonzero gradient。

默认近期训练：

```bash
source scripts/activate_local_env.sh
ARCHITECTURE=transformer_only RUN_NAME=t-only-d256-l2-post \
  bash scripts/train_recent_protocol.sh
```

pre/post-attn、shared reader、五 epoch 或 margin arm 只改显式变量：

```bash
ARCHITECTURE=traditional FUSION_TIMING=post_attn EPOCHS=5 MAX_STEPS=36540 \
  RUN_NAME=traditional-post-5e bash scripts/train_recent_protocol.sh

ARCHITECTURE=transformer_only READER_SHARING=shared READER_LAYERS=4 \
  RUN_NAME=t-only-d256-l4-shared bash scripts/train_recent_protocol.sh

ARCHITECTURE=transformer_only LOSS=margin MARGIN=.05 MARGIN_WEIGHT=.1 \
  RUN_NAME=t-only-margin-m05-w01 bash scripts/train_recent_protocol.sh
```

统一评测：

```bash
CHECKPOINT="$DFM_RUNS/t-only-d256-l2-post/step-00007308" \
EVAL_OUTPUT="$DFM_RUNS/t-only-d256-l2-post/universal-test" \
  bash scripts/evaluate_universal_test.sh
```

更详细的架构语义、矩阵和历史验收指标见
`docs/WIKITEXT103_RECENT_PROTOCOL.md`。
