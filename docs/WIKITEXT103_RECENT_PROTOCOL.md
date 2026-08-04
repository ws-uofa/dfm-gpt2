# WikiText-103 recent experiment reproduction contract

This is the self-contained reproduction contract extracted from
`WIKITEXT103_EXPERIMENT_RESULTS.md` at source commit `7e18203`.

## Universal test protocol

Every formal result uses exactly 280 cross-article WikiText-103 test blocks and
the same three conditions: retrieved memory, deterministic seed-42 disjoint
real-datastore random memory, and true memory bypass. Training packing may be
`cross_article` or `article_partial`; it must not change the test target packing.

The evaluator rejects a run unless metadata says block 1024, chunk 16, Top-16,
cross-article targets, key plus continuation, and an audited disjoint random
control. It writes token-weighted NLL/PPL/top-1, sample-equal paired differences,
10,000-resample paired bootstrap intervals, sample win rates, random-ID audit and
hash, artifact hashes, and all 280 per-row metrics.

## Exact insertion semantics

For a frozen GPT-2 block input `h_l`:

```text
h_attn = h_l + SA(LN_1(h_l))
```

Transformer-only pre-attn computes `delta = Reader(memory, h_l)` and passes
`h_attn + delta` to the frozen MLP. Post-attn uses `h_attn` as the reader query
and passes the reader's residual result to the MLP. Invalid memory is an exact
zero update in either timing.

Traditional pre-attn memory attention runs in parallel with base self-attention
from the same `LN_1(h_l)` query. Traditional post-attn computes `h_attn` first,
applies a dedicated per-fusion-layer LayerNorm, then performs the memory read.
The post-attn LayerNorms add exactly 9,216 parameters.

Expected added parameter counts are strict architecture checks:

| architecture | added parameters |
|---|---:|
| traditional pre-attn | 15,609,416 |
| traditional post-attn | 15,618,632 |
| t-only d256/L2/H8 independent | 13,492,224 |
| t-only d256/L2/H8 shared | 2,248,704 |

## Default and ablations

The transformer-only default is causal, post-attn, residual, d256/L2/H8,
FF multiplier 4, dropout 0, and six independent readers at layers
`[0,2,5,8,10,11]`. This arm reached retrieved PPL 17.954598 and random PPL
18.2226 in the original one-epoch sweep; these are historical acceptance
references, not claims about a newly executed run.

The complete post-attn sweep crosses widths 128/192/256/512, depths 1/2/4, and
independent/shared readers. Head counts are 4/6/8/16 respectively so head width
stays 32. The absolute best historical arm was d512/L1 independent at retrieved
PPL 17.895126; the best shared arm was d256/L4 at 18.561897.

The timing comparison uses both traditional DFM and transformer-only d192/L2/H6
with pre/post-attn timing. The loss sweep changes only the objective relative to
the architecture's CE baseline. Historical margin pairs are `(m,w)` =
`(.02,.1)`, `(.05,.1)`, `(.10,.1)`, plus the strong-weight arm `(.05,1)` for
transformer-only; traditional also recorded `(.10,1)`.

## Training and acceptance

- frozen GPT-2 small;
- data: audited cross-article + exclude-current-block by default;
- global batch 16, seed 42, LR 1e-3, linear decay, no warmup, no weight decay;
- max grad norm 1;
- 7,308 steps for one epoch, 36,540 for five epochs;
- formal training and evaluation run on ClusterX GPUs;
- accept a run only after training audit, frozen-base fingerprint equality,
  gradient coverage, checkpoint/config binding, universal three-condition
  evaluation, and durable output inspection all pass.

The executable defaults and sweep axes live in
`configs/experiments/recent_wikitext103.json`.
