# OFT depth predictor: 라벨, 입력 피처, 학습, 추론 수식 정리 (공식 OFT 기준, 2026-10-12)

코드: `oft_oracle/label_rlds_official.py` (라벨/피처), `oft_oracle/core.py` (DP, ranker, 예측기),
`oft_oracle/train.py` (학습, 스케줄), `oft_oracle/official.py` (공식 OFT forward에 이식한 추론).

## 0. 기호

| 기호 | 값 | 뜻 |
|---|---|---|
| $L$ | 32 | LLaMA-2 layer 수 |
| $H$ | 32 | attention head 수 |
| $D$ | 4096 | hidden 차원 |
| $M$ | 512 | visual 토큰 수 (agentview 256 + wrist 256) |
| $m$ | 2 | 예측기 위치 (layer $m$ 입력 = layer 1 출력), 모든 토큰의 최소 깊이 |
| $P$ | 512 | PCA 차원 |
| $T$ | 시퀀스 길이 | $T = N_{ids} + 570$ ($N_{ids}$ = BOS와 빈 토큰 29871을 포함한 prompt input_ids 길이) |

$h^{(l)}_p \in \mathbb{R}^D$: layer $l$ **입력**의 위치 $p$ hidden ($h^{(0)}$ = 임베딩, $h^{(m)}$ = layer $m-1$ 출력).

## 1. 토큰 배치 (공식 OFT, L1 regression + proprio)

```
[BOS] [visual 512] [proprio] [prompt text (N_ids - 1)] [action 56] [stop]
  0     1 .. 512     513      514 .. T-58              T-57..T-2    T-1
```

- visual: $V = \{1, \dots, 512\}$, visual 토큰 $i \in \{0..511\}$ 는 위치 $1+i$.
- **액션 쿼리** $Q = [T-58,\ T-2)$, $|Q| = 56$: 공식 액션 head가 읽는 위치.
  즉 마지막 prompt 토큰과 앞쪽 액션 토큰 55개예요. 위치 $p$ 의 hidden이 토큰 $p+1$ 을 예측해서, `finetune.py` 학습과 같아요.
  (이전 VLA-Pruner 경로 라벨은 액션 토큰 위치 $[T-57, T-1)$ 를 썼어요. 55개가 겹치고, 상위 20% 겹침은 99.6%였어요.)
- **지시문 쿼리** $I$: prompt `"In: What action should the robot take to {instr}?\nOut:"` 에서 지시문 단어 토큰들.
  $a = |\mathrm{tok}(\text{prefix})|$, $b = |\mathrm{tok}(\text{prefix} + \text{" "} + \text{instr})|$ 이면 $I = [514 + a - 1,\ 514 + b - 1)$.

## 2. Teacher forward와 attention

RLDS 학습 데모의 모든 프레임을 공식 `get_vla_action` 으로 forward해요.
입력은 두 이미지 (center crop), 정규화된 proprio예요. 모델은 full이고 프루닝은 하지 않아요.
라벨 추출 때만 모든 층에서 `output_attentions=True` 를 켜요 (eager, mask 없음 = 양방향).
이 attention은 SDPA 경로와 같은 계산이에요 (bf16 커널 차이만 있어요).

layer $l$, head $h$ 의 attention (양방향, causal mask 없음):

$$
A^{(l)}_{h}[q, k] = \mathrm{softmax}_k\!\left(\frac{(W^Q_h \tilde h^{(l)}_q)^\top\, \mathrm{RoPE}(W^K_h \tilde h^{(l)}_k)}{\sqrt{D/H}}\right),
\quad \tilde h = \mathrm{RMSNorm}(h)
$$

## 3. 라벨 (teacher)

**액션 라벨** (층별, visual 토큰별): 56개 액션 쿼리와 32 head에 대해 평균한 visual 토큰 $i$ 로의 attention

$$
s_{l,i} = \frac{1}{H\,|Q|} \sum_{h=1}^{H} \sum_{q \in Q} A^{(l)}_h[q,\ 1+i], \qquad l = 0..L-1,\ i = 0..M-1
$$

**지시문 라벨** (two-head용):

$$
s^{\text{instr}}_{l,i} = \frac{1}{H\,|I|} \sum_{h} \sum_{q \in I} A^{(l)}_h[q,\ 1+i]
$$

**순위 목표** (예측기 위치 이후 층의 합):

$$
r_i = \sum_{l=m}^{L-1} s_{l,i}
$$

저장: `s_layer` $(T_{ep}, L, M)$ f16, `s_instr` $(T_{ep}, L, M)$ f16, `r` $(T_{ep}, M)$ f32.

## 4. 예측기 입력 피처 (layer $m=2$ 입력에서, 같은 forward)

**(a) visual hidden `hv`** (토큰별): layer 1 출력의 visual 토큰 hidden을 표준화한 뒤 PCA로 줄여요

$$
x_i = h^{(m)}_{1+i} \in \mathbb{R}^{D}, \qquad
\mathrm{hv}_i = \left(\frac{x_i - \mu}{\sigma}\right)^{\!\top} C \in \mathbb{R}^{P}
$$

- $\mu, \sigma \in \mathbb{R}^D$: 차원별 평균과 표준편차 ($\sigma = \sqrt{\mathrm{diag}\,\Sigma} + 10^{-6}$).
- $C \in \mathbb{R}^{D \times P}$: 상관행렬 $\Sigma / (\sigma \sigma^\top)$ 의 상위 $P$ 개 고유벡터.
- 학습 데모 앞 30개 에피소드에서 3프레임마다 하나씩, 모든 visual 토큰 벡터로 계산해요 (`pca.npz`).

**(b) 액션 쿼리 요약 `hq`** (프레임당 하나):

$$
\mathrm{hq} = \frac{1}{|Q|} \sum_{q \in Q} h^{(m)}_q \in \mathbb{R}^{D}
$$

**(c) 앞 층 attention `avis`** (head별, 토큰별): layer 0, 1에서 액션 쿼리가 visual 토큰 $i$ 에 준 attention

$$
\mathrm{avis}_{(l,h),\,i} = \frac{1}{|Q|} \sum_{q \in Q} A^{(l)}_h[q,\ 1+i], \qquad l \in \{0, 1\},\ h = 1..32
\ \Rightarrow\ \mathrm{avis} \in \mathbb{R}^{2H \times M} = \mathbb{R}^{64 \times 512}
$$

(순서: layer 0의 head 0..31, 다음 layer 1의 head 0..31)

**(d) 위치 $i$**: 학습되는 임베딩 $E_{\text{pos}}[i] \in \mathbb{R}^{64}$ (카메라와 패치 위치).

참고: 학습 때는 저장된 피처를, 추론 때는 같은 forward 안에서 계산한 피처를 써요.
`hv`, `hq`, `avis` 모두 위 1–2절과 같은 위치에서 계산되고, 이식 검증에서 두 값이 같은 걸 확인했어요.

## 5. Ranker (DepthRanker, 파라미터 1.82M)

토큰 $i$ 마다 같은 MLP를 적용해요 (토큰끼리는 독립):

$$
\begin{aligned}
v_i &= W_v\, \mathrm{hv}_i & &\in \mathbb{R}^{256} \\
t &= W_t\, \mathrm{LN}(\mathrm{hq}) & &\in \mathbb{R}^{256} \\
a_i &= W_a \log(\mathrm{avis}_{:,i} + 10^{-6}) & &\in \mathbb{R}^{64} \\
z_i &= \mathrm{MLP}\big([v_i;\ t;\ a_i;\ E_{\text{pos}}[i]]\big) & &\in \mathbb{R}^{n_{\text{out}}}
\end{aligned}
$$

MLP: $640 \to 512 \to 512 \to n_{\text{out}}$, GELU, dropout 0.1. $n_{\text{out}} = 1$ (액션만) 또는 2 (two-head).

## 6. 학습 손실

**액션만 (spatial)**: 프레임마다 listwise KL. 순위만 배우고 budget과는 무관해요.

$$
\mathcal{L} = \mathrm{KL}\!\left(\frac{r}{\sum_j r_j}\ \Big\|\ \mathrm{softmax}(z)\right)
$$

**two-head (goal)**: 공유 trunk에서 head 두 개가 각자의 라벨을 배워요

$$
\mathcal{L} = \mathrm{KL}\!\left(\hat r^{\text{act}} \,\|\, \mathrm{softmax}(z^{\text{act}})\right)
+ \mathrm{KL}\!\left(\hat r^{\text{instr}} \,\|\, \mathrm{softmax}(z^{\text{instr}})\right),
\quad \hat r^{\text{instr}}_i \propto \sum_{l \ge m} s^{\text{instr}}_{l,i}
$$

추론 점수는 $p = (1-w)\,\mathrm{softmax}(z^{\text{act}}) + w\,\mathrm{softmax}(z^{\text{instr}})$ 이고, $w$ 는 `OFT_HEAD_WEIGHT` 로 정해요.

학습 설정: AdamW (lr 1e-3, wd 0.01, OneCycle), batch 64, 20 epoch. 검증은 태스크별 10% 데모예요.
체크포인트는 고정 스케줄에서 아래 7절 목적함수의 teacher 순위 대비 비율이 가장 높은 것으로 골라요.

## 7. Budget, oracle DP, 고정 스케줄

**budget** (VLA-Pruner 설정과 맞춘 token-layer 수):

$$
B = (K+1)\,M + (L-K-1)\,k, \qquad k = \mathrm{round}(M(1-r_{\text{fastv}}))
$$

20% 기준: $K=3$, $r_{\text{fastv}} = 0.9140625$ 이면 $k=44$, $B = 4 \cdot 512 + 28 \cdot 44 = 3280$ ($3280/16384 = 20.02\%$).

**oracle 깊이** (프레임별, 정확한 DP):

$$
d^* = \arg\max_{d} \sum_{i} \sum_{l < d_i} s_{l,i}
\quad \text{s.t.} \quad \sum_i d_i = B,\quad d_i \in \{m, \dots, L\}
$$

토큰 $i$ 는 layer $0..d_i-1$ 에서 남아 있고, 이때 받는 attention 합이 최대가 되게 정해요.

**고정 스케줄** (층별 남길 visual 토큰 수): 학습 프레임 2000개의 oracle 층별 개수 평균을 정수로 반올림해요 (최대 나머지 방식, 합은 $B$ 유지).

$$
n_l = \mathrm{round}\Big(\mathbb{E}_{\text{frames}}\ \#\{i : d^*_i > l\}\Big),\qquad n_0 = n_1 = M,\quad n_l \text{ 비증가},\quad \sum_l n_l = B
$$

spatial b3280: $n = [512, 512, 122, 121, 100, \dots, 39, 38]$.
two-head는 섞은 라벨 $s^{\text{mix}} = (1-\text{mix})\, s / \lVert s_{l \ge m} \rVert_1 + \text{mix}\ s^{\text{instr}} / \lVert s^{\text{instr}}_{l \ge m} \rVert_1$ 로 DP를 돌려요.

## 8. 추론 (공식 OFT forward, `official.py`)

1. layer $0, 1$: 모든 토큰을 써요. 이 두 층만 eager attention으로 돌려서 `avis` 행을 얻어요.
2. layer 2 입력에서 피처 (4절)를 계산하고 점수 $z$ (two-head면 $p$) 를 구해요.
3. 순위 $\rho_i$ 를 매기고 (내림차순, 동점은 작은 인덱스 우선) 깊이를 정해요:
   $$ d_i = \#\{l : n_l > \rho_i\} \quad\Leftrightarrow\quad \text{token } i \text{ is kept in layer } l \iff \rho_i < n_l $$
4. layer $l \ge 2$ 는 visual 토큰 $\{i : d_i > l\}$ (= 순위 상위 $n_l$ 개) 만 남겨요. 텍스트, proprio, 액션 토큰은 항상 남아요.
   구현: 예측기 직후 시퀀스를 [비시각 토큰; 시각 토큰 순위순] 으로 한 번 재배치해서, 각 층은 앞부분만 잘라 써요 (층별 GPU 동기화 없음, 남는 토큰과 결과는 같음).
   $n_l$ 이 비증가라서 남는 집합이 층마다 포함관계를 이뤄요 (nested). RoPE 위치는 원래 위치를 그대로 써요.
   SDPA를 mask 없이 써서 양방향이고, attention 행렬을 따로 만들지 않아요.
5. 마지막 hidden을 원래 길이로 되돌려 놓아요 (잘린 visual 자리는 0). 공식 `predict_action` 이 평소 인덱스 $[T-58, T-2)$ 로 액션 hidden을 잘라 액션 head에 넣어요.
6. 첫 쿼리부터 프루닝해요 (warm start 없음).

평균 visual retention은 $\frac{1}{L M}\sum_l n_l = B/(LM)$ 예요.
LLM FLOPs는 dense 대비 0.316이에요 (텍스트와 액션 토큰 포함).
