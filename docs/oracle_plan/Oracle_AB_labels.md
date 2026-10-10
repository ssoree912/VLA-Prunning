# Oracle 라벨과 시각 토큰 깊이 예측기

2026-10-07 · 기본 OpenVLA-7B (LIBERO-Spatial finetuned) · 구현: `oracle_pruning/VLA-Pruner/src/openvla`

전체 흐름:

```
dense teacher 의 action→시각 KV attention  s (32×256)          ← 라벨 (§2)
   ├─ Oracle A / B / B(깊이≥2): s 로 남길 토큰·깊이를 정해 closed-loop 평가 (§4, §5)
   ├─ 고정 스케줄 n_ℓ: 학습셋 s 에 DP 를 풀어 레이어별 토큰 수를 평균 (§6)
   └─ 예측기: layer 1 출력에서 토큰 순위 r̂ 를 예측 → 고정 스케줄로 깊이 결정 (§7–§9)
```

## 1. 표기

| 기호 | 의미 | 값 |
|---|---|---|
| \(L\) | LLM 레이어 수, \(\ell=0,\dots,L-1\) | 32 |
| \(M\) | 시각 토큰 수, \(i=1,\dots,M\) (시퀀스 위치 \(1..M\), `<BOS>` 바로 뒤) | 256 |
| \(H\) | attention head 수 | 32 |
| \(D\) | 행동 토큰 수 (7차원 행동 하나) | 7 |
| \(K\) | VLA-Pruner / Oracle A 의 프루닝 레이어 | 3 |
| \(k\) | 레이어 \(K\) 이후 남기는 시각 토큰 수, \(\lfloor (1-\texttt{fastv\_r})M\rfloor\) | 32 |
| \(m\) | 최소 깊이: 레이어 \(m\) 전에는 시각 토큰을 제거하지 않음 | 0 또는 2 |
| \(n_\ell\) | 레이어 \(\ell\) 에 입력되는 시각 토큰 수 | |
| \(d_i\) | 토큰 \(i\) 의 유지 깊이: 레이어 \(0,\dots,d_i-1\) 에만 입력 | \(\{m,\dots,L\}\) |

텍스트·특수 토큰과 행동 토큰은 항상 모든 레이어에 남는다.

## 2. 라벨: teacher 의 레이어별 KV attention

관측 이미지 \(x\), 지시문 \(c\) 에 대해 시각 토큰을 제거하지 않은 OpenVLA 로 행동 토큰 \(D\)개를 greedy 생성한다.
\(A_{\ell,h,r,i}\) 는 레이어 \(\ell\), 헤드 \(h\) 에서 **행동 토큰 \(a_r\) 을 예측하는 query** 가 시각 토큰 \(i\) 의 KV 에 준 attention 이다.

\[
A_{\ell,h,r,i}=\frac{\exp\!\left(Q_{\ell,h,r}^{\top}K_{\ell,h,i}/\sqrt{d_h}\right)}{\sum_{j\in\mathcal C_r}\exp\!\left(Q_{\ell,h,r}^{\top}K_{\ell,h,j}/\sqrt{d_h}\right)}
\]

| 예측 대상 | query 위치 | 문맥 \(\mathcal C_r\) |
|---|---|---|
| \(a_1\) | prefill 의 마지막 prompt 위치 (토큰 29871) | prompt 전체 (\(P\)) |
| \(a_r,\ r\ge2\) | 직전 생성 토큰 \(a_{r-1}\) (decode step, query 1개) | prompt + \(a_1..a_{r-1}\) |

softmax 는 전체 문맥 기준이며 시각 토큰 열만 잘라낸다 (이미지 안에서 재정규화하지 않음).

\[
\boxed{\,s_{\ell,i}=\frac{1}{D}\sum_{r=1}^{D}\frac{1}{H}\sum_{h=1}^{H}A_{\ell,h,r,i}\,}\qquad s\in\mathbb R^{32\times256}
\]

구현: `_action_vision_layer_scores` (`prismatic/extern/hf/modeling_prismatic.py`).
라벨은 \(s\) 하나이며, 깊이 제약 \(m\) · 예산 · 스케줄은 라벨이 아니라 \(s\) 를 쓰는 단계에서 적용한다.

## 3. 공통 토큰 예산

\[
B=KM+(L-K)k=3\cdot256+29\cdot32=\boxed{1696},\qquad
\rho_{\text{layer}}=\frac{\sum_\ell n_\ell}{LM}=\frac{1696}{8192}=20.703\%
\]

비교하는 모든 방법(VLA-Pruner, Oracle A/B, 고정 스케줄, 예측기)이 같은 \(\sum_\ell n_\ell=1696\) 을 쓴다. 에피소드마다 처음 3개 query 는 모든 방법이 시각 토큰 256개를 사용한다 (VLA-Pruner 의 warm start).

## 4. Oracle A — 레이어 \(K\) 에서 Top-\(k\) 고정

\[
S^A=\operatorname{TopK}_i\Big(\tfrac{1}{L-K}\textstyle\sum_{\ell=K}^{L-1}s_{\ell,i},\ k\Big),\qquad
n_\ell=\begin{cases}256,&\ell<3\\32,&\ell\ge3\end{cases}
\]

옵션 `--oracle_mode topk --oracle_layers post_k` (method `oracle_a`).

## 5. Oracle B — 토큰별 유지 깊이 (DP)

\[
\boxed{\;d^*=\arg\max_{d}\ \sum_{i=1}^{M}\sum_{\ell=0}^{d_i-1}s_{\ell,i}
\quad\text{s.t.}\quad\sum_i d_i=B,\ \ d_i\in\{m,\dots,L\}\;}
\qquad n_\ell=\sum_i\mathbf 1[\ell<d_i]
\]

남는 집합은 항상 \(S_{\ell+1}\subseteq S_\ell\) (제거된 토큰은 돌아오지 않음). 정확해는 동적 계획법으로 구한다.

\[
U_i(d)=\sum_{\ell<d}s_{\ell,i}\ (U_i(d)=-\infty\text{ for }d<m),\qquad
F(i,b)=\max_{0\le d\le\min(L,b)}\big[F(i-1,b-d)+U_i(d)\big]
\]

CPU 에서 query 당 약 70 ms. 구현: `oracle_depth_dp(scores, budget, min_depth)` (`oracle_llama.py`).

| 버전 | \(m\) | 옵션 / method | 평균 \(n_\ell\) (layer 0, 1, 2, …, 31) |
|---|---|---|---|
| Oracle B | 0 | `--oracle_mode depth` / `oracle_b` | 255, 179, 48, …, 29 |
| Oracle B, 깊이 ≥ 2 | 2 | `--oracle_min_depth 2` / `oracle_b_m2` | 256, 256, 45, …, 27 |

## 6. 고정 스케줄 (Oracle B-fixed) — DP 제거

Oracle B(\(m=2\)) 의 \(n_\ell\) 은 query 마다 거의 같다 (layer 2–31 표준편차 평균 2.0 토큰). 그래서 레이어별 토큰 수를 고정하고 토큰 순위만 정한다.

**스케줄 (학습셋 라벨에서 계산).** RLDS 학습셋 프레임 3,000개에 \(\operatorname{DP}(s,B,m{=}2)\) 를 풀고 \(n_\ell\) 을 평균한 뒤, 합이 정확히 \(B\) 이고 비증가가 되도록 반올림한다 (largest remainder 후 내림차순 정렬).

```
n_ℓ = [256, 256, 45×10 (ℓ=2..11), 44×4 (ℓ=12..15), 42, 39, 38, 37, 37, 36, 36, 35, 35, 35, 34, 32, 32, 32, 30, 28]
```

파일: `oracle_scripts/schedules/libero_spatial_m2.json` (`oracle_scripts/make_schedule.py`).

**순위와 깊이.**

\[
r_i=\sum_{\ell=m}^{L-1}s_{\ell,i}\quad(m=2),\qquad
\text{rank}_i=\text{토큰 } i \text{ 의 } r \text{ 내림차순 순위 }(0\text{부터}),\qquad
\boxed{\,d_i=\#\{\ell:\ n_\ell>\text{rank}_i\}\,}
\]

그러면 레이어 \(\ell\) 에는 정확히 상위 \(n_\ell\) 개 토큰이 남고 \(\sum_i d_i=\sum_\ell n_\ell=B\) 이다. 정렬 한 번이면 되므로 DP 가 필요 없다.

**DP 대비 손실 (평가 에피소드의 teacher 점수, query 1,500개).**

| 깊이 결정 | DP 최적값 대비 목적함수 (평균 / 하위 5%) |
|---|---|
| DP | 1.000 / 1.000 |
| 고정 스케줄 + \(r_i=\sum_{\ell\ge2}s_{\ell,i}\) 순위 | **0.984 / 0.977** |
| 고정 스케줄 + layer 16 점수 순위 | 0.952 / 0.924 |
| 고정 스케줄 + layer 2 점수 순위 | 0.837 / 0.701 |

옵션 `--oracle_mode schedule --oracle_schedule <json>` (method `oracle_bfix`, teacher 순위 사용 = 예측기의 상한).

## 7. 학습 데이터

### 7.1 라벨 수집 (`oracle_scripts/label_rlds.py`)

- 데이터: `openvla/modified_libero_rlds` / `libero_spatial_no_noops` (OpenVLA finetune 데이터). demo 432개, 프레임 52,970개, task 10개.
- 이미지 전처리를 평가 경로와 동일하게: \(\operatorname{CenterCrop}_{0.9}\big(\operatorname{Resize}^{\text{lanczos3}}_{224}(\operatorname{JPEG}^{-1}\!\circ\operatorname{JPEG}(x))\big)\).
- 프롬프트 `In: What action should the robot take to {instruction}?\nOut:` + 29871, greedy 7 토큰, §2 의 \(s\) 저장.
- 확인: teacher 행동과 demo 행동 차이 \(|a^{\text{teacher}}-a^{\text{demo}}|/(q_{99}-q_{01})\) 중앙값 0.001.
- 속도 2.9 frame/s (약 5시간), 저장 `/workspace/vla/data/oracle_labels/libero_spatial/epXXXX.npz` (`images`, `s_layer`, `action_ids`, `demo_action`, `instruction`).

### 7.2 예측기 입력 feature (`oracle_scripts/predictor_features.py`)

예측기는 **layer 1 출력 (layer 2 진입 직전)** 에 붙는다. 프레임마다 vision encoder + projector + LLM layer 0, 1 만 실행해 저장한다.

| 이름 | 크기 | 내용 |
|---|---|---|
| `hv` | \(256\times512\) | 시각 토큰 hidden state (layer 1 출력), 표준화 후 PCA 512 (분산 89.1%) |
| `hlast` | 4096 | 마지막 prompt 토큰 hidden state |
| `avis` | \(64\times256\) | 마지막 prompt 토큰 → 시각 토큰 attention, layer 0·1 × 32 head |
| `r` | 256 | 정답 \(r_i=\sum_{\ell\ge2}s_{\ell,i}\) |

입력 순서가 `<BOS>, 이미지, 지시문` 이고 attention 이 causal 이므로 **시각 토큰 hidden state 는 지시문을 볼 수 없다.** 지시문 정보는 `hlast` 와 `avis` 로만 들어온다.
확인: 부분 forward 의 layer 1 출력 = 전체 모델 `hidden_states[2]` (상대 오차 8e-6). 속도 62 frame/s (약 15분).

## 8. 예측기 (토큰 순위 학습)

### 8.1 구조 (`prismatic/extern/hf/oracle_predictor.py`, 1.80M 파라미터)

\[
z_i=\big[\,W_v\,hv_i\ ;\ W_t\,\operatorname{LN}(hlast)\ ;\ W_a\log(avis_i+10^{-6})\ ;\ p_i\,\big],\qquad
\hat r_i=\operatorname{MLP}(z_i)
\]

| 부분 | 설정 | 파라미터 |
|---|---|---|
| \(W_v\) | Linear 512→256 | 0.13M |
| \(W_t\) | LayerNorm + Linear 4096→256 | 1.06M |
| \(W_a\) | Linear 64→64 | 4K |
| \(p_i\) | 패치 위치 Embedding 256×64 | 16K |
| MLP (토큰 간 가중치 공유) | 640→512→512→1, GELU, dropout 0.1 | 0.59M |

### 8.2 손실과 학습 (`oracle_scripts/train_predictor.py`)

\[
\mathcal L=\operatorname{KL}\!\Big(\tfrac{r}{\sum_j r_j}\ \Big\|\ \operatorname{softmax}(\hat r)\Big)\quad(\text{프레임마다 256 토큰에 대한 분포})
\]

절대값이 아니라 **순위**를 학습한다. 추론 시 \(\hat r\) 로 정렬하고 §6 의 식으로 깊이를 정하며, 같은 순위로 다른 \(k\) 나 다른 스케줄에도 그대로 쓸 수 있다.

| 항목 | 값 |
|---|---|
| 분할 | demo 단위, task 별 10% 검증: 학습 389 demo / 47,828 프레임, 검증 43 demo / 5,142 프레임 |
| epoch / batch | 20 / 64 프레임 (epoch 당 747 step, 약 15–40초) |
| optimizer | AdamW, lr 1e-3 (OneCycle, warmup 5%), weight decay 0.01 |
| 선택 기준 | 검증 목적함수 비율 (예측 순위 깊이 / teacher 순위 깊이) 최고 epoch |
| 결과 | 검증 KL 0.0137, 목적함수 비율 0.9965 |

체크포인트: `/workspace/vla/data/oracle_predictor/libero_spatial_m2_tf1_s0/best.pt`

### 8.3 오프라인 검증 (`oracle_scripts/eval_predictor.py`)

teacher top-\(k\) 와 겹치는 비율, 목적함수 (teacher 순위 대비):

| 데이터 | 방법 | top16 | top32 | top45 | top64 | top128 | 목적함수 |
|---|---|---|---|---|---|---|---|
| 검증 demo (5,142) | **예측기** | 0.920 | 0.936 | 0.943 | 0.950 | 0.967 | 0.9965 |
| | layer 0–1 attention 만 | 0.489 | 0.566 | 0.617 | 0.672 | 0.821 | 0.8499 |
| | 학습 평균 (이미지 안 봄) | 0.397 | 0.458 | 0.509 | 0.573 | 0.702 | 0.7057 |
| closed-loop 상태 (30 에피소드, 776 query) | **예측기** | 0.914 | 0.934 | 0.941 | 0.947 | 0.965 | 0.9963 |
| | layer 0–1 attention 만 | 0.498 | 0.579 | 0.629 | 0.680 | 0.830 | 0.8571 |
| | 학습 평균 | 0.394 | 0.452 | 0.504 | 0.560 | 0.710 | 0.6992 |

DP 최적값 대비 (검증 demo): 예측기 0.981, teacher 순위 0.984.

## 9. 추론 (closed-loop, teacher 없음)

query 마다 forward **1회**:

1. layer 0, 1: 시각 토큰 256개 모두 계산하며 마지막 prompt 토큰의 시각 attention (layer 0·1) 을 모은다.
2. layer 2 진입 직전: 예측기로 \(\hat r\) 계산 → 정렬 → \(d_i=\#\{\ell:n_\ell>\text{rank}_i\}\).
3. 같은 prefill 을 이어서 레이어 \(\ell\) 진입 전에 \(\{i:d_i>\ell\}\) 만 남긴다 (원래 position id 유지, causal mask 재구성, 레이어별 KV 길이 \(=\) 텍스트 \(+\,n_\ell+\) 생성 토큰).
4. decode 는 그대로 (레이어별로 줄어든 KV 사용).

구현: `_oracle_depth_prefill` 의 `oracle_depth_fn` 훅 (`oracle_llama.py`), 옵션 `--oracle_mode predictor --oracle_predictor_ckpt <best.pt>` (method `predictor`).

smoke test (4/4 통과): 훅으로 고정 깊이를 넣으면 깊이 경로와 logit 동일 · 추론 중 예측기 입력 = 학습 feature (차이 0) · 레이어별 토큰 수 = 스케줄, 합 1696 · warm start 후 query 경로 정상.

### 연산량 (query 당, TFLOPs, 텍스트 29 + 시각 256 토큰)

| | vision encoder | LLM prefill | decode (6 step) | 합계 | full 대비 |
|---|---|---|---|---|---|
| full | 0.43 | 3.71 | 0.08 | 4.22 | 100% |
| VLA-Pruner | 0.43 | 1.07 | 0.08 | 1.57 | 37% |
| 예측기 (본 방법) | 0.43 | 1.07 | 0.08 | 1.57 | 37% |

토큰×레이어 합 2,624 로 VLA-Pruner 와 동일 (예측기 · 정렬 비용은 무시할 수준). batch 1 GPU 시간은 decode 의 가중치 로딩이 커서 FLOPs 만큼 줄지 않는다 (이전 측정: full 263 ms, K=3 228 ms).

## 10. 결과 (LIBERO-Spatial, init state 0–9 × 10 task, seed 7)

| 방법 | teacher | forward/query | 성공률 (100 ep) |
|---|---|---|---|
| full | – | 1 | 84% |
| VLA-Pruner | – | 1 | 70% |
| Oracle A | O | 2 | 69% |
| Oracle B | O | 2 | 76% |
| **Oracle B, 깊이 ≥ 2** | O | 2 | **82%** |
| Oracle B-fixed | O | 2 | 79% |
| **예측기** | **X** | **1** | **80%** |

짝지은 비교 (100 에피소드, 95% bootstrap CI):

| 비교 | 차이 | CI | 한쪽만 성공 |
|---|---|---|---|
| B(≥2) − VLA-Pruner | +12%p | [+4, +20] | 16 : 4 |
| B(≥2) − Oracle A | +13%p | [+5, +21] | 18 : 5 |
| B(≥2) − B | +6%p | [−1, +13] | 11 : 5 |
| B(≥2) − full | −2%p | [−9, +5] | 7 : 9 |
| B-fixed − B(≥2) | −3%p | [−11, +5] | 7 : 10 |
| B-fixed − VLA-Pruner | +9%p | [−1, +19] | 20 : 11 |
| **예측기 − VLA-Pruner** | **+10%p** | **[+1, +19]** | 18 : 8 |
| 예측기 − Oracle A | +11%p | [+2, +20] | 18 : 7 |
| 예측기 − B-fixed | +1%p | [−7, +9] | 11 : 10 |
| 예측기 − B(≥2) | −2%p | [−9, +5] | 6 : 8 |
| 예측기 − full | −4%p | [−12, +4] | 8 : 12 |

요약: 같은 연산량(FLOPs 37%)에서 예측기가 VLA-Pruner 보다 +10%p 높고 (CI 가 0 위), teacher 를 쓰는 oracle 들(B(≥2), B-fixed)과 차이가 없다. DP 를 고정 스케줄로 바꾼 손실(B-fixed − B(≥2) = −3%p)도 CI 안이다. LIBERO-Spatial 100 에피소드(표준 500 의 1/5) 결과이다.

## 11. 파일과 실행

| 무엇 | 위치 |
|---|---|
| 평가 로그 | `src/openvla/experiments/logs/oracle/` (`spatial10_status.txt`, `EVAL-…--spatial10_<method>.{txt,jsonl}`, `_diag/`) |
| 라벨 / feature / 체크포인트 | `/workspace/vla/data/oracle_labels/`, `oracle_features/`, `oracle_predictor/` |
| 스크립트 | `oracle_scripts/{label_rlds, make_schedule, predictor_features, train_predictor, eval_predictor, analyze}.py` |

```bash
# src/openvla 에서
python oracle_scripts/label_rlds.py --suite libero_spatial                 # 라벨 (GPU)
python oracle_scripts/make_schedule.py --suite libero_spatial --min_depth 2
python oracle_scripts/predictor_features.py --suite libero_spatial         # feature (GPU)
python oracle_scripts/train_predictor.py --suite libero_spatial --epochs 20
python oracle_scripts/eval_predictor.py --ckpt /workspace/vla/data/oracle_predictor/libero_spatial_m2_tf1_s0/best.pt
CKPT=checkpoints/openvla-7b-finetuned-libero-spatial SUITE=libero_spatial bash -c \
  'source oracle_scripts/common.sh; run_method 1 predictor spatial10 --num_trials_per_task 10 --save_video False'
python oracle_scripts/analyze.py --tag spatial10
```
