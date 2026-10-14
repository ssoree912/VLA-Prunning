# OFT 벤치마크 측정 방법 (2026-10-13)

LIBERO-Spatial 기준 비교표에 들어간 숫자를 어떻게 쟀는지 정리한 문서예요. 방법별 실행 설정, 성공률 프로토콜, 그리고 FLOPs / latency / 메모리 정의와 수식을 담았어요.

## 0. 결과표 (LIBERO-Spatial)

| 방법 | 성공률 (500ep) | LLM FLOPs | 추론 latency 중앙값 | speedup | p90 | 제어 주파수 | LLM ms (speedup) | 최대 GPU 메모리 |
|---|---|---|---|---|---|---|---|---|
| Full | 99.0% | 1.000 | 177.9 ms | 1.00× | 189.8 | 45 Hz | 96.5 (1.00×) | 14.98 GB |
| 우리 (two-head −avis) | 98.4% | 0.317 | 101.8 ms | 1.75× | 117.7 | 79 Hz | 42.6 (2.26×) | 14.60 GB |
| VLA-Pruner (20%) | 97.4% | 0.501 | 131.7 ms | 1.35× | 187.5 | 61 Hz | 54.6 (1.77×) | 16.68 GB |
| SpecPrune 동적 끔 | (spatial 미평가) | 0.366 | 149.4 ms | 1.19× | 164.9 | 54 Hz | 60.2 (1.60×) | 14.71 GB |
| SpecPrune 기본값 | 93.0% | 0.324 | 151.1 ms | 1.18× | 169.9 | 53 Hz | 56.8 (1.70×) | 14.63 GB |
| VLA-Cache | 98.2% | 0.793 | 194.7 ms | 0.91× | 219.6 | 41 Hz | 101.1 (0.95×) | 16.19 GB |

- 성공률은 500ep 평가에서 나왔어요(1, 2절).
- 나머지 열(FLOPs, latency, 메모리)은 따로 돌린 "깨끗한 측정"에서 나왔어요(3–5절).

## 1. 공통 설정

| 항목 | 값 |
|---|---|
| 모델 | OpenVLA-OFT LIBERO 체크포인트 (`moojink/openvla-7b-oft-finetuned-libero-*`), L1 회귀 액션 head, proprio 사용, 카메라 2개 (agentview + wrist), action chunk 8 |
| 시뮬레이터 | LIBERO, MuJoCo osmesa 렌더링, env 이미지 256×256, `num_steps_wait` 10 |
| 정밀도 / 배치 | bf16, batch 1, `torch.compile` 없음 |
| seed | 7 |
| GPU | NVIDIA RTX A6000 (49 GB) |
| 체크포인트 폴더 | 방법마다 별도 폴더. 가중치는 symlink, 폴더 이름은 방법마다 다르게 붙임. `get_vla`가 폴더 안의 modeling 파일을 덮어쓰고, HF가 원격 코드를 폴더 이름으로 캐시하기 때문이에요. |
| 실행 스크립트 | `scripts/eval.sh` (방법 × suite 한 번), `scripts/latency.sh` (깨끗한 측정) |

## 2. 방법별 실행 설정

| 방법 | 코드 / 스택 | 설정 | 최대 step (spatial / object / goal / 10) |
|---|---|---|---|
| Full | 공식 OFT (`baselines/openvla-oft-official`) + transformers fork `v4.40.1-openvla-oft` | 프루닝 없음 (`eval/run_oft.py`, 예측기 없음) | 220 / 280 / 300 / 520 |
| 우리 | 공식 OFT 같은 스택 + `gazeprune/oft.py` | two-head ranker (입력 hv, hq, 위치; avis 제외), 두 head 섞는 비율 $w=0.5$, 섞은 라벨 (mix 0.5)로 만든 20% 스케줄 b3280, 첫 쿼리부터 프루닝 | 220 / 280 / 300 / 520 |
| SpecPrune 기본값 | `SpecPrune-VLA/openvla-oft` (공개 코드) + 같은 fork, `eval/shard_eval.py` | 공개 기본값 (α = `STATIC_PRUNE_RATIO` 0.8, 동적 프루닝 켬) | **240 / 330 / 380 / 600** (공개 코드 값) |
| SpecPrune 동적 끔 | 같은 코드, `eval/specprune_flags.py` | 위 설정에서 `dynamic_prune=False`만 바꿈 (저자가 GitHub issue #1에서 권한 설정) | 240 / 330 / 380 / 600 |
| VLA-Pruner | `VLA-Pruner/src/openvla-oft` (자체 transformers 4.47 fork), `run_eval.py` | `fastv_k 3`, `fastv_r 0.9140625` (k=44, 프루닝된 쿼리당 3280 token-layer, 우리와 같은 예산), layer 15, `semantic_action`, warm start (공개 구현) | 220 / 280 / 300 / 520 |
| VLA-Cache | `vla-cache/src/openvla-oft` + transformers 브랜치 `vla-cache-openvla-oft` (4.47), `eval/shard_eval.py` | `use_vla_cache True`, 나머지 공개 기본값 | 220 / 280 / 300 / 520 |

로컬에서 바꾼 것은 두 가지예요. 둘 다 L1 회귀 경로의 동작은 바꾸지 않아요.
- 공식 OFT, VLA-Cache: `prismatic/models/action_heads.py`의 `diffusers` import를 선택적 import로 바꿨어요. 공용 환경의 peft 버전이 맞지 않아서이고, diffusion head 전용이에요.
- SpecPrune: α와 동적 비율을 환경변수로 받게 했어요 (`SPEC_ALPHA`, `SPEC_DYN_RATIO`). 기본값은 공개값 그대로예요.

## 3. 성공률

$$
\text{SR} = \frac{\#\{\text{성공 에피소드}\}}{\#\{\text{에피소드}\}},\qquad \text{에피소드} = \text{task 10개} \times \text{init state } 0..49 = 500
$$

- 에피소드 하나는 LIBERO 기본 init state에서 시작해요. 처음 10 step은 대기하고, 그 뒤 최대 step 안에 task를 성공하면 성공이에요.
- 일부 방법은 init 0–19 (200개)를 먼저 돌리고, init 20–49 (300개)를 이어서 돌려 합쳤어요 (`OFT_INIT_OFFSET=20`).
  - spatial의 Full, SpecPrune 기본값, 우리 방법이 이렇게 합친 결과예요.
  - 에피소드 단위 결과는 `*.jsonl`에 (task_id, init_state, success)로 저장돼 있어요.
- task 분할: GPU 하나에서 shard 3개 (task 0–3, 4–6, 7–9)로 나눠 돌렸어요. VLA-Pruner와 VLA-Cache는 메모리 때문에 shard 2개 (0–4, 5–9)예요.

## 4. FLOPs (LLM만, 2·MAC 기준)

공용 profiler (`eval/profiler.py`)가 LLaMA decoder layer마다 forward hook을 걸어요. 각 호출에서 실제 query 길이 $q_l$ 과 key/value 길이 $kv_l$ 을 읽어서 더해요. 숫자는 $d=4096$, $m=11008$ (SwiGLU), $L=32$, MHA예요.

$$
F_{\text{layer}}(q, kv) = \underbrace{8\,q\,d^2}_{Q,K,V,O\ \text{projection}} + \underbrace{4\,q\,kv\,d}_{QK^\top,\ AV} + \underbrace{6\,q\,d\,m}_{\text{gate, up, down}}
$$

$$
F_{\text{query}} = \sum_{l=0}^{L-1} F_{\text{layer}}(q_l, kv_l),\qquad
F_{\text{dense}} = L \cdot F_{\text{layer}}(T, T),\qquad
\text{FLOPs ratio} = \frac{1}{N}\sum_{\text{query}} \frac{F_{\text{query}}}{F_{\text{dense}}}
$$

- $T$ 는 그 쿼리의 전체 시퀀스 길이예요: BOS + visual 512 + proprio + 프롬프트 + 액션 56 + stop, 약 600.
- $F_{\text{dense}}$ 는 같은 프롬프트로 full 모델을 돌렸을 때의 값이에요. $T=605$ 이면 쿼리당 $8.03\times 10^{12}$ FLOPs예요.
- 행렬곱 $(a\times b)(b\times c)$ 을 $2abc$ FLOPs로 세요 (곱셈 1 + 덧셈 1 = 2 FLOPs). head 수는 $h\cdot d_h = d$ 라서 공식에 남지 않아요.
- 포함하지 않는 것: vision encoder / projector (모든 방법이 같음), lm_head, softmax 같은 원소별 연산 ($O(h q\, kv)$, 행렬곱 대비 $1/d_h$ 수준).
- OFT는 쿼리마다 forward 1번 (parallel decoding)이라 decode 단계가 없어요. 모든 방법에서 `n_decode = 0` 이에요.
- 프루닝 방법은 층마다 실제 $q_l$ 이 줄어서 자동으로 반영돼요.
  - 우리: $q_l = (T-512) + n_l$
  - SpecPrune: 층별 남은 토큰 수
  - VLA-Pruner: warm start 쿼리는 dense
  - VLA-Cache: 재사용으로 줄어든 $q$ 와 전체 $kv$
- 표의 FLOPs는 깨끗한 측정 실행의 쿼리 평균이에요. suite별 500ep 평가 로그에서 계산한 값도 거의 같아요.

참고로 visual retention 정의는 이렇게 써요.

$$
\text{retention} = \frac{1}{L\cdot 512}\sum_{l=0}^{L-1} \big(q_l - (T - 512)\big)
$$

layer 0–1의 전체 512개도 포함해요. 우리 20% 스케줄은 $\sum_l n_l = 3280$, $3280/16384 = 20.0\%$ 예요.

## 5. Latency, speedup, 메모리

### 5.1 측정 프로토콜 (깨끗한 측정)
- GPU 0에서 **한 번에 한 방법만** 돌렸어요 (shard 1개).
  - 같은 시간에 GPU 1, 3에서는 라벨 추출이 돌아 CPU는 나눠 썼어요.
  - 그래서 절대값은 조금 높을 수 있지만 모든 방법이 같은 조건이에요.
- 입력: LIBERO-Spatial **실제 rollout**, task 0과 5 × init 0–2 = 에피소드 6개 (방법당 쿼리 67–96개).
  - 연속 쿼리라서 이전 쿼리 상태를 쓰는 방법 (SpecPrune의 이전 step 토큰 재사용, VLA-Cache의 캐시, VLA-Pruner의 attention history와 warm start)도 평가 때와 똑같이 동작해요.
- 통계: 에피소드 전체 쿼리를 다 포함해서 중앙값과 p90을 냈어요. 첫 쿼리나 warm start 쿼리도 빼지 않았어요.

### 5.2 세 가지 시간

| 이름 | 범위 | 재는 법 |
|---|---|---|
| **추론 latency** (`t_action_ms`) | 정책 쿼리 1회: 관측 (이미지 2장 + proprio + 지시문) → 액션 chunk 8개. 이미지 전처리 (resize, center crop), 토크나이즈, vision encoder, LLM, 액션 head, 역정규화, **방법별 추가 계산** 포함. 시뮬레이터 step과 로그 저장은 제외 | `get_action` 앞뒤에 `torch.cuda.synchronize()`를 두고 wall-clock (`time.perf_counter`) |
| model ms (`t_model_ms`) | `vla.predict_action` (vision + LLM + 액션 head) | 위와 같은 방식 |
| LLM ms (`t_llm_prefill_ms`) | LLaMA layer 0 입력부터 layer 31 출력까지 | CUDA event: layer 0 forward pre-hook에서 record, 마지막 layer forward hook에서 record, 둘 사이 `elapsed_time` |

### 5.3 지표 수식

$$
\text{latency} = \operatorname{median}_{\text{query}}\ t_{\text{action}},\qquad
\text{p90} = \operatorname{percentile}_{90}\ t_{\text{action}}
$$

$$
\text{speedup}_{\text{method}} = \frac{\operatorname{median}\ t_{\text{action}}^{\text{Full}}}{\operatorname{median}\ t_{\text{action}}^{\text{method}}},\qquad
\text{LLM speedup} = \frac{\operatorname{median}\ t_{\text{LLM}}^{\text{Full}}}{\operatorname{median}\ t_{\text{LLM}}^{\text{method}}}
$$

$$
\text{제어 주파수 [Hz]} = \frac{\text{chunk 길이 } 8}{\text{latency [s]}} = \frac{8000}{\text{latency [ms]}}
$$

$$
\text{최대 GPU 메모리} = \max_{\text{query}}\ \texttt{torch.cuda.max\_memory\_allocated}()
$$

- 메모리는 쿼리 시작 때 `reset_peak_memory_stats()`로 초기화해요. 프로세스 단위 PyTorch 할당량이에요 (CUDA context나 캐시 여유분은 제외).

### 5.4 숫자 해석 메모
- **FLOPs와 LLM speedup의 차이:** 우리는 FLOPs가 3.2배 줄었는데 LLM은 2.26배 빨라졌어요.
  - batch 1 행렬곱은 토큰 수가 적으면 가중치를 읽는 시간이 바닥값이 돼요 (memory-bound).
  - A6000의 경계 (연산 속도 / 대역폭)는 대략 $155\ \text{TFLOPS} / 768\ \text{GB/s} \approx 200$ 토큰이에요.
  - 우리 layer 2–31의 토큰 수 (약 130–215)는 이 근처나 아래라서, 토큰을 줄인 만큼 그대로 빨라지지는 않아요.
- **SpecPrune:** LLM 1.6–1.7배에 비해 추론 latency는 1.2배예요. 모델 밖에서 프레임 간 패치 유사도를 계산하는 비용 (쿼리당 약 50 ms)이 `get_action` 안에 들어가기 때문이에요.
- **VLA-Pruner:**
  - 중앙값 1.35배에 비해 p90은 Full 수준 (187.5 ms)이에요. 에피소드 앞쪽 warm start 쿼리를 full로 돌리기 때문이에요.
  - 메모리 +1.7 GB는 32개 층 attention을 다 만들어서 다음 쿼리까지 들고 있는 공개 구현 때문이에요. 쿼리 사이에 남는 메모리가 14.65–16.40 GB로 오르내려요.
- **VLA-Cache:** FLOPs는 0.79로 줄지만, 캐시 처리 비용 때문에 latency는 Full보다 길어요.

## 6. 결과 파일 위치

| 항목 | 경로 |
|---|---|
| 깨끗한 측정 (latency / FLOPs / 메모리) | `eval_logs/oft_latency/libero_spatial/<method>_libero_spatial_lat.prof.jsonl` |
| 500ep 성공률 | `eval_logs/oft_spatial200/` (Full: `oft200_offic_full*`, 우리: `oft200_offnew_twohead_noavis_w0.5_b3280*`, SpecPrune 기본값: `oft200_specprune*`), `eval_logs/oft_bench500/libero_spatial/` (VLA-Pruner, VLA-Cache) |
| profiler 한 줄 형식 | `{q, flops, flops_dense, ratio, prefill_q[32], t_action_ms, t_model_ms, t_llm_prefill_ms, mem_peak_mb, mem_alloc_mb, ...}` (쿼리 1개) |
