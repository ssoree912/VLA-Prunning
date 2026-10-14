# 기본 OpenVLA 시각 토큰 프루닝: Phase 0 오라클 실험 계획

2026-10-02 · 오라클 검증 전용 수정본

## 1. 이번 실험의 질문

**현재 관측의 full-token action→vision attention 상위 `k`개 시각 토큰만 남기면, 같은 토큰 예산의 VLA-Pruner보다 폐루프 작업 성공률이 높아지는가?** 이것만 검증한다. 결과가 좋으면 이후에 이 점수를 예측하는 작은 선택기나 KD를 연구할 근거가 생긴다. 이번에는 선택기 학습, LoRA, RL, 패치별 제거 효과 측정, 궤적 예측 모델 학습을 하지 않는다.

여기서 *오라클*은 가능한 모든 토큰 집합 중 최적해가 아니다. 현재 관측을 **전체 시각 토큰으로 한 번 먼저 추론하여 얻은 action attention**을, 원래 프루닝 시점에는 알 수 없는데도 사용할 수 있는 정보 오라클이다. 실제 미래 이미지, 미래 성공 여부, 패치 제거의 인과 효과는 보지 않는다.

## 2. 모델과 고정 조건

| 항목 | 이번 실험의 설정 |
| --- | --- |
| 모델 | **기본 OpenVLA의 LIBERO fine-tuned checkpoint**. OpenVLA-OFT는 사용하지 않음 |
| 입력 | 현재 시각 이미지 `I_t`와 작업 지시문 `ℓ`. 원래 OpenVLA의 `predict_action(image, instruction)` 경로 사용 |
| 로봇 상태 | LIBERO 환경의 상태는 평가·기록에는 쓸 수 있지만, **기본 OpenVLA의 모델 입력에는 넣지 않음** |
| 행동 출력 | 한 번의 질의에서 7차원 행동 **하나**를 7개의 이산 action token으로 자기회귀 생성. 행동 하나를 실행한 다음 새 이미지를 관측하여 다시 질의 |
| 시각 토큰 | 기본 설정에서 `M ≈ 256` 예상. 실제 토큰 위치와 개수는 사용 checkpoint의 vision backbone에서 확인 |
| 프루닝 | VLA-Pruner와 동일한 Transformer layer `K = 3` 및 시각 토큰 예산. **오라클 선택은 현재 action 점수의 직접 TopK** |
| 예산 | 주 설정 `ρ = 0.125`, `k = ⌊ρM⌋` (`M = 256`이면 `k = 32`). `ρ = 0.25`는 보조 재현 확인 |
| 주 평가 | LIBERO-Long, 세 방법 모두 같은 task·초기 상태·seed 사용 |

**중요한 단위 구분:** 7개의 action token은 미래 7개 시점의 궤적이 아니라 **현재 한 행동 벡터의 7개 차원**이다. 따라서 이 실험은 미래 궤적 중요도를 직접 재는 실험이 아니라, 현재 행동 생성에 대한 시각 토큰 중요도의 정보 오라클을 시험한다. `t`는 행동 하나를 실행하고 다시 관측하는 제어 질의 번호다.

실행 전 checkpoint ID·revision, 이미지 crop/resize, action unnormalization key, gripper 변환, 실제 행동 차원 `D`, 시각 토큰 수 `M`, 위치 인덱스를 기록한다. 기본 OpenVLA의 공식 LIBERO 평가 코드는 매 환경 단계에서 이미지를 얻고 행동 하나를 실행한다.

## 3. 오라클 점수와 선택 집합의 수식

질의 `t`의 입력은 `x_t = (I_t, ℓ)`이다. 시각 토큰의 원래 위치 집합을 `V_t = {1,…,M}`이라 한다. `D`는 checkpoint가 출력하는 행동 차원이며 LIBERO 기본값은 7이다. 전체 토큰을 넣은 추론에서 `r`번째 action token을 **예측하는 query 위치**가 시각 토큰 `m`을 보는 attention을 `A_t^(l,h,r)[m]`이라 정의한다. `l`은 Transformer layer, `h`는 head다.

자기회귀 생성의 인덱싱에 주의한다. `r = 1`의 query는 마지막 prompt 위치이고, `r ≥ 2`의 query는 직전까지 생성된 action token에 해당한다. 생성된 `r`번째 token의 *자기 위치* attention을 `r`번째 token 예측의 attention으로 잘못 대응시키지 않는다. 실제 `generate()` 호출과 attention hook의 위치를 smoke test에서 대조한다.

후반부 layer 집합 `L_late`를 전체 LLM layer의 뒤쪽 절반으로 고정한다. 현재 구현 기준 0-based 인덱스 `L_late = {16,…,31}`이다. 주 **teacher 점수**는 다음 평균이다.

$$
S_t^{\mathrm{act}}(m)
= \frac{1}{|L_{\mathrm{late}}|\,N_h\,D}
  \sum_{l\in L_{\mathrm{late}}}\sum_{h=1}^{N_h}\sum_{r=1}^{D}
  A_t^{(l,h,r)}[m],\qquad m\in V_t.
$$

모델의 softmax 이후 attention 값을 집계한다. Layer·head·action 차원의 평균은 **사전에 고정**하고, 파일럿 결과에 따라 max 집계나 특정 layer를 골라 바꾸지 않는다. 이 점수는 action과 관련된 *attention proxy*이지 패치의 인과적 중요도나 행동 궤적 변화량이 아니다.

`TopK_k(S)`는 점수가 높은 시각 토큰 `k`개다. **오라클의 보존 집합은 아래 한 식으로 끝난다.** Semantic 점수, 후보 합집합, MMDP 다양성 필터는 오라클에 적용하지 않는다.

$$
R_t^{\mathrm{oracle}}
= \operatorname{TopK}_k\!\left(S_t^{\mathrm{act}}\right),
\qquad |R_t^{\mathrm{oracle}}|=k.
$$

동점은 원래 시각 토큰 인덱스가 작은 순서로 해소하여 항상 정확히 `k`개를 선택한다. 이 실험의 오라클 정책은 `R_t^oracle`만 남긴 **두 번째 추론의 행동**을 실행한다.

비교 대상 VLA-Pruner는 현재 full-token attention을 모르므로 이전 질의들의 action attention을 이용한다. 비교군은 공식 구현을 그대로 재현한다.

$$
\widehat S_t^{\mathrm{act}}(m)
= \frac{\sum_{i=1}^{w}\gamma^i S_{t-i}^{\mathrm{act,pruned}}(m)}
        {\sum_{i=1}^{w}\gamma^i},
\qquad w=3,\quad\gamma=0.8,
$$

비교군 내부에서는 과거 점수와 현재 semantic 점수의 후보 합집합을 MMDP로 거른다. **이 절차는 비교군에만 속하며 오라클의 토큰 선택에는 사용하지 않는다.** 과거 프루닝 실행에서 제거된 시각 토큰의 점수 처리, 시간 창과 warm start는 공식 OpenVLA 구현과 일치시킨다. 현재 확인된 코드에서 비교군의 과거 action 점수는 0-based **layer 15 하나**, 주 오라클의 현재 action 점수는 **layer 16–31 평균**이다. 따라서 주 비교의 목적은 같은 보존율에서 두 정책의 성공률 비교다. 개선 원인을 현재/과거 점수, layer, semantic/MMDP 중 하나로 분해하지 않는다.

## 4. 실제로 돌릴 세 방법

| 방법 | 현재 query의 시각 토큰 선택 | 행동을 만들기 위한 forward 수 |
| --- | --- | ---: |
| Full | 제거 없음 | 1 |
| VLA-Pruner | 공식 과거 action attention EMA + 현재 semantic 점수 + MMDP | 1 |
| **Oracle-Act-TopK** | **현재 full-token action attention 상위 `k`개를 직접 선택** | **2** |

Oracle-Act-TopK의 질의 `t` 실행 순서:

1. 동일한 `I_t, ℓ`로 **전체 토큰 추론**을 끝까지 수행한다. 나온 행동은 실행하지 않고 `S_t^act`만 얻는다.
2. `TopK_k(S_t^act)`로 정확히 `k`개의 원래 시각 토큰 인덱스를 정한다. **같은 `I_t, ℓ`**에 대해 `K = 3`에서 해당 토큰만 남기는 두 번째 추론을 수행한다.
3. 두 번째 추론의 행동 **하나**를 환경에 실행한다. 새 관측 `I_{t+1}`을 받고 1–3을 반복한다.

첫 `w = 3`개 질의는 비교가 맞도록 **세 방법 모두 full-token 행동을 실행**한다. 이 기간에 VLA-Pruner의 attention 기록을 채운다. 이후에만 Oracle-Act-TopK와 VLA-Pruner의 토큰 선택을 달리한다. 동일 초기 상태로 시작해도 첫 분기 이후에는 각 정책이 서로 다른 이미지 궤적을 볼 수 있다. 이것이 폐루프 정책 비교의 일부다.

**구현상 중요한 변경:** 현재처럼 `historical_attention` 자리에 `S_t^act`를 넣어 공식 `fastv_forward`의 `TopK∪TopK → MMDP` 경로를 그대로 타면 위 오라클이 아니다. 오라클 전용 경로에서 `TopK_k(S_t^act)`의 인덱스를 프루닝 위치에 직접 적용해야 한다. 동시 실행 중인 작업에 영향을 주지 않도록 설치된 `transformers`는 수정하지 않고, 별도 프로젝트 코드 경로에서 오라클 분기를 구현한다.

두 번 추론하는 오라클은 배포 가능한 가속기가 아니다. `Oracle-Act-TopK`의 측정 latency를 VLA-Pruner의 배포 latency와 비교하거나, 오라클이 빠르다고 주장하지 않는다.

## 5. 최소 검증과 평가 절차

### 5.1 Smoke test

한 task의 2–3 episode에서 다음 항목을 통과한 뒤 파일럿으로 간다.

1. Full 정책이 공식 기본 OpenVLA LIBERO 평가 경로의 행동과 일치한다. 이미지 전처리, 행동 역정규화, 그리퍼 부호를 확인한다.
2. `r = 1,…,D`의 attention 행이 **해당 action token을 예측한 query**에 맞게 수집된다. 이미 생성된 token의 attention과 한 칸 어긋나지 않았는지 로그로 확인한다.
3. 시각 토큰의 `M`개 인덱스가 prefill과 모든 decode step에서 일관되며, 프루닝 후에도 원래 patch 좌표에 대응한다.
4. `ρ = 1`이면 프루닝 코드가 Full과 같은 행동을 낸다. Eager attention 로깅과 기본 attention backend의 Full 행동 차이도 점검한다.
5. 실제 남은 토큰 수가 정확히 `k`이고, 삭제는 layer `K` 이후에만 적용된다.
6. 오라클의 선택 인덱스 집합이 계산한 `TopK_k(S_t^act)`와 정확히 같고, semantic 후보나 MMDP를 거치지 않았는지 확인한다.

전체 attention 행렬은 저장하지 않는다. 필요한 action→vision 행을 forward 중 집계하여 `M`차원 점수와 선택 인덱스만 저장한다.

### 5.2 폐루프 파일럿과 확인 평가

주 실험은 **LIBERO-Long, `ρ = 0.125`**로 시작한다. VLA-Pruner 논문에서 기본 OpenVLA의 이 설정은 Full 52.2%, VLA-Pruner 42.8%로 보고되어 있어 비교할 간격이 있다. 이 숫자는 기존 논문의 참고값이고, 본 실험에서는 **같은 checkpoint와 환경에서 Full을 다시 측정**한다.

| 단계 | 초기 상태 수 | 목적 |
| --- | ---: | --- |
| 파일럿 | task당 동일한 10개, 총 100 episode/방법 | 구현 이상, 성공률 방향과 분산 확인 |
| 확인 평가 | 파일럿의 0–9와 겹치지 않는 초기 상태 10–49, task당 40개, 총 400 episode/방법 | 사전 고정한 주 비교 보고 |

파일럿을 보고 점수 집계 방식, `K`, `ρ`, 주 suite를 바꾸지 않는다. 예산이 허용되면 확인 평가 뒤 `ρ = 0.25`를 보조 조건으로 실행한다. 각 방법의 task별 성공 횟수·시도 횟수, seed, checkpoint, suite 버전, episode 길이를 기록한다.

주 지표는 task를 동일 가중한 LIBERO-Long 성공률 차이다.

$$
\Delta_{\mathrm{oracle}}
= \operatorname{SR}(\mathrm{Oracle\!\text{-}\!Act\!\text{-}\!TopK})
 - \operatorname{SR}(\mathrm{VLA\!\text{-}\!Pruner}).
$$

같은 초기 상태별 성공/실패를 짝지어 보고하고, task 구조를 보존한 bootstrap 95% 신뢰구간을 제시한다. Full 대비 성능 손실도 함께 보고한다. 보조 진단은 각 query의 선택 토큰 수·위치, `S_t^act`의 top-k와 과거 EMA top-k의 중첩, 그리퍼 변화 전후 실패 사례다. 이 진단은 새 선택기를 학습하거나 실험 조건을 사후 변경하는 데 쓰지 않는다.

## 6. 결과 해석과 다음 결정

| 결과 | 해석 |
| --- | --- |
| Oracle-Act-TopK가 VLA-Pruner보다 일관되게 높고 신뢰구간도 0보다 위 | **현재 full-token action attention 상위 `k`개만으로 같은 보존율에서 더 높은 성공률을 냄.** 이후 이 점수를 저비용 선택기로 예측할 근거가 생김 |
| Oracle-Act-TopK가 VLA-Pruner와 비슷함 | 이 직접 TopK 오라클의 추가 성능 이득이 제한적임. attention KD로 곧바로 진행할 근거가 약함 |
| Oracle-Act-TopK가 더 낮음 | action attention만으로 고른 `k`개가 필요한 semantic·공간 정보를 충분히 보존하지 못했을 수 있음. 이것만으로 현재 action attention 자체가 무용하다고 결론 내리지 않음 |

성공률 차이가 나더라도 그것이 **미래 행동 궤적을 정확히 예측했다**거나 **현재/과거 attention 차이만으로 개선됐다**는 증거는 아니다. 기본 OpenVLA는 한 query에서 행동 하나를 생성하며, 오라클은 현재 행동 생성의 attention을 미리 본 것이다. 두 정책은 후보 선택 규칙도 다르다. 또한 이 오라클은 최적 토큰 집합의 수학적 상한이 아니다. 두 번째 full-token 추론이 추가되므로 같은 보존율 비교는 같은 실제 추론 시간 비교가 아니다. KD의 목적함수와 배포 속도는 오라클 가치가 확인된 다음 단계에서 별도로 설계·측정한다.

## 7. 구현·비교 기준 자료

- [OpenVLA의 `predict_action` 구현](https://github.com/openvla/openvla/blob/main/prismatic/models/vlas/openvla.py): 이미지·지시문 입력, 행동 차원만큼의 자기회귀 토큰 생성.
- [OpenVLA의 LIBERO 평가 코드](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py): 기본 모델은 proprio를 입력으로 받지 않고, 이미지마다 행동 하나를 질의·실행.
- [VLA-Pruner 논문](https://arxiv.org/html/2511.16449v5): action attention 평균, 과거 `w = 3` 질의의 EMA, `K = 3`, Combine-then-Filter와 기본 OpenVLA의 LIBERO 결과.
- [VLA-Pruner 공식 구현](https://github.com/MINT-SJTU/VLA-Pruner): **비교군**의 semantic 점수, MMDP, warm start 및 위치 처리 기준. 오라클의 직접 TopK 선택 규칙은 별도로 구현한다.
