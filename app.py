import streamlit as st
import numpy as np
import pandas as pd
import time

st.set_page_config(layout="wide", page_title="BCI 파킨슨 약물별 보정 시뮬레이터")

# --------------------------------------------------------
# [세션 상태(메모리) 초기화]
# --------------------------------------------------------
history_len = 100
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(
        [[75.0, 75.0]] * history_len,
        columns=["🎯 운동 의도 (목표선)", "🦾 실제 근육 운동량 (결과선)"]
    )
if 't_hours' not in st.session_state:
    st.session_state.t_hours = 0.0
if 'current_disease_years' not in st.session_state:
    st.session_state.current_disease_years = 5.0
if 'prev_slider' not in st.session_state:
    st.session_state.prev_slider = 5.0

# --------------------------------------------------------
# [좌측 제어판] 환자 설정 및 4대 약물 군 컨트롤
# --------------------------------------------------------
st.sidebar.header("⚙️ 모의 환자 통제 제어판")

# 과부하 방지용 재생/일시정지 토글
is_running = st.sidebar.toggle("▶️ 시뮬레이션 실시간 재생", value=True)

st.sidebar.markdown("---")

intent_base = st.sidebar.slider("🎯 기준 운동 의도 (%)", min_value=10, max_value=90, value=75, step=1)
disease_slider = st.sidebar.slider("⏳ 발병 기간 설정 및 시작점 (년차)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

# 슬라이더 조작 시 유병기간 업데이트
if st.session_state.prev_slider != disease_slider:
    st.session_state.current_disease_years = disease_slider
    st.session_state.prev_slider = disease_slider

st.sidebar.markdown("---")
st.sidebar.subheader("💊 의학적 약물 투여 설정")

med_choice = st.sidebar.selectbox(
    "투약할 약물 분류 선택",
    [
        "선택 안함 (약효 없음)",
        "1. 레보도파 계열 (도파민 직접 보충)",
        "2. 보조요법제제 (도파민 수용체 작용)",
        "3. 도파민 분해 억제제 (MAOB / COMT 억제)",
        "4. 항콜린제 (도파민 부족으로 인한 떨림 조절)"
    ]
)

if st.sidebar.button("💊 선택한 약물 투여 (투약시간 초기화)", use_container_width=True):
    st.session_state.t_hours = 0.0

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 BCI 알고리즘")
bci_on = st.sidebar.toggle("BCI 활성화 함수 기반 자동 보정 ON", value=False)

# --------------------------------------------------------
# [내부 타이머 및 유병기간 연산] (재생 중일 때만 시간 증가)
# --------------------------------------------------------
if is_running:
    st.session_state.t_hours += 0.05
    st.session_state.current_disease_years += 0.003

current_t = st.session_state.t_hours
current_years = st.session_state.current_disease_years

# 기저 상태 연산 (파킨슨 진행에 따른 자연 효율 감소 및 떨림 증가)
base_efficiency = max(0.2, 1.0 - (current_years * 0.07))
base_tremor = current_years * 3.5

# 약물별 수치 연산 (약동학적 감쇄 모델 적용)
eff_boost = 0.0
tremor_reduction = 0.0
med_info = ""

if med_choice == "1. 레보도파 계열 (도파민 직접 보충)":
    eff_boost = 0.6 * np.exp(-0.35 * current_t)
    tremor_reduction = 0.7 * np.exp(-0.35 * current_t)
    med_info = "🧪 운동 능력 대폭 향상되나 빠른 약효 소실"
elif med_choice == "2. 보조요법제제 (도파민 수용체 작용)":
    eff_boost = 0.35 * np.exp(-0.15 * current_t)
    tremor_reduction = 0.45 * np.exp(-0.15 * current_t)
    med_info = "🧬 남아있는 수용체를 자극하여 완만한 효능 유지"
elif med_choice == "3. 도파민 분해 억제제 (MAOB / COMT 억제)":
    eff_boost = 0.25 * np.exp(-0.08 * current_t)
    tremor_reduction = 0.3 * np.exp(-0.08 * current_t)
    med_info = "🛑 도파민 파괴를 막아 잔류 시간 지속 증가"
elif med_choice == "4. 항콜린제 (도파민 부족으로 인한 떨림 조절)":
    eff_boost = 0.1 * np.exp(-0.2 * current_t)
    tremor_reduction = 0.9 * np.exp(-0.2 * current_t)
    med_info = "🌿 '진전(떨림)' 증상을 억제하여 집중 치료"
else:
    med_info = "⚠️ 어떠한 의학적 치료제도 개입하지 않은 상태"

# 현재 상태 계산
current_efficiency = np.clip(base_efficiency + eff_boost, 0.1, 0.95)
current_tremor = max(0.5, base_tremor * (1.0 - tremor_reduction))

# 파킨슨병 특이적 베타파 세기 연산
base_beta = 45.0 + (current_years * 4.5)
beta_suppression = eff_boost * 55.0
current_beta = max(12.0, base_beta - beta_suppression)

# =====================================================================
# 🧠 [핵심 평가 요소] 비선형 활성화 함수(Activation Function)를 이용한 BCI 신호 보정
# =====================================================================
if bci_on:
    # 1. Sigmoid 활성화 함수 정의: 신호 전달 효율의 비선형적 부스트 모델링
    def sigmoid_activation(x):
        return 1 / (1 + np.exp(-x))
    
    # 2. 오차 신호 도출: 정상 상태(효율 1.0, 떨림 최소)와의 격차 계산
    efficiency_deficit = 1.0 - current_efficiency
    tremor_error_signal = current_tremor
    
    # 3. 활성화 함수 적용 
    # - 운동 효율: Sigmoid를 통과시켜 부족한 효율을 S자 곡선으로 부드럽게 증폭
    # - 떨림 억제: np.tanh (Hyperbolic Tangent)를 사용하여 비정상적인 큰 진폭을 -1 ~ 1 사이로 압축 필터링
    bci_efficiency_boost = sigmoid_activation(efficiency_deficit * 4.0) * 0.6
    bci_tremor_suppression_ratio = np.tanh(tremor_error_signal * 0.2) * 0.9 
    
    # 4. 최종 뇌파 및 운동 출력 보정
    current_efficiency = min(0.98, current_efficiency + bci_efficiency_boost)
    current_tremor = current_tremor * (1.0 - bci_tremor_suppression_ratio)
    
    # BCI 알고리즘에 의한 병적 동기화 베타파 강제 감쇄 (비선형 감쇄 적용)
    current_beta = current_beta * sigmoid_activation(-current_beta * 0.1) 
# =====================================================================

current_beta = np.clip(current_beta + np.random.normal(0, 0.8), 0.0, 100.0)

# --------------------------------------------------------
# [화면 출력부]
# --------------------------------------------------------
st.title("🧠 뇌 운동 신호 vs 🦾 실제 근육 운동량 실시간 모니터")

if not is_running:
    st.warning("⏸️ 현재 시뮬레이션이 일시정지 상태입니다. 좌측 상단의 '실시간 재생' 토글을 켜주세요.")

if bci_on:
    actual_color = "#32CD32" # BCI 활성화 시 안정적인 초록색 파형
    msg = f"🤖 BCI 보정 ON (Sigmoid/Tanh 활성화 함수 필터링 중) | ⏳ 유병기간: {current_years:.2f}년차 | ⏱️ 투약 후: {current_t:.1f}시간째"
    st.success(msg)
else:
    if current_efficiency < 0.5 or current_tremor > 10.0:
        actual_color = "#FF4B4B" # 위험 상태 빨간색 파형
        warn_text = med_info if med_choice != "선택 안함 (약효 없음)" else "약물 미복용으로 인한 운동 장애"
        msg = f"⚠️ 증상 악화 | ⏳ 유병기간: {current_years:.2f}년차 | ⏱️ 투약 후: {current_t:.1f}시간째 | {warn_text}"
        st.error(msg)
    else:
        actual_color = "#FFA500" # 약효 반응 주황색 파형
        msg = f"💊 약효 반응 상태 | ⏳ 유병기간: {current_years:.2f}년차 | ⏱️ 투약 후: {current_t:.1f}시간째 | {med_info}"
        st.info(msg)

st.metric(label="🧠 병적 베타파 고착도 (Beta Band)", value=f"{current_beta:.1f} %")

# 파형 계산 및 데이터 누적
intent_wave = intent_base + 3.0 * np.sin(current_t * 4.0)
# 떨림파에 노이즈를 섞어 실제 파킨슨 진전(Tremor) 모사
tremor_wave = current_tremor * np.sin(current_t * 25.0) + np.random.normal(0, current_tremor * 0.3)
actual_movement = np.clip((intent_wave * current_efficiency) + tremor_wave, 0, 100)

new_row = pd.DataFrame([[float(intent_wave), float(actual_movement)]], columns=st.session_state.df.columns)
st.session_state.df = pd.concat([st.session_state.df.iloc[1:], new_row], ignore_index=True)

# 차트 그리기
st.line_chart(st.session_state.df, color=["#00FFFF", actual_color], height=450)

# --------------------------------------------------------
# [실시간 무한 루프] 프레임 속도 안정화
# --------------------------------------------------------
if is_running:
    time.sleep(0.25)
    st.rerun()
