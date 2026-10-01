import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 기본 설정
# ==========================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write("서울의 연평균기온 데이터를 이용해 기온 변화를 분석하고 미래 기온을 예측합니다.")

# ==========================================
# 데이터 불러오기
# ==========================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜를 날짜형으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온 숫자형 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()

# ==========================================
# 2025년까지의 데이터만 사용
# ==========================================
df = df[df["연도"] <= 2025].copy()

# 평균기온이 없는 자료 제거
df = df.dropna(subset=["평균기온"])

# ==========================================
# 연도별 관측일 수와 연평균기온 계산
# ==========================================
yearly = (
    df.groupby("연도")
    .agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    )
    .reset_index()
)

# 관측일이 300일 이상인 해만 사용
yearly = yearly[yearly["관측일수"] >= 300].copy()

# 연도순 정렬
yearly = yearly.sort_values("연도").reset_index(drop=True)

# ==========================================
# 회귀분석
# 독립변수: 1908년부터 지난 연수
# ==========================================
yearly["경과연수"] = yearly["연도"] - 1908

x = yearly["경과연수"].to_numpy()
y = yearly["연평균기온"].to_numpy()

# 1차 선형회귀
slope, intercept = np.polyfit(x, y, 1)

# 예측값
yearly["회귀예측기온"] = slope * yearly["경과연수"] + intercept

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# ==========================================
# 회귀선 계산용 연도
# 1900 ~ 2100
# ==========================================
reg_years = np.arange(1900, 2101)
reg_x = reg_years - 1908
reg_y = slope * reg_x + intercept

# ==========================================
# 회귀분석 정보
# ==========================================
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
data_count = len(yearly)

st.subheader("📊 회귀분석 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("사용한 연도 수", f"{data_count}개")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")

with col4:
    st.metric("상관계수", f"{correlation:.3f}")

st.caption(
    f"※ 2025년까지의 자료 중 연간 관측일수가 300일 이상인 연도만 사용했습니다. "
    f"회귀분석의 독립변수는 '연도 - 1908'입니다."
)

# ==========================================
# 연도 선택 슬라이더
# ==========================================
st.subheader("🔮 연도별 예상 기온")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

selected_x = selected_year - 1908
predicted_temp = slope * selected_x + intercept

st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{predicted_temp:.2f} °C"
)

# ==========================================
# 산점도 + 회귀직선
# ==========================================
st.subheader("📈 서울 연평균기온과 회귀직선")

fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        text=[
            f"{year}년<br>"
            f"연평균기온: {temp:.2f}°C<br>"
            f"관측일수: {days}일"
            for year, temp, days in zip(
                yearly["연도"],
                yearly["연평균기온"],
                yearly["관측일수"]
            )
        ],
        hovertemplate="%{text}<extra></extra>"
    )
)

# 회귀직선
fig.add_trace(
    go.Scatter(
        x=reg_years,
        y=reg_y,
        mode="lines",
        name="회귀직선",
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 예측기온: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)

# 선택한 연도의 예측 위치
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예측",
        marker=dict(size=13),
        hovertemplate=(
            f"{selected_year}년<br>"
            f"예측기온: {predicted_temp:.2f}°C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    hovermode="closest",
    legend_title="구분",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 회귀식
# ==========================================
st.subheader("🧮 회귀식")

st.write(
    f"연평균기온 = {slope:.5f} × (연도 - 1908) + {intercept:.2f}"
)

st.write(
    f"상관계수: **{correlation:.3f}**"
)

# ==========================================
# 연도별 데이터 표
# ==========================================
with st.expander("📋 분석에 사용된 연도별 데이터 보기"):
    display_df = yearly[
        ["연도", "관측일수", "연평균기온", "회귀예측기온"]
    ].copy()

    display_df["연평균기온"] = display_df["연평균기온"].round(2)
    display_df["회귀예측기온"] = display_df["회귀예측기온"].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
