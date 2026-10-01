import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
st.set_page_config(
    page_title="서울 기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

BASE_YEAR = 1908
LAST_DATA_YEAR = 2025
MIN_OBSERVATIONS = 300


# ------------------------------------------------------------
# 데이터 불러오기
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig",
    )

    # 날짜를 날짜형으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 필요한 숫자형 열 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 날짜 또는 평균기온이 없는 행 제거
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 연도 생성
    df["연도"] = df["날짜"].dt.year

    return df


# ------------------------------------------------------------
# 연도별 평균기온 계산
# ------------------------------------------------------------
@st.cache_data
def make_yearly_data(df):
    yearly = (
        df.groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count"),
        )
        .reset_index()
    )

    # 수업 기준:
    # 1. 2025년까지
    # 2. 관측일수가 300일 이상인 해만 사용
    yearly = yearly[
        (yearly["연도"] <= LAST_DATA_YEAR)
        & (yearly["관측일수"] >= MIN_OBSERVATIONS)
    ].copy()

    # 1908년부터 지난 연수
    yearly["경과연수"] = yearly["연도"] - BASE_YEAR

    return yearly


# ------------------------------------------------------------
# 회귀 계산
# ------------------------------------------------------------
@st.cache_data
def calculate_regression(yearly):
    x = yearly["경과연수"].to_numpy(dtype=float)
    y = yearly["평균기온"].to_numpy(dtype=float)

    # y = slope * x + intercept
    slope, intercept = np.polyfit(x, y, 1)

    # 상관계수
    correlation = np.corrcoef(x, y)[0, 1]

    return slope, intercept, correlation


# ------------------------------------------------------------
# 앱 본문
# ------------------------------------------------------------
st.title("🌡️ 서울 기온 예측기")
st.caption("서울 연평균기온 데이터를 이용한 선형 회귀 기반 예측")

try:
    raw_df = load_data()
    yearly = make_yearly_data(raw_df)

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


if yearly.empty or len(yearly) < 2:
    st.error("회귀 분석에 사용할 수 있는 연도 데이터가 충분하지 않습니다.")
    st.stop()


slope, intercept, correlation = calculate_regression(yearly)


# ------------------------------------------------------------
# 회귀선의 기준 정보
# ------------------------------------------------------------
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
year_count = len(yearly)

st.info(
    f"회귀선에 사용된 연도 수: **{year_count}개**  |  "
    f"시작 연도: **{start_year}년**  |  "
    f"끝 연도: **{end_year}년**"
)


# ------------------------------------------------------------
# 연도 슬라이더
# ------------------------------------------------------------
selected_year = st.slider(
    "예측할 연도",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)


# 경과 연수 = 선택 연도 - 1908
selected_elapsed_years = selected_year - BASE_YEAR

# 회귀식에 따른 예상 기온
predicted_temperature = (
    slope * selected_elapsed_years + intercept
)


# ------------------------------------------------------------
# 선택 연도의 예상 기온 크게 표시
# ------------------------------------------------------------
st.markdown(
    f"""
    <div style="
        background-color: #f0f7ff;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        margin: 10px 0 25px 0;
        border: 1px solid #d5e8ff;
    ">
        <div style="font-size: 1.2rem; color: #555;">
            {selected_year}년 예상 평균기온
        </div>
        <div style="
            font-size: 3.5rem;
            font-weight: 700;
            color: #1976d2;
            margin-top: 5px;
        ">
            {predicted_temperature:.2f} °C
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 주요 통계
# ------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("상관계수", f"{correlation:.4f}")

with col2:
    st.metric("회귀선 기울기", f"{slope:.5f} °C/년")

with col3:
    st.metric("회귀선 사용 기간", f"{start_year}–{end_year}")


# ------------------------------------------------------------
# 산점도 + 회귀선
# ------------------------------------------------------------
# 회귀선은 1908년부터 2100년까지 표시
line_years = np.arange(BASE_YEAR, 2101)
line_elapsed = line_years - BASE_YEAR
line_temperatures = slope * line_elapsed + intercept

fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7,
            color="#1976D2",
            opacity=0.75,
        ),
        customdata=yearly["관측일수"],
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        ),
    )
)

# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temperatures,
        mode="lines",
        name="회귀 직선",
        line=dict(
            color="#E53935",
            width=3,
        ),
        hovertemplate=(
            "%{x}년<br>"
            "회귀 예측: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

# 현재 선택한 연도의 예측값
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temperature],
        mode="markers",
        name=f"{selected_year}년 예측",
        marker=dict(
            size=15,
            color="#FF9800",
            symbol="star",
            line=dict(
                color="white",
                width=2,
            ),
        ),
        hovertemplate=(
            f"<b>{selected_year}년</b><br>"
            f"예상 평균기온: {predicted_temperature:.2f} °C"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    title="서울 연평균기온과 선형 회귀",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    xaxis=dict(
        # 가로축에는 숫자 연도를 그대로 표시
        tickmode="linear",
        dtick=10,
        range=[1900, 2100],
    ),
    yaxis=dict(
        zeroline=False,
    ),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    height=600,
    margin=dict(l=50, r=30, t=80, b=50),
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ------------------------------------------------------------
# 회귀식 표시
# ------------------------------------------------------------
st.subheader("회귀식")

st.latex(
    rf"\hat{{T}} = {intercept:.4f} + {slope:.5f}"
    rf"\times (\mathrm{{연도}} - {BASE_YEAR})"
)

st.caption(
    "회귀의 독립 변수는 1908년을 기준으로 한 경과 연수입니다. "
    "2025년 이후 데이터와 관측일수가 300일 미만인 연도는 회귀 계산에서 제외했습니다."
)


# ------------------------------------------------------------
# 사용 데이터 확인
# ------------------------------------------------------------
with st.expander("회귀에 사용된 연도별 데이터 보기"):
    display_df = yearly.copy()
    display_df["평균기온"] = display_df["평균기온"].round(2)

    st.dataframe(
        display_df[
            ["연도", "평균기온", "관측일수", "경과연수"]
        ],
        use_container_width=True,
        hide_index=True,
    )
