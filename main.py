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

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()

# ==========================================
# 2025년까지의 자료만 사용
# ==========================================
df = df[df["연도"] <= 2025].copy()
df = df.dropna(subset=["평균기온"])

# ==========================================
# 연도별 평균기온 및 관측일수 계산
# ==========================================
yearly = (
    df.groupby("연도")
    .agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    )
    .reset_index()
)

# 관측일수가 300일 이상인 해만 사용
yearly = yearly[yearly["관측일수"] >= 300].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)

# ==========================================
# 전체 기간 회귀분석
# 독립변수 = 1908년부터 지난 연수
# ==========================================
yearly["경과연수"] = yearly["연도"] - 1908

x = yearly["경과연수"].to_numpy()
y = yearly["연평균기온"].to_numpy()

전체_기울기, 전체_절편 = np.polyfit(x, y, 1)

# 100년에 몇 도 상승하는가?
전체_100년_상승 = 전체_기울기 * 100

# 전체 기간 상관계수
전체_상관계수 = np.corrcoef(x, y)[0, 1]

# ==========================================
# 최근 20년 회귀분석
# ==========================================
최근20 = yearly.tail(20).copy()

최근20_x = (최근20["연도"] - 1908).to_numpy()
최근20_y = 최근20["연평균기온"].to_numpy()

최근20_기울기, 최근20_절편 = np.polyfit(
    최근20_x,
    최근20_y,
    1
)

최근20_100년_상승 = 최근20_기울기 * 100

최근20_상관계수 = np.corrcoef(
    최근20_x,
    최근20_y
)[0, 1]

# ==========================================
# 회귀선
# ==========================================
reg_years = np.arange(1900, 2101)
reg_x = reg_years - 1908

전체_reg_y = 전체_기울기 * reg_x + 전체_절편

# ==========================================
# 기본 정보
# ==========================================
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
data_count = len(yearly)

최근20_start = int(최근20["연도"].min())
최근20_end = int(최근20["연도"].max())

# ==========================================
# 회귀분석 정보
# ==========================================
st.subheader("📊 회귀분석 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("사용한 연도 수", f"{data_count}개")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")

with col4:
    st.metric("전체 기간 상관계수", f"{전체_상관계수:.3f}")

st.caption(
    "2025년까지의 자료 중 관측일수가 300일 이상인 연도만 사용했습니다."
)

# ==========================================
# 100년당 기온 상승량
# ==========================================
st.subheader("🌡️ 100년에 몇 °C 오르는가?")

st.metric(
    "전체 기간",
    f"{전체_100년_상승:+.2f} °C",
    help="전체 분석 기간의 회귀직선 기울기를 100년 기준으로 환산한 값입니다."
)

# ==========================================
# 전체 기간 vs 최근 20년 비교
# ==========================================
st.subheader("📈 전체 기간과 최근 20년 비교")

compare_col1, compare_col2 = st.columns(2)

with compare_col1:
    st.markdown("### 전체 기간")
    st.metric(
        "100년당 기온 변화",
        f"{전체_100년_상승:+.2f} °C"
    )
    st.write(
        f"분석 기간: **{start_year}~{end_year}년**"
    )
    st.write(
        f"사용 연도: **{data_count}개**"
    )

with compare_col2:
    st.markdown("### 최근 20년")
    st.metric(
        "100년당 기온 변화",
        f"{최근20_100년_상승:+.2f} °C"
    )
    st.write(
        f"분석 기간: **{최근20_start}~{최근20_end}년**"
    )
    st.write(
        f"사용 연도: **{len(최근20)}개**"
    )

# ==========================================
# 비교 막대그래프
# ==========================================
compare_fig = go.Figure()

compare_fig.add_trace(
    go.Bar(
        x=["전체 기간", "최근 20년"],
        y=[전체_100년_상승, 최근20_100년_상승],
        text=[
            f"{전체_100년_상승:+.2f}°C",
            f"{최근20_100년_상승:+.2f}°C"
        ],
        textposition="outside",
        hovertemplate=(
            "%{x}<br>"
            "100년당 기온 변화: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)

compare_fig.update_layout(
    xaxis_title="분석 기간",
    yaxis_title="100년당 기온 변화 (°C)",
    height=400
)

st.plotly_chart(
    compare_fig,
    use_container_width=True
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
predicted_temp = 전체_기울기 * selected_x + 전체_절편

st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{predicted_temp:.2f} °C"
)

# ==========================================
# 산점도 + 회귀직선
# ==========================================
st.subheader("📈 서울 연평균기온과 회귀직선")

fig = go.Figure()

# 실제 연평균기온
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

# 전체 기간 회귀직선
fig.add_trace(
    go.Scatter(
        x=reg_years,
        y=전체_reg_y,
        mode="lines",
        name="전체 기간 회귀직선",
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 예측기온: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)

# 선택한 연도 예측값
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예측",
        marker=dict(size=14),
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
    height=600
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# 회귀식
# ==========================================
st.subheader("🧮 회귀식")

st.write(
    f"전체 기간 회귀식: "
    f"연평균기온 = {전체_기울기:.5f} × (연도 - 1908) "
    f"+ {전체_절편:.2f}"
)

st.write(
    f"최근 20년 회귀식: "
    f"연평균기온 = {최근20_기울기:.5f} × (연도 - 1908) "
    f"+ {최근20_절편:.2f}"
)

# ==========================================
# 데이터 표
# ==========================================
with st.expander("📋 분석에 사용된 연도별 데이터 보기"):
    display_df = yearly[
        ["연도", "관측일수", "연평균기온"]
    ].copy()

    display_df["연평균기온"] = display_df["연평균기온"].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
