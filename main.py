import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균기온 선형회귀 예측기")

st.write(
    "과거 연평균기온을 학습하여 최근 20년의 기온을 얼마나 잘 예측하는지 비교합니다."
)

# =========================================================
# 데이터 불러오기
# =========================================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()

# =========================================================
# 2025년까지 사용
# =========================================================
df = df[df["연도"] <= 2025].copy()

df = df.dropna(
    subset=["연도", "평균기온"]
)

# =========================================================
# 연도별 연평균기온 계산
# 관측일수 300일 미만인 해 제외
# =========================================================
yearly = (
    df.groupby("연도")
    .agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    )
    .reset_index()
)

yearly = yearly[
    yearly["관측일수"] >= 300
].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)

# =========================================================
# 학습 / 테스트 데이터 분리
# =========================================================
train_50 = yearly[
    (yearly["연도"] >= 1956) &
    (yearly["연도"] <= 2005)
].copy()

train_100 = yearly[
    (yearly["연도"] >= 1906) &
    (yearly["연도"] <= 2005)
].copy()

test = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()

# =========================================================
# 독립변수
# 이전 앱과 동일하게 '1908년부터 지난 연수' 사용
# =========================================================
train_50["경과연수"] = train_50["연도"] - 1908
train_100["경과연수"] = train_100["연도"] - 1908
test["경과연수"] = test["연도"] - 1908

X_50 = train_50[["경과연수"]]
y_50 = train_50["연평균기온"]

X_100 = train_100[["경과연수"]]
y_100 = train_100["연평균기온"]

X_test = test[["경과연수"]]
y_test = test["연평균기온"]

# =========================================================
# 선형회귀 모델 학습
# =========================================================
model_50 = LinearRegression()
model_50.fit(X_50, y_50)

model_100 = LinearRegression()
model_100.fit(X_100, y_100)

# =========================================================
# 테스트 데이터 예측
# =========================================================
pred_50 = model_50.predict(X_test)
pred_100 = model_100.predict(X_test)

# =========================================================
# 평가 지표 계산
# =========================================================
mae_50 = mean_absolute_error(y_test, pred_50)
mse_50 = mean_squared_error(y_test, pred_50)
r2_50 = r2_score(y_test, pred_50)

mae_100 = mean_absolute_error(y_test, pred_100)
mse_100 = mean_squared_error(y_test, pred_100)
r2_100 = r2_score(y_test, pred_100)

# =========================================================
# 기울기 계산
# °C / 년 → °C / 100년
# =========================================================
slope_50 = model_50.coef_[0]
slope_100 = model_100.coef_[0]

slope_50_100yr = slope_50 * 100
slope_100_100yr = slope_100 * 100

# =========================================================
# 제목
# =========================================================
st.header("1. 학습 데이터와 테스트 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "최근 50년 학습",
        f"{len(train_50)}개 연도"
    )
    st.caption(
        f"{train_50['연도'].min()}~{train_50['연도'].max()}년"
    )

with col2:
    st.metric(
        "최근 100년 학습",
        f"{len(train_100)}개 연도"
    )
    st.caption(
        f"{train_100['연도'].min()}~{train_100['연도'].max()}년"
    )

with col3:
    st.metric(
        "공통 테스트",
        f"{len(test)}개 연도"
    )
    st.caption(
        f"{test['연도'].min()}~{test['연도'].max()}년"
    )

st.info(
    "두 모델은 모두 2006~2025년 데이터를 학습에 사용하지 않고, "
    "이 기간을 공통 테스트 데이터로 사용합니다."
)

# =========================================================
# 기울기 비교
# =========================================================
st.header("2. 회귀선의 기울기 비교")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "1956~2005년 학습",
        f"{slope_50_100yr:+.2f} °C / 100년"
    )

with col2:
    st.metric(
        "1906~2005년 학습",
        f"{slope_100_100yr:+.2f} °C / 100년"
    )

st.caption(
    "기울기는 1년당 기온 변화량을 100년 기준으로 환산한 값입니다."
)

# =========================================================
# 테스트 성능 비교
# =========================================================
st.header("3. 최근 20년 테스트 성능 비교")

st.write(
    "2006~2025년의 실제 연평균기온을 두 회귀모델이 얼마나 잘 예측했는지 평가합니다."
)

result_df = pd.DataFrame({
    "모델": [
        "최근 50년 학습 (1956~2005)",
        "최근 100년 학습 (1906~2005)"
    ],
    "기울기 (°C/100년)": [
        slope_50_100yr,
        slope_100_100yr
    ],
    "MAE (°C)": [
        mae_50,
        mae_100
    ],
    "MSE (°C²)": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})

result_display = result_df.copy()

result_display["기울기 (°C/100년)"] = (
    result_display["기울기 (°C/100년)"].map(
        lambda x: f"{x:+.3f}"
    )
)

result_display["MAE (°C)"] = (
    result_display["MAE (°C)"].map(
        lambda x: f"{x:.3f}"
    )
)

result_display["MSE (°C²)"] = (
    result_display["MSE (°C²)"].map(
        lambda x: f"{x:.3f}"
    )
)

result_display["R²"] = (
    result_display["R²"].map(
        lambda x: f"{x:.3f}"
    )
)

st.dataframe(
    result_display,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# 지표를 크게 비교
# =========================================================
st.subheader("📊 평가 지표")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 최근 50년 학습")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric("MAE", f"{mae_50:.3f} °C")

    with metric_col2:
        st.metric("MSE", f"{mse_50:.3f}")

    with metric_col3:
        st.metric("R²", f"{r2_50:.3f}")

with col2:
    st.markdown("### 최근 100년 학습")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric("MAE", f"{mae_100:.3f} °C")

    with metric_col2:
        st.metric("MSE", f"{mse_100:.3f}")

    with metric_col3:
        st.metric("R²", f"{r2_100:.3f}")

# =========================================================
# 테스트 데이터 실제값 vs 예측값
# =========================================================
st.header("4. 테스트 기간 실제 기온과 예측값")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="lines+markers",
        name="실제 연평균기온",
        hovertemplate=(
            "%{x}년<br>"
            "실제 기온: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines",
        name="1956~2005 학습 예측",
        hovertemplate=(
            "%{x}년<br>"
            "예측 기온: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_100,
        mode="lines",
        name="1906~2005 학습 예측",
        hovertemplate=(
            "%{x}년<br>"
            "예측 기온: %{y:.2f}°C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(
        tickmode="linear",
        dtick=2
    ),
    height=600,
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# 회귀선 비교
# =========================================================
st.header("5. 두 회귀선 비교")

compare_years = np.arange(
    1906,
    2026
)

compare_x = (
    compare_years - 1908
).reshape(-1, 1)

line_50 = model_50.predict(compare_x)
line_100 = model_100.predict(compare_x)

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=train_50["연도"],
        y=train_50["연평균기온"],
        mode="markers",
        name="1956~2005 학습 데이터"
    )
)

fig2.add_trace(
    go.Scatter(
        x=train_100["연도"],
        y=train_100["연평균기온"],
        mode="markers",
        name="1906~2005 학습 데이터",
        marker=dict(
            symbol="circle-open"
        )
    )
)

fig2.add_trace(
    go.Scatter(
        x=compare_years,
        y=line_50,
        mode="lines",
        name="1956~2005 회귀선"
    )
)

fig2.add_trace(
    go.Scatter(
        x=compare_years,
        y=line_100,
        mode="lines",
        name="1906~2005 회귀선"
    )
)

# 테스트 데이터 영역 표시
fig2.add_vrect(
    x0=2006,
    x1=2025,
    fillcolor="gray",
    opacity=0.12,
    line_width=0,
    annotation_text="테스트 기간",
    annotation_position="top left"
)

fig2.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    height=650,
    hovermode="x unified"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# =========================================================
# 해석
# =========================================================
st.header("6. 결과 해석")

# MAE 기준으로 더 좋은 모델
if mae_50 < mae_100:
    mae_winner = "최근 50년 학습 모델"
else:
    mae_winner = "최근 100년 학습 모델"

# MSE 기준
if mse_50 < mse_100:
    mse_winner = "최근 50년 학습 모델"
else:
    mse_winner = "최근 100년 학습 모델"

# R² 기준
if r2_50 > r2_100:
    r2_winner = "최근 50년 학습 모델"
else:
    r2_winner = "최근 100년 학습 모델"

st.write(
    f"""
**기울기:**  
최근 50년 학습 모델의 100년당 기온 변화량은
**{slope_50_100yr:+.2f}°C**이고,
최근 100년 학습 모델은 **{slope_100_100yr:+.2f}°C**입니다.

**MAE:**  
더 작은 모델은 **{mae_winner}**입니다.

**MSE:**  
더 작은 모델은 **{mse_winner}**입니다.

**R²:**  
더 큰 모델은 **{r2_winner}**입니다.

따라서 최근 20년의 기온을 예측할 때는
단순히 더 긴 기간의 데이터를 사용하는 것이 항상 더 좋은 것은 아니며,
학습 기간에 따라 회귀선의 기울기와 예측 성능이 달라질 수 있습니다.
"""
)

# =========================================================
# 데이터 보기
# =========================================================
with st.expander("📋 연도별 데이터 확인"):

    display_df = yearly[
        ["연도", "관측일수", "연평균기온"]
    ].copy()

    display_df["연평균기온"] = (
        display_df["연평균기온"].round(2)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
