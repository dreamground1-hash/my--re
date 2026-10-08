import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="선형회귀 모델 평가",
    page_icon="📈",
    layout="wide"
)

st.title("📈 선형회귀 모델 평가")
st.write("과거 연도 데이터를 학습하여 최근 20년의 연평균 기온을 얼마나 잘 예측하는지 비교합니다.")


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

df = pd.read_csv(URL, encoding="utf-8")

# 날짜를 날짜 형식으로 변환
df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

# 연도 추출
df["연도"] = df["날짜"].dt.year

# 평균기온 숫자형 변환
df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

# 결측값 제거
df = df.dropna(subset=["연도", "평균기온"])


# --------------------------------------------------
# 연도별 평균기온 계산
# 관측일수가 300일 미만인 연도 제외
# 2025년까지의 데이터만 사용
# --------------------------------------------------
yearly = (
    df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

yearly = yearly[
    (yearly["연도"] <= 2025) &
    (yearly["관측일수"] >= 300)
].copy()

yearly = yearly.sort_values("연도")


# --------------------------------------------------
# 훈련 데이터 / 테스트 데이터 분리
# --------------------------------------------------

# 최근 50년: 1956~2005
train_50 = yearly[
    (yearly["연도"] >= 1956) &
    (yearly["연도"] <= 2005)
].copy()

# 최근 100년: 1906~2005
train_100 = yearly[
    (yearly["연도"] >= 1906) &
    (yearly["연도"] <= 2005)
].copy()

# 공통 테스트 데이터: 2006~2025
test = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()


# --------------------------------------------------
# 데이터 확인
# --------------------------------------------------
if len(train_50) < 2 or len(train_100) < 2 or len(test) < 2:
    st.error("회귀분석을 진행하기에 충분한 데이터가 없습니다.")
    st.stop()


# --------------------------------------------------
# 독립변수 설정
# 이전 기온 예측기와 동일하게 1908년을 기준으로 사용
# --------------------------------------------------
train_50["경과연수"] = train_50["연도"] - 1908
train_100["경과연수"] = train_100["연도"] - 1908
test["경과연수"] = test["연도"] - 1908


X_50 = train_50[["경과연수"]]
y_50 = train_50["연평균기온"]

X_100 = train_100[["경과연수"]]
y_100 = train_100["연평균기온"]

X_test = test[["경과연수"]]
y_test = test["연평균기온"]


# --------------------------------------------------
# 선형회귀 모델 학습
# --------------------------------------------------
model_50 = LinearRegression()
model_100 = LinearRegression()

model_50.fit(X_50, y_50)
model_100.fit(X_100, y_100)


# --------------------------------------------------
# 테스트 데이터 예측
# --------------------------------------------------
pred_50 = model_50.predict(X_test)
pred_100 = model_100.predict(X_test)


# --------------------------------------------------
# 평가 지표 계산
# --------------------------------------------------
mae_50 = mean_absolute_error(y_test, pred_50)
mse_50 = mean_squared_error(y_test, pred_50)
r2_50 = r2_score(y_test, pred_50)

mae_100 = mean_absolute_error(y_test, pred_100)
mse_100 = mean_squared_error(y_test, pred_100)
r2_100 = r2_score(y_test, pred_100)


# --------------------------------------------------
# 기울기 계산
# °C/년 → °C/100년
# --------------------------------------------------
slope_50 = model_50.coef_[0]
slope_100 = model_100.coef_[0]

slope_50_100 = slope_50 * 100
slope_100_100 = slope_100 * 100


# --------------------------------------------------
# 학습 데이터 / 테스트 데이터 정보
# --------------------------------------------------
st.subheader("📊 데이터 구성")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "최근 50년 학습",
        f"{train_50['연도'].min()}~{train_50['연도'].max()}",
        f"{len(train_50)}개 연도"
    )

with col2:
    st.metric(
        "최근 100년 학습",
        f"{train_100['연도'].min()}~{train_100['연도'].max()}",
        f"{len(train_100)}개 연도"
    )

with col3:
    st.metric(
        "공통 테스트",
        f"{test['연도'].min()}~{test['연도'].max()}",
        f"{len(test)}개 연도"
    )


# --------------------------------------------------
# 회귀선 기울기 비교
# --------------------------------------------------
st.subheader("📈 회귀선 기울기 비교")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "최근 50년 학습 기울기",
        f"{slope_50_100:.2f} °C / 100년"
    )

with col2:
    st.metric(
        "최근 100년 학습 기울기",
        f"{slope_100_100:.2f} °C / 100년"
    )

st.info(
    "기울기가 클수록 시간의 흐름에 따라 연평균 기온이 더 빠르게 상승하는 것으로 나타납니다."
)


# --------------------------------------------------
# 테스트 데이터 예측 성능 비교
# --------------------------------------------------
st.subheader("🎯 테스트 데이터 예측 성능 비교")

comparison = pd.DataFrame({
    "평가지표": ["MAE", "MSE", "R²"],
    "최근 50년 학습": [mae_50, mse_50, r2_50],
    "최근 100년 학습": [mae_100, mse_100, r2_100]
})

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

st.markdown("""
**MAE**는 실제 기온과 예측 기온의 평균적인 차이를 나타내며,  
**MSE**는 예측 오차를 제곱하여 평균한 값입니다.  
**R²**는 모델이 기온의 변화를 얼마나 잘 설명하는지를 나타내며, 1에 가까울수록 좋습니다.
""")


# --------------------------------------------------
# 실제값 vs 예측값
# --------------------------------------------------
st.subheader("🌡️ 최근 20년 실제 기온과 예측 기온 비교")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=y_test,
        mode="lines+markers",
        name="실제 연평균 기온"
    )
)

fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines+markers",
        name="50년 학습 예측"
    )
)

fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_100,
        mode="lines+markers",
        name="100년 학습 예측"
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (°C)",
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 회귀선 비교
# --------------------------------------------------
st.subheader("📉 50년 학습과 100년 학습 회귀선 비교")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=train_50["연도"],
        y=train_50["연평균기온"],
        mode="markers",
        name="50년 학습 데이터"
    )
)

fig2.add_trace(
    go.Scatter(
        x=train_100["연도"],
        y=train_100["연평균기온"],
        mode="markers",
        name="100년 학습 데이터"
    )
)

# 회귀선용 연도
line_years = np.arange(
    int(yearly["연도"].min()),
    2026
)

line_x = (line_years - 1908).reshape(-1, 1)

fig2.add_trace(
    go.Scatter(
        x=line_years,
        y=model_50.predict(line_x),
        mode="lines",
        name="50년 학습 회귀선"
    )
)

fig2.add_trace(
    go.Scatter(
        x=line_years,
        y=model_100.predict(line_x),
        mode="lines",
        name="100년 학습 회귀선"
    )
)

fig2.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (°C)",
    hovermode="x unified",
    height=550
)

st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------
# 결과 해석
# --------------------------------------------------
st.subheader("📝 결과 해석")

if mae_50 < mae_100:
    mae_result = "최근 50년 학습 모델의 MAE가 더 작아 최근 기온을 평균적으로 더 정확하게 예측했습니다."
else:
    mae_result = "최근 100년 학습 모델의 MAE가 더 작아 최근 기온을 평균적으로 더 정확하게 예측했습니다."

if mse_50 < mse_100:
    mse_result = "최근 50년 학습 모델의 MSE가 더 작아 큰 예측 오차가 상대적으로 적었습니다."
else:
    mse_result = "최근 100년 학습 모델의 MSE가 더 작아 큰 예측 오차가 상대적으로 적었습니다."

if r2_50 > r2_100:
    r2_result = "최근 50년 학습 모델의 R²가 더 높아 최근 20년의 기온 변화를 더 잘 설명했습니다."
else:
    r2_result = "최근 100년 학습 모델의 R²가 더 높아 최근 20년의 기온 변화를 더 잘 설명했습니다."

st.write(f"- {mae_result}")
st.write(f"- {mse_result}")
st.write(f"- {r2_result}")
st.write(
    f"- 회귀선의 기울기는 최근 50년 학습 모델이 "
    f"{slope_50_100:.2f} °C/100년, 최근 100년 학습 모델이 "
    f"{slope_100_100:.2f} °C/100년으로 나타났습니다."
)
