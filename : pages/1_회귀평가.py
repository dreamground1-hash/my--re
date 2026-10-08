import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="선형회귀 모델 평가",
    page_icon="📈",
    layout="wide"
)

st.title("📈 연평균 기온 선형회귀 모델 평가")

st.write(
    "과거 연평균 기온 데이터를 훈련데이터와 테스트데이터로 나누어 "
    "선형회귀 모델의 예측 성능을 비교합니다."
)


# =========================================================
# 1. 데이터 불러오기
# =========================================================
URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

df = pd.read_csv(URL, encoding="utf-8")

df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
df["연도"] = df["날짜"].dt.year
df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

df = df.dropna(subset=["연도", "평균기온"])


# =========================================================
# 2. 연도별 연평균 기온 계산
#    - 2025년까지 사용
#    - 관측일수 300일 미만인 연도 제외
# =========================================================
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


# =========================================================
# 3. 데이터 구분
# =========================================================

# 최근 50년 학습 데이터
train_50 = yearly[
    (yearly["연도"] >= 1956) &
    (yearly["연도"] <= 2005)
].copy()

# 최근 100년 학습 데이터
train_100 = yearly[
    (yearly["연도"] >= 1906) &
    (yearly["연도"] <= 2005)
].copy()

# 공통 테스트 데이터
test = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()


# 데이터가 충분한지 확인
if len(train_50) < 2 or len(train_100) < 2 or len(test) < 2:
    st.error("회귀분석을 진행하기에 충분한 데이터가 없습니다.")
    st.stop()


# =========================================================
# 4. 독립변수 설정
#    이전 기온 예측기와 동일하게 1908년을 기준으로 함
# =========================================================
for data in [train_50, train_100, test]:
    data["경과연수"] = data["연도"] - 1908


X_50 = train_50[["경과연수"]]
y_50 = train_50["연평균기온"]

X_100 = train_100[["경과연수"]]
y_100 = train_100["연평균기온"]

X_test = test[["경과연수"]]
y_test = test["연평균기온"]


# =========================================================
# 5. 50년 / 100년 선형회귀 모델 학습
# =========================================================
model_50 = LinearRegression()
model_100 = LinearRegression()

model_50.fit(X_50, y_50)
model_100.fit(X_100, y_100)


# =========================================================
# 6. 테스트 데이터 예측
# =========================================================
pred_50 = model_50.predict(X_test)
pred_100 = model_100.predict(X_test)


# =========================================================
# 7. 예측 성능 평가
# =========================================================

# 50년 모델
mae_50 = mean_absolute_error(y_test, pred_50)
mse_50 = mean_squared_error(y_test, pred_50)
r2_50 = r2_score(y_test, pred_50)

# 100년 모델
mae_100 = mean_absolute_error(y_test, pred_100)
mse_100 = mean_squared_error(y_test, pred_100)
r2_100 = r2_score(y_test, pred_100)


# =========================================================
# 8. 회귀선 기울기
#    °C/년 → °C/100년
# =========================================================
slope_50 = model_50.coef_[0]
slope_100 = model_100.coef_[0]

slope_50_100 = slope_50 * 100
slope_100_100 = slope_100 * 100


# =========================================================
# 9. 지난번 전체 데이터 회귀모델
#    ※ 참고용 비교
# =========================================================
yearly["경과연수"] = yearly["연도"] - 1908

X_all = yearly[["경과연수"]]
y_all = yearly["연평균기온"]

model_all = LinearRegression()
model_all.fit(X_all, y_all)

slope_all = model_all.coef_[0] * 100


# =========================================================
# 10. 데이터 구성 표시
# =========================================================
st.subheader("1. 훈련데이터와 테스트데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "최근 50년 훈련데이터",
        "1956~2005",
        f"{len(train_50)}개 연도"
    )

with col2:
    st.metric(
        "최근 100년 훈련데이터",
        "1906~2005",
        f"{len(train_100)}개 연도"
    )

with col3:
    st.metric(
        "공통 테스트데이터",
        "2006~2025",
        f"{len(test)}개 연도"
    )

st.info(
    "50년 모델과 100년 모델은 모두 학습에 사용하지 않은 "
    "동일한 2006~2025년 데이터를 테스트에 사용합니다."
)


# =========================================================
# 11. 회귀선 기울기 비교
# =========================================================
st.subheader("2. 회귀선 기울기 비교")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "전체 데이터",
        f"{slope_all:.2f} °C/100년"
    )

with col2:
    st.metric(
        "최근 50년 학습",
        f"{slope_50_100:.2f} °C/100년"
    )

with col3:
    st.metric(
        "최근 100년 학습",
        f"{slope_100_100:.2f} °C/100년"
    )

st.write(
    "기울기는 100년 동안 연평균 기온이 얼마나 변하는지를 나타냅니다."
)


# =========================================================
# 12. 테스트 데이터 예측 성능 비교
# =========================================================
st.subheader("3. 테스트 데이터 예측 성능 비교")

comparison = pd.DataFrame({
    "평가지표": ["MAE", "MSE", "R²"],
    "최근 50년 학습": [
        mae_50,
        mse_50,
        r2_50
    ],
    "최근 100년 학습": [
        mae_100,
        mse_100,
        r2_100
    ]
})

st.dataframe(
    comparison.style.format({
        "최근 50년 학습": "{:.4f}",
        "최근 100년 학습": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 13. 평가 지표 설명
# =========================================================
st.markdown("""
- **MAE**: 실제 기온과 예측 기온의 평균적인 차이입니다. **작을수록 좋습니다.**
- **MSE**: 예측 오차를 제곱하여 평균한 값입니다. **작을수록 좋습니다.**
- **R²**: 실제 기온의 변화를 모델이 얼마나 잘 설명하는지를 나타냅니다. **1에 가까울수록 좋습니다.**
""")


# =========================================================
# 14. 실제값과 예측값 비교
# =========================================================
st.subheader("4. 최근 20년 실제 기온과 예측 기온")

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


# =========================================================
# 15. 50년 / 100년 회귀선 비교
# =========================================================
st.subheader("5. 50년 학습과 100년 학습 회귀선 비교")

line_years = np.arange(
    int(yearly["연도"].min()),
    2026
)

line_x = (line_years - 1908).reshape(-1, 1)

pred_line_50 = model_50.predict(line_x)
pred_line_100 = model_100.predict(line_x)

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=train_100["연도"],
        y=train_100["연평균기온"],
        mode="markers",
        name="학습 데이터"
    )
)

fig2.add_trace(
    go.Scatter(
        x=line_years,
        y=pred_line_50,
        mode="lines",
        name="50년 학습 회귀선"
    )
)

fig2.add_trace(
    go.Scatter(
        x=line_years,
        y=pred_line_100,
        mode="lines",
        name="100년 학습 회귀선"
    )
)

fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="markers",
        name="테스트 실제값"
    )
)

fig2.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (°C)",
    hovermode="x unified",
    height=550
)

st.plotly_chart(fig2, use_container_width=True)


# =========================================================
# 16. 어떤 모델이 더 좋은지 비교
# =========================================================
st.subheader("6. 모델 성능 비교")

if mae_50 < mae_100:
    mae_result = "MAE는 최근 50년 학습 모델이 더 작아 평균적인 예측 오차가 더 작습니다."
else:
    mae_result = "MAE는 최근 100년 학습 모델이 더 작아 평균적인 예측 오차가 더 작습니다."

if mse_50 < mse_100:
    mse_result = "MSE는 최근 50년 학습 모델이 더 작아 큰 예측 오차가 상대적으로 적습니다."
else:
    mse_result = "MSE는 최근 100년 학습 모델이 더 작아 큰 예측 오차가 상대적으로 적습니다."

if r2_50 > r2_100:
    r2_result = "R²는 최근 50년 학습 모델이 더 높아 최근 기온 변화를 더 잘 설명합니다."
else:
    r2_result = "R²는 최근 100년 학습 모델이 더 높아 최근 기온 변화를 더 잘 설명합니다."

st.write(f"• {mae_result}")
st.write(f"• {mse_result}")
st.write(f"• {r2_result}")

st.write(
    f"• 회귀선의 기울기는 최근 50년 학습이 "
    f"{slope_50_100:.2f} °C/100년, "
    f"최근 100년 학습이 {slope_100_100:.2f} °C/100년입니다."
)

st.write(
    "따라서 과거 데이터를 얼마나 넓은 기간 동안 학습시키느냐에 따라 "
    "회귀선의 기울기와 최근 기온에 대한 예측 성능이 달라질 수 있습니다."
)
