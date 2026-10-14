import os
import random
from datetime import date, timedelta

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

SEED = 42
WINDOW = 60  # số phiên dùng làm đầu vào cho mỗi mẫu

st.set_page_config(page_title="Dự báo giá cổ phiếu LSTM", page_icon="📈")
plt.style.use("dark_background")


# ----------------------------------------------------------------------------
# Dữ liệu
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_close(ticker: str, start: date, end: date) -> pd.Series:
    """Tải giá đóng cửa. yfinance coi `end` là mốc loại trừ nên cộng thêm 1 ngày."""
    df = yf.download(
        ticker,
        start=start,
        end=end + timedelta(days=1),
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        return pd.Series(dtype=float)
    # yfinance bản mới trả về cột MultiIndex ('Close', 'VNM.VN') -> làm phẳng
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].dropna().astype(float)


def make_sequences(arr: np.ndarray, window: int = WINDOW):
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i - window : i])
        y.append(arr[i])
    return np.array(X), np.array(y)


# ----------------------------------------------------------------------------
# Huấn luyện + dự báo
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_pipeline(close: pd.Series, horizon: int) -> dict:
    # Import TensorFlow muộn để trang tải nhanh khi chưa bấm chạy
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.models import Sequential

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.keras.backend.clear_session()

    data = close.values.reshape(-1, 1)

    # Chia theo thời gian (không xáo trộn): 80% train / 20% test
    train_size = int(0.8 * len(data))
    train_data, test_data = data[:train_size], data[train_size:]

    # Chỉ fit scaler trên train để không rò rỉ thông tin từ tương lai
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = make_sequences(train_scaled)
    # Ghép 60 phiên cuối của train vào đầu test để dự báo được ngay phiên test đầu tiên
    test_combined = np.vstack((train_scaled[-WINDOW:], test_scaled))
    X_test, y_test = make_sequences(test_combined)

    # Validation = 10% cuối của train (giữ thứ tự thời gian)
    val_size = int(0.1 * len(X_train))
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_fit, y_fit = X_train[:-val_size], y_train[:-val_size]

    model = Sequential(
        [
            Input(shape=(WINDOW, 1)),
            LSTM(50, return_sequences=True),
            LSTM(50),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        shuffle=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=0,
    )

    # --- Đánh giá trên tập test ---
    test_pred = scaler.inverse_transform(model.predict(X_test, verbose=0)).ravel()
    y_real = scaler.inverse_transform(y_test).ravel()
    # Mô hình "ngây thơ": giá hôm nay = giá hôm qua, dùng làm mốc so sánh
    naive_pred = scaler.inverse_transform(X_test[:, -1, :]).ravel()

    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_real, test_pred))),
        "MAE": float(mean_absolute_error(y_real, test_pred)),
        "MAPE": float(np.mean(np.abs((y_real - test_pred) / y_real)) * 100),
        "R2": float(r2_score(y_real, test_pred)),
        "RMSE_naive": float(np.sqrt(mean_squared_error(y_real, naive_pred))),
    }

    # --- Dự báo `horizon` phiên tiếp theo (đệ quy) ---
    window = np.vstack((train_scaled, test_scaled))[-WINDOW:].reshape(1, WINDOW, 1)
    future_scaled = []
    for _ in range(horizon):
        nxt = model.predict(window, verbose=0)
        future_scaled.append(nxt[0, 0])
        window = np.concatenate([window[:, 1:, :], nxt.reshape(1, 1, 1)], axis=1)
    future = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).ravel()

    # Ngày giao dịch tiếp theo (bỏ T7/CN; chưa trừ ngày lễ)
    future_dates = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=horizon)

    test_dates = close.index[train_size:]
    return {
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "metrics": metrics,
        "test_dates": test_dates,
        "y_real": y_real,
        "test_pred": test_pred,
        "future": pd.Series(future, index=future_dates, name="Giá dự báo"),
        "last_close": float(close.iloc[-1]),
        "last_date": close.index[-1],
        "recent": close.iloc[-90:],
    }


# ----------------------------------------------------------------------------
# Giao diện
# ----------------------------------------------------------------------------
st.title("📈 Mô hình Dự báo Giá Cổ phiếu bằng LSTM")
st.write(
    "Dự báo giá đóng cửa (Close) cho 10 phiên giao dịch tiếp theo. "
    "Kết quả mang tính tham khảo, không phải khuyến nghị đầu tư."
)

with st.form("params"):
    ticker = st.text_input(
        "Mã cổ phiếu (theo yfinance, ví dụ: VNM.VN, VIC.VN, AAPL)",
        value="VNM.VN",
        help="Dùng đúng mã trên Yahoo Finance. Cổ phiếu Việt Nam có đuôi .VN (VD: VNM.VN).",
    )
    c1, c2 = st.columns(2)
    start = c1.date_input(
        "Ngày bắt đầu",
        value=date(2018, 1, 1),
        min_value=date(2000, 1, 1),
        max_value=date.today(),
        format="YYYY/MM/DD",
    )
    end = c2.date_input(
        "Ngày kết thúc",
        value=date.today(),
        min_value=date(2000, 1, 1),
        max_value=date.today(),
        format="YYYY/MM/DD",
    )
    horizon = st.selectbox("Số phiên muốn dự báo", [5, 10, 15, 20, 30], index=1)
    submitted = st.form_submit_button("🚀 Chạy dự báo", type="primary", width="stretch")

if not submitted:
    st.info("Nhập mã cổ phiếu và khoảng thời gian, sau đó bấm **Chạy dự báo** để xem kết quả.")
    st.stop()

ticker = ticker.strip().upper()
if not ticker:
    st.error("Vui lòng nhập mã cổ phiếu.")
    st.stop()
if start >= end:
    st.error("Ngày bắt đầu phải nhỏ hơn ngày kết thúc.")
    st.stop()

with st.spinner(f"Đang tải dữ liệu {ticker}..."):
    try:
        close = load_close(ticker, start, end)
    except Exception as e:  # lỗi mạng / yfinance
        st.error(f"Không tải được dữ liệu: {e}")
        st.stop()

if close.empty:
    st.error(
        f"Không có dữ liệu cho mã **{ticker}**. Hãy kiểm tra lại mã (cổ phiếu VN có đuôi .VN) "
        "hoặc thử một mã khác như AAPL. Lưu ý Yahoo Finance không phải lúc nào cũng có đủ dữ liệu "
        "cổ phiếu Việt Nam."
    )
    st.stop()

MIN_ROWS = 300
if len(close) < MIN_ROWS:
    st.error(f"Chỉ có {len(close)} phiên dữ liệu, cần ít nhất {MIN_ROWS}. Hãy kéo dài khoảng thời gian.")
    st.stop()

with st.spinner("Đang huấn luyện mô hình LSTM (có thể mất 1–3 phút)..."):
    res = run_pipeline(close, horizon)

m = res["metrics"]
st.success(f"Đã tải {len(close):,} phiên của {ticker} ({close.index[0]:%d/%m/%Y} → {close.index[-1]:%d/%m/%Y}).")

st.subheader("Đánh giá trên tập test")
a, b, c, d = st.columns(4)
a.metric("RMSE", f"{m['RMSE']:,.2f}")
b.metric("MAE", f"{m['MAE']:,.2f}")
c.metric("MAPE", f"{m['MAPE']:.2f}%")
d.metric("R²", f"{m['R2']:.3f}")
better = m["RMSE"] < m["RMSE_naive"]
st.caption(
    f"Mốc so sánh 'giá hôm nay = giá hôm qua' có RMSE = {m['RMSE_naive']:,.2f}. "
    + ("Mô hình tốt hơn mốc này." if better else "Mô hình chưa vượt được mốc này — cần thận trọng khi dùng kết quả.")
)

st.subheader("Loss: Train vs Validation")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(res["loss"], label="Train Loss")
ax.plot(res["val_loss"], label="Validation Loss")
ax.set_xlabel("Epoch")
ax.legend()
st.pyplot(fig)

st.subheader("Giá thật vs Dự báo (tập test)")
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(res["test_dates"], res["y_real"], label="Giá thật")
ax.plot(res["test_dates"], res["test_pred"], label="Dự báo")
ax.legend()
fig.autofmt_xdate()
st.pyplot(fig)

st.subheader(f"Dự báo {horizon} phiên tới")
fut = res["future"]
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(res["recent"].index, res["recent"].values, label="Giá thật (90 phiên gần nhất)")
# nối điểm cuối thực tế với đường dự báo cho liền mạch
xs = [res["last_date"], *fut.index]
ys = [res["last_close"], *fut.values]
ax.plot(xs, ys, marker="o", color="tab:orange", label=f"{horizon} phiên tới")
ax.legend()
fig.autofmt_xdate()
st.pyplot(fig)

table = fut.rename("Giá đóng cửa dự báo").to_frame()
table.index = table.index.strftime("%d/%m/%Y")
table.index.name = "Ngày"
st.dataframe(table.style.format("{:,.2f}"), width="stretch")

st.caption(
    "⚠️ Dự báo đệ quy nhiều bước sẽ tích lũy sai số, và danh sách ngày chưa loại trừ ngày lễ. "
    "Chỉ mang tính tham khảo, không phải khuyến nghị đầu tư."
)
