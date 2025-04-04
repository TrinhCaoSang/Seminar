import pandas as pd
import numpy as np
import json
import os
from lime.lime_tabular import LimeTabularExplainer

# Kiểm tra file dữ liệu
file_path = "perturbed_data.csv"
if not os.path.exists(file_path):
    print("❌ Lỗi: File perturbed_data.csv không tồn tại!")
    exit(1)

# Đọc dữ liệu từ file
try:
    df = pd.read_csv(file_path)
    X = np.array([json.loads(state) for state in df["state"]], dtype=object)
    y = np.array([json.loads(probs) for probs in df["probs"]], dtype=object)
except Exception as e:
    print(f"❌ Lỗi khi đọc dữ liệu: {e}")
    exit(1)

# Chuyển đổi dữ liệu về dạng phù hợp
X = X.reshape(X.shape[0], -1)
if len(y.shape) > 1:
    y = y[:, 0]  # Chỉ lấy cột đầu tiên nếu y có nhiều chiều

# Kiểm tra dữ liệu đầu vào
print(f"✅ Dữ liệu đầu vào: X.shape = {X.shape}, y.shape = {y.shape}")

# Kiểm tra xem y có toàn số 0 không
y_zero_ratio = np.mean(y == 0)
print(f"🟡 Tỷ lệ y = 0: {y_zero_ratio * 100:.2f}%")
if y_zero_ratio > 0.95:
    print("⚠️ Cảnh báo: Quá nhiều giá trị y = 0, có thể có lỗi trong dữ liệu!")

# Khởi tạo LIME Explainer
explainer = LimeTabularExplainer(
    X, mode="regression", feature_names=[f"cell_{i}" for i in range(X.shape[1])], discretize_continuous=False
)

# Chọn một mẫu ngẫu nhiên để giải thích
idx = np.random.randint(0, len(X))
print(f"🔍 Đang giải thích mẫu {idx}")

# Kiểm tra giá trị mẫu
print(f"X[idx]: {X[idx][:10]}")  # In 10 giá trị đầu để tránh quá dài
print(f"y[idx]: {y[idx]}")

# Hàm dự đoán mẫu (CẦN THAY BẰNG MÔ HÌNH THẬT nếu có)
def model_predict(x):
    predictions = np.array([y[idx]] * len(x))
    print(f"📊 Dự đoán cho {len(x)} mẫu: {predictions[:5]}")  # In thử 5 giá trị đầu
    return predictions

# Giải thích mẫu
exp = explainer.explain_instance(X[idx], model_predict, num_features=40)

# Xuất kết quả ra file HTML
exp.save_to_file("lime_explanation.html")

print("✅ Hoàn thành! Kết quả đã được lưu vào lime_explanation.html")