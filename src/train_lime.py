import pandas as pd
import numpy as np
import json
import os
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import seaborn as sns
# Kiểm tra file dữ liệu
file_path = "perturbed_data.csv"
if not os.path.exists(file_path):
    print("❌ Lỗi: File perturbed_data.csv không tồn tại!")
    exit(1)

# Đọc dữ liệu từ file
try:
    df = pd.read_csv(file_path)
    X = np.array([json.loads(state) for state in df["state"]], dtype=object)
    # Chuyển đổi cột visit và probs thành số thực
    def extract_first_element(value):
        if isinstance(value, str):
            value = json.loads(value)
        return float(value[0]) if isinstance(value, list) else float(value)
    
    df["visit"] = df["visit"].apply(extract_first_element)
    df["probs"] = df["probs"].apply(extract_first_element)
    visits = df["visit"].values.reshape(-1, 1)  # Chuyển thành mảng cột
    probs = df["probs"].values.reshape(-1, 1)  # Chuyển thành mảng cột
except Exception as e:
    print(f"❌ Lỗi khi đọc dữ liệu: {e}")
    exit(1)

# Chuyển đổi dữ liệu về dạng phù hợp
print(f"✅ Dữ liệu đầu vào: X.shape = {X.shape}")
# print(f"✅ Dữ liệu đầu vào: visits = {visits}")
# print(f"✅ Dữ liệu đầu vào: probs = {probs}")
# X = X.reshape(X.shape[0], -1)
# X = X[:, 0, :, :].reshape(X.shape[0], -1)  # Lấy layer đầu tiên và phẳng hóa thành (10900, 100)
# Kiểm tra dữ liệu đầu vào
# X = np.hstack((X, visits, probs))
X = np.hstack((visits, probs))
print(f"✅ Dữ liệu đầu vào: X.shape = {X.shape}")

# Tạo danh sách tên đặc trưng
# feature_names = [f"cell_{i}" for i in range(X.shape[1] - 2)] + ["visit", "probs"]
feature_names = ["visit", "probs"]
# Khởi tạo LIME Explainer
explainer = LimeTabularExplainer(
    X, mode="regression", feature_names=feature_names, discretize_continuous=False
)

# Chọn một mẫu ngẫu nhiên để giải thích
idx = np.random.randint(0, len(X))
print(f"🔍 Đang giải thích mẫu {idx}")

# Kiểm tra giá trị mẫu
print(f"X[idx]: {X[idx][:10]}")  # In 10 giá trị đầu để tránh quá dài


# Hàm dự đoán mẫu (CẦN THAY BẰNG MÔ HÌNH THẬT nếu có)
def model_predict(x):
    print(len(x))
    predictions =  np.random.rand(len(x)) # ngau nhien tu 0-1
    # print(f"📊 Dự đoán cho {len(x)} mẫu: {predictions[:5]}")  # In thử 5 giá trị đầu
    return predictions

exp = explainer.explain_instance(X[idx], model_predict)

# Xuất kết quả ra file HTML
exp.save_to_file("lime_explanation.html")

print("✅ Hoàn thành! Kết quả đã được lưu vào lime_explanation.html")
# Vẽ biểu đồ tầm quan trọng của đặc trưng
feature_importance = sorted(exp.local_exp[1], key=lambda x: -abs(x[1]))

feature_names_sorted = [feature_names[index] for index, _ in feature_importance]
importance_values = [weight for _, weight in feature_importance]

plt.figure(figsize=(10, 6))
plt.barh(feature_names_sorted[:20], importance_values[:20], color='orange')
plt.xlabel("Tầm quan trọng")
plt.ylabel("Đặc trưng")
plt.title("Biểu đồ tầm quan trọng của đặc trưng")
plt.gca().invert_yaxis()
plt.savefig("feature_importance.png")
plt.show()
