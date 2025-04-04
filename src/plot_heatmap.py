import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Tránh lỗi giao diện đồ họa khi hiển thị

import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ JSON
file_path = "mcts_data.json"
try:
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    print("✅ Dữ liệu JSON hợp lệ!")
    print("🔍 Dữ liệu đầu tiên:", data[0])  # Kiểm tra cấu trúc dữ liệu
except json.JSONDecodeError as e:
    print("❌ Lỗi JSON:", e)
    exit()
except IndexError:
    print("❌ Lỗi: File JSON trống!")
    exit()

# Kích thước bàn cờ (chỉnh sửa nếu game của bạn dùng kích thước khác)
board_size = 10  # Thay đổi nếu game không dùng 10x10

# Tạo heatmap ban đầu với giá trị 0
heatmap = np.zeros((board_size, board_size))

# Xây dựng heatmap từ "actions" và "probs"
for entry in data:
    if "actions" in entry and "probs" in entry:
        for action, prob in zip(entry["actions"], entry["probs"]):
            x, y = divmod(action, board_size)  # Chuyển chỉ số 1D thành tọa độ 2D
            heatmap[x, y] += prob  # Cộng dồn xác suất từ nhiều lần chạy MCTS

# Chuẩn hóa heatmap về khoảng [0,1] để tránh bias
if np.max(heatmap) > 0:
    heatmap /= np.max(heatmap)

# Vẽ heatmap
plt.figure(figsize=(8, 6))
ax = sns.heatmap(heatmap, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")
plt.title("Heatmap Ảnh Hưởng Của MCTS")
plt.xlabel("Cột")
plt.ylabel("Hàng")

# Lưu heatmap thành file ảnh
plt.savefig("heatmap.png")
print("✅ Heatmap đã lưu thành file `heatmap.png`")
