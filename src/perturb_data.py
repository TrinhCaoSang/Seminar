import json
import numpy as np
import random
import pandas as pd

# Load dữ liệu từ MCTS
with open("mcts_data.json", "r") as f:
    lines = f.readlines()

data = [json.loads(line) for line in lines]

perturbed_data = []
num_samples = 100  # Số mẫu nhiễu cần tạo

for d in data:
    state = np.array(d["state"])
    actions = d["actions"]
    probs = np.array(d["probs"])
    visit = d["visit_counts"]
    for _ in range(num_samples):
        noisy_state = state.copy()
        
        # 🎲 Tăng số ô bị nhiễu từ 7-10 thay vì 5-8
        for _ in range(random.randint(7, 10)):
            x, y = random.randint(0, state.shape[1] - 1), random.randint(0, state.shape[2] - 1)
            noisy_state[:, x, y] = 1  # Đặt ô cờ thành trống, đen hoặc trắng

        # 🎯 Tăng mức độ nhiễu của probs từ ±0.4 lên ±0.6
        noise_level = 0.6  
        perturbed_probs = [
            max(0, min(1, p + np.random.uniform(-noise_level, noise_level)))
            for p in probs
        ]

        perturbed_data.append({
            "state": noisy_state.tolist(),
            "actions": actions,
            "probs": perturbed_probs,
            "visit": visit,
        })

# Lưu thành DataFrame
df = pd.DataFrame(perturbed_data)
df.to_csv("perturbed_data.csv", index=False)

# Kiểm tra mức độ nhiễu
nonzero_probs = sum(np.count_nonzero(p) for p in df["probs"])
total_probs = sum(len(p) for p in df["probs"])
nonzero_percentage = (nonzero_probs / total_probs) * 100

if nonzero_percentage < 10:
    print(f"⚠️ Cảnh báo: Chỉ {nonzero_percentage:.2f}% giá trị `probs` khác 0. Cần tăng mức nhiễu!")
else:
    print(f"✅ Dữ liệu tốt: {nonzero_percentage:.2f}% giá trị `probs` có nhiễu.")

print("✅ Đã tạo tập dữ liệu bị nhiễu xong!")
