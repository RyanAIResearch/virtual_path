import json
import matplotlib.pyplot as plt

# 加载两个 Accuracy JSON 文件
with open("/home/yran1/FlZO_Vritual_Path/virtual_path_output/5_clients_agnews_noniid_acc_cl5.json", "r") as f1:
    accuracy_data_1 = json.load(f1)

with open("/home/yran1/FlZO_Vritual_Path/evaluate_metrics_client_3.json", "r") as f2:
    accuracy_data_2 = json.load(f2)

# 提取前 100 个 step 的数据，包括第 100 个 step
max_steps = 100
epochs_1 = [data["epoch"] for data in accuracy_data_1[:max_steps]]
accuracy_1 = [data["accuracy"] for data in accuracy_data_1[:max_steps]]

epochs_2 = [data["epoch"] for data in accuracy_data_2[:max_steps]]
accuracy_2 = [data["accuracy"] for data in accuracy_data_2[:max_steps]]

# 绘图
plt.figure(figsize=(10, 6))

# 第一个文件的数据
plt.plot(epochs_1, accuracy_1, marker="o", label="Agnews NonIID", color="blue")

# # 第二个文件的数据
# plt.plot(epochs_2, accuracy_2, marker="s", label="Accuracy Class 0 & Class 1", color="green")

# 图表标签和标题
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()
plt.grid(True)

# 保存图片到本地
plt.savefig("accuracy_01235_agnews_noniidcl5.png", dpi=300)  # 保存为 PNG 格式，分辨率 300 DPI
plt.show()
