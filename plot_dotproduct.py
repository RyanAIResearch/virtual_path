import json
import matplotlib.pyplot as plt

# 加载 JSON 文件
with open("/home/yran1/FlZO_Vritual_Path/norm_gemma_2b_AGNews_client0_global_step0.json", "r") as f1:
    cosine_similarity_data = json.load(f1)

# 过滤数据 (仅取 step <= 100)
filtered_data = [item for item in cosine_similarity_data if item["step"] <= 100]

# 提取数值
steps = [data["step"] for data in filtered_data]
cosine_similarity = [data["dot_product"] for data in filtered_data]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(steps, cosine_similarity, marker="o", label="Dot Product", color="blue")

# 图表标签和标题
plt.xlabel("Step")
plt.ylabel("Dot Product")
plt.title("Dot Product Steps")
plt.legend()
plt.grid(True)

# # 设置 y 轴刻度为 20 的单位
# min_y = int(min(cosine_similarity) // 20) * 20  # y 轴的最小值
# max_y = int(max(cosine_similarity) // 20 + 1) * 20  # y 轴的最大值
# plt.yticks(range(min_y, max_y + 20, 20))

# 保存图片到本地
plt.savefig("norm_gemma_2b_AGNews_client0_global_step0.png", dpi=300)  # 保存为 PNG 格式
plt.show()
