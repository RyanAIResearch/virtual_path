# import json
# import matplotlib.pyplot as plt

# # 加载两个 JSON 文件
# with open("/home/yran1/FlZO_Vritual_Path/cos_sim_client_3.json", "r") as f1:
#     cosine_similarity_data_1 = json.load(f1)

# # with open("/home/yran1/FlZO_Vritual_Path/cos_sim_client_0_class0.json", "r") as f2:  # 第二个文件路径
# #     cosine_similarity_data_2 = json.load(f2)
# with open("/home/yran1/FlZO_Vritual_Path/evaluate_metrics_client_0.json", "r") as f2:  # 第二个文件路径
#     cosine_similarity_data_2 = json.load(f2)

# # 提取数值
# steps_1 = [data["step"] for data in cosine_similarity_data_1]
# cosine_similarity_1 = [data["cosine_similarity"] for data in cosine_similarity_data_1]

# steps_2 = [data["step"] for data in cosine_similarity_data_2]
# cosine_similarity_2 = [data["cosine_similarity"] for data in cosine_similarity_data_2]

# # 绘图
# plt.figure(figsize=(10, 6))

# # # 第一个文件的数据
# # plt.plot(steps_1, cosine_similarity_1, marker="o", label="Cosine Similarity Class 0 & Class 1", color="blue")

# # 第二个文件的数据
# plt.plot(steps_2, cosine_similarity_2, marker="s", label="Cosine Similarity Only Class 0", color="red")

# # 图表标签和标题
# plt.xlabel("Step")
# plt.ylabel("Cosine Similarity")
# plt.title("Cosine Similarity Comparison Between Two Data Distribution")
# plt.legend()
# plt.grid(True)

# # 保存图片到本地
# plt.savefig("cosine_similarity_comparison_plot.png", dpi=300)  # 保存为 PNG 格式，分辨率 300 DPI
# plt.show()
import json
import matplotlib.pyplot as plt
# /home/yran1/FlZO_Vritual_Path/cos_sim_client_0_class0_dotproduct.json
# 加载 JSON 文件
# with open("/home/yran1/FlZO_Vritual_Path/cos_sim_client_0.json", "r") as f1:
with open("/home/yran1/FlZO_Vritual_Path/agnews_norm_client0_dis2.json", "r") as f1:
    cosine_similarity_data = json.load(f1)

# 过滤数据 (仅取 step <= 100)
filtered_data = [item for item in cosine_similarity_data if item["step"] <= 100]

# 提取数值
steps = [data["step"] for data in filtered_data]
cosine_similarity = [data["norm"] for data in filtered_data]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(steps, cosine_similarity, marker="o", label="Norm", color="blue")

# 图表标签和标题
plt.xlabel("Step")
plt.ylabel("Norm")
plt.title("Norm Over Steps")
plt.legend()
plt.grid(True)

# 保存图片到本地
plt.savefig("Norm_plot_agnews_client0_dis2.png", dpi=300)  # 保存为 PNG 格式
plt.show()
