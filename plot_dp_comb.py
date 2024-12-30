import json
import matplotlib.pyplot as plt

# 定义文件路径和标签
files = [
    ("/home/yran1/FlZO_Vritual_Path/dot_product_agnews_client0_dis1.json", "Client 0"),
    ("/home/yran1/FlZO_Vritual_Path/dot_product_agnews_client0_dis2.json", "Client 0 dis 2"),
    ("/home/yran1/FlZO_Vritual_Path/dot_product_agnews_client1_dis1.json", "Client 1"),
    ("/home/yran1/FlZO_Vritual_Path/dot_product_agnews_client2_dis1.json", "Client 2"),
    ("/home/yran1/FlZO_Vritual_Path/dot_product_agnews_client8_dis1.json", "Client 8"),
]

# 初始化绘图
plt.figure(figsize=(10, 6))

# 遍历每个文件
for file_path, label in files:
    # 加载 JSON 文件
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # 过滤数据 (仅取 step <= 100)
    filtered_data = [item for item in data if item["step"] <= 100]
    
    # 提取数值
    steps = [item["step"] for item in filtered_data]
    dot_products = [item["dot_product"] for item in filtered_data]
    
    # 绘制每个文件的数据
    plt.plot(steps, dot_products, marker="o", label=label)

# 图表标签和标题
plt.xlabel("Step")
plt.ylabel("Dot Product")
plt.title("Dot Product Across Clients")
plt.legend()
plt.grid(True)

# 保存图片到本地
plt.savefig("dp_plot_agnews_all_clients.png", dpi=300)  # 保存为 PNG 格式
plt.show()
