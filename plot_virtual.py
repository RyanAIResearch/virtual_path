import json
import matplotlib.pyplot as plt

# 从 JSON 文件读取数据
file_path = "/home/yran1/FlZO_Vritual_Path/BoolQ_client_0_global_step_0.json"  # 替换为你的 JSON 文件路径
with open(file_path, "r") as json_file:
    data = json.load(json_file)

# 提取 step 和 dot_product
steps = [entry["step"] for entry in data]
dot_products = [entry["dot_product"] for entry in data]

# 绘制图表
plt.figure(figsize=(8, 5))
plt.plot(steps, dot_products, marker='o', label='Dot Product', color='orange')
plt.title('Step vs. Dot Product')
plt.xlabel('Step')
plt.ylabel('Dot Product')
plt.grid(True)
plt.legend()

# 保存图片到本地
plt.savefig("step_vs_dot_product.png", dpi=300, bbox_inches='tight')

# 显示图表
plt.show()
