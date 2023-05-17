'''
绘制折线图，可视化分析
'''
import matplotlib.pyplot as plt

# 准备数据
f1 = []
ap = []
rad = []
# for i in range(len(delta_rad)):
#     f1x = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
#     f1.append(f1x)

# 绘制折线图
plt.plot(rad,ap)

# 添加标签和标题
plt.xlabel('rad')
plt.ylabel('ap')
# 显示图形
plt.show()

