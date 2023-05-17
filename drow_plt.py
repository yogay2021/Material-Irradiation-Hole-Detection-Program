import matplotlib.pyplot as plt

# 准备数据
# val_rad = 14.099
# delta_rad = [18.249-val_rad, 21.723-val_rad, 25.453-val_rad]
# precision = [0.809, 0.81, 0.678]
# recall = [0.769, 0.803, 0.712]
f1 = []
ap = [0.976, 0.988, 0.945, 0.944, 0.849, 0.867, 0.858, 0.957,0.957, 0.912]
rad = [27.76, 28.51, 29.72, 29.89, 31.89, 32.59, 32.79, 33.42, 33.68, 46.57]
# for i in range(len(delta_rad)):
#     f1x = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
#     f1.append(f1x)
# print(f1)


# 绘制折线图
plt.plot(rad,ap)

# 添加标签和标题
plt.xlabel('rad')
plt.ylabel('ap')
# 显示图形
plt.show()

