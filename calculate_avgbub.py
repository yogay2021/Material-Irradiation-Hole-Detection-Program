'''
由导出的txt文件计算对应图中孔洞的平均直径
'''
import os
import numpy as np

# 指定要遍历的文件夹路径
folder_path = 'E:\\DeskTop\\photo\\bubble\\analysis_bubble\\sexangle\\labels'
size_img = 640

rad_list = []
# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    # 检查文件是否为txt文件
    if file_name.endswith('.txt'):
        # 拼接文件路径
        file_path = os.path.join(folder_path, file_name)
        # 打开文件并读取第一行数据
        with open(file_path, 'r') as f:
            for line in f:
                first_line = f.readline().strip()
                rad_list.append(float(first_line.split(" ")[-1]))
                print(file_name,float(first_line.split(" ")[-1]))

rad = np.mean(rad_list)  # 计算平均值
print("平均半径 = {}".format(rad))