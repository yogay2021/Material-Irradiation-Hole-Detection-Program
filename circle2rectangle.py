'''
此脚本实现将圆心半径的标注格式转换为标准的yolo矩形标注格式
'''

import cv2
import json

predict_datapath = 'E:/DeskTop/photo/bubble/small/point_data_circle/10_pred.txt'
img_path = 'E:/DeskTop/photo/bubble/small/10.jpg'
image = cv2.imread(img_path)
size_img = image.shape
w_img = size_img[0]
h_img = size_img[1]

data_pres = []  # 预测数据列表
with open(predict_datapath, "r") as file:
    data_pre = file.read().splitlines()
for num_data2 in range(len(data_pre)):
    data_pres.append(json.loads(data_pre[num_data2]))

rectangle = []
# 输出到指定的txt文件中
with open("E:\\DeskTop\\photo\\bubble\\dataset\\labels\\10.txt", "w") as file:
    for numd in range(len(data_pres)):

        px = max(0, data_pres[numd][0] / w_img)
        py = max(0, data_pres[numd][1] / h_img)
        pw = max(0, data_pres[numd][2] * 2 / w_img)
        ph = max(0, data_pres[numd][2] * 2 / h_img)

        file.write('0' + ' ' + str(px) + ' ' + str(py) + ' ' + str(pw) + ' ' + str(ph) + "\n")

# 显示带有标注框的图像
# cv2.imshow('Image with Bounding Boxes', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
