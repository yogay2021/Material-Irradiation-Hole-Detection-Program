# ======================框架====================== #
# 导入测试集以及预测数据
# 测试集总数先默认算FN(理解为全部没找出来)
# 预测总数先默认算FP(理解为全部找错了)
# 嵌套循环，每一个预测数据与所有的测试集求iou，取出max
# max与设置的iou阈值进行比较，判正一个 +TP -FP -FN
# 计算precision, recall, F1-score
# =============================================== #
import os
import json
import math


def cal_iou(l_pre, l_test):
    # 提取圆心坐标和半径
    x1, y1, r1 = l_pre[0], l_pre[1], l_pre[2]
    x2, y2, r2 = l_test[0], l_test[1], l_test[2]

    # 计算圆心距离
    d = float(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    # 判断两个圆之间的位置状态
    # 两个圆不相交
    if r1 + r2 <= d:
        iou = 0
        return iou

    # 1包裹2
    if r1 - r2 >= d:
        intersection_area = math.pi * r2 * r2
        union_area = math.pi * r1 * r1
        iou = intersection_area / union_area
        return iou

    # 2包裹1
    if r2 - r1 >= d:
        intersection_area = math.pi * r1 * r1
        union_area = math.pi * r2 * r2
        iou = intersection_area / union_area
        return iou

    # 两个圆相交
    else:
        # 计算交集面积
        if 2 * d * r1 == 0:
            print(l_pre, l_test)
        angle1 = math.acos((r1 * r1 + d * d - r2 * r2 ) / (2 * d * r1))
        angle2 = math.acos((r2 * r2 + d * d - r1 * r1) / (2 * d * r2))
        intersection_area = r1 * r1 * angle1 + r2 * r2 * angle2 - math.sin(angle1) * r1 * d
        # 计算并集面积
        union_area = math.pi * r1 ** 2 + math.pi * r2 ** 2 - intersection_area
        # 计算交并比
        iou = intersection_area / union_area
        return iou


test_datapath = 'E:/DeskTop/photo/bubble/small/point_data/6_test.txt'  # 测试集路径
predict_datapath = 'E:/DeskTop/photo/bubble/small/point_data/6_pred.txt'  # 预测数据路径

# 读入数据并转成列表存储
data_tests = []  # 测试集数据列表
with open(test_datapath, "r") as file:
    data_test = file.read().splitlines()
for num_data1 in range(len(data_test)):
    data_tests.append(json.loads(data_test[num_data1]))

data_pres = []  # 预测数据列表
with open(predict_datapath, "r") as file:
    data_pre = file.read().splitlines()
for num_data2 in range(len(data_pre)):
    data_pres.append(json.loads(data_pre[num_data2]))

# 定义FN FP TP iouthres
FN = len(data_tests)
FP = len(data_pres)
TP = 0
iouthres = 0.33

# 求取max iou
for num_pre in range(len(data_pres)):
    iou_max = 0
    for num_test in range(len(data_tests)):
        ioux = cal_iou(data_pres[num_pre], data_tests[num_test])  # 计算每一个predict的iou
        if ioux > iou_max:
            iou_max = ioux
    # 阈值判定
    if iou_max > iouthres:
        TP = TP + 1
        FN = FN - 1
        FP = FP - 1
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * precision * recall / (precision + recall)
print("precision = {}\n".format(precision),"recall = {}\n".format(recall),"f1-score = {}\n".format(F1))
with open("E:/DeskTop/photo/bubble/small/f1/6_f1.txt", "w") as file:
    file.write("precision = {}\n".format(precision))
    file.write("recall = {}\n".format(recall))
    file.write("f1-score = {}\n".format(F1))
