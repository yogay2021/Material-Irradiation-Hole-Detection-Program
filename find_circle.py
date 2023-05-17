# ==============框架=============#
# 从标注好的图像获得未知的标注信息
# 关注红色的标注圆以及蓝色的序号标注
# 做通道相减获得纯净的圆，返回坐标半径
# ===============================#

import cv2
import numpy as np
imgori = cv2.imread("E:\\DeskTop\\photo\\bubble\\small\\test\\5.jpg")

# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
b = imgori[:,:,0]
g = imgori[:,:,1]
r = imgori[:,:,2]

back = cv2.add(b, -r)
img = np.zeros((back.shape[0], back.shape[1], 3), dtype=np.uint8)

for x in range(back.shape[0]):
    for y in range(back.shape[1]):
        img[x][y][0] = back[x][y]
        img[x][y][1] = back[x][y]
        img[x][y][2] = back[x][y]

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
kernel = np.ones((1,1),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)

erosion = cv2.cvtColor(erosion,cv2.COLOR_BGR2GRAY)
cv2.imshow('Circles', back)
cv2.waitKey(0)
circles = cv2.HoughCircles(erosion, cv2.HOUGH_GRADIENT, 1, 20,param1=80, param2=15, minRadius=1, maxRadius=20)

# 确保至少检测到一个圆
if circles is not None:
    circles = circles[0]  # 提取圆的参数
    circle_list = []    # 将检测到的圆圈绘制在原图上，并记录下其坐标和半径
    for circle in circles:
        x, y, radius = circle
        x, y, radius = int(x), int(y), int(radius)
        cv2.circle(imgori, (x, y), radius, (255, 0, 0), 1)
        print("圆心坐标：({}, {})，半径：{}".format(x, y, radius))
        circle_list.append([x,y,radius])
    with open("E:\\DeskTop\\photo\\bubble\\small\\point_data\\5_test.txt", "w") as file:
        for item in circle_list:
            file.write(str(item)+'\n')


    # 显示包含圆圈的图像
    cv2.imshow('Circles', imgori)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# cv2.imshow("x",img)
# cv2.waitKey(0)