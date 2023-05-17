import cv2
import numpy as np
from PIL import Image
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola
path='E:\\DeskTop\\photo\\bubble\\large\\sbf\\original_img\\sbf001.bmp'

img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
image = Image.fromarray(img)

image = np.array(image)
ori_patch = image.copy()
# 高斯滤波，滤除部分噪声
patch=cv2.GaussianBlur(image,(5,5),0)
# sobel算子进行气泡边缘提取
sobelx = cv2.Sobel(patch,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(patch,cv2.CV_64F,0,1,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
sobelxy =  cv2.addWeighted(sobelx,0.5,sobely,0.5,0)

# sauvola算法确定二值化阈值t_sauvola
t_sauvola = threshold_sauvola(sobelxy, window_size=13, k=0.2, r=None)
# 对图像进行二值化
patch_new = np.zeros((sobelxy.shape[0],sobelxy.shape[1]),dtype = np.uint8)
patch_new = sobelxy > t_sauvola
patch_new = patch_new * (sobelxy-t_sauvola)
patch_new = np.uint8(patch_new)

patchx = cv2.GaussianBlur(patch_new,(7,7),0)

# cv2.namedWindow("1", 0)
# cv2.resizeWindow("1", 700,700)
# cv2.namedWindow("2", 0)
# cv2.resizeWindow("2", 700,700)
# # cv2.imshow("1",patch_new)
# cv2.imshow("2",patchx)
# cv2.waitKey(0)

# 霍夫变换圆形检测
# circles1 = cv2.HoughCircles(patchx,cv2.HOUGH_GRADIENT,1,27,param1=138,param2=13,minRadius=36,maxRadius=48)
# # 在原图上绘制检测结果
# ori_patch = cv2.cvtColor(ori_patch, cv2.COLOR_GRAY2RGB)
# if circles1 is not None:
#     for i in circles1[0]:
#         ori_patch=cv2.circle(ori_patch,(int(i[0]),int(i[1])),int(i[2]),(0,0,255),3)

edges = cv2.Canny(patchx, 5, 90)
# 找到轮廓并绘制红色边缘
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# ori_patch = cv2.cvtColor(ori_patch, cv2.COLOR_GRAY2RGB)
cv2.drawContours(ori_patch, contours, -1, (0, 0, 255), 3)
img = cv2.medianBlur(ori_patch,5)
# 霍夫变换圆形检测
circles1 = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,27,param1=138,param2=13,minRadius=20,maxRadius=58)
# 在原图上绘制检测结果
cir_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

num_circle = 0
center_list = []

if circles1 is not None:
    for i in circles1[0]:
        cir_img=cv2.circle(cir_img,(int(i[0]),int(i[1])),int(i[2]),(0,0,255),3)
        center_list.append((num_circle,int(i[0]),int(i[1])))
        num_circle = num_circle + 1

    # 打开文件并写入数据
# with open("E:\\DeskTop\\photo\\bubble\\small\\1_center.txt", "w") as file:
    # for item in center_list:
        # file.write(str(item) + "\n")



cv2.namedWindow("circle", 0)
cv2.resizeWindow("circle", 700,700)
cv2.imshow('circle',edges)
# cv2.imshow('circle',image)
cv2.waitKey(0)
# cv2.imwrite("E:\\DeskTop\\photo\\bubble\\small\\result\\1_result.jpg",cir_img)