import cv2
import numpy as np
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola
from sklearn.cluster import KMeans
import os
from tqdm import tqdm


path='E:/DeskTop/photo/bubble/large/bf001.bmp'

def kmean_bubble(path):

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

    patchx=cv2.GaussianBlur(patch_new,(7,7),0)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(patchx, cv2.MORPH_CLOSE, kernel)
    patchx=cv2.GaussianBlur(closing, (7, 7), 0)

    # 将图像转换为NumPy数组
    data = np.array(patchx, dtype=np.float64)

    # 将二维数组转换为一维数组
    rows, cols = data.shape
    data = data.reshape(rows * cols, 1)

    # 使用K均值聚类算法进行聚类
    kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
    labels = kmeans.labels_

    # 将聚类结果转换为二维数组
    labels = labels.reshape(rows, cols)

    for y in range(1, labels.shape[0] - 1):
        for x in range(1, labels.shape[1] - 1):
            # 获取当前像素点的标签值
            label = labels[y, x]
            # 如果当前像素点的标签值为1，则在周围八个像素点中搜索是否有标签值为1的像素点
            if label == 1:
                has_one_neighbor = False
                for j in range(y - 1, y + 2):
                    for i in range(x - 1, x + 2):
                        if labels[j, i] == 1:
                            has_one_neighbor = True
                            break
                    if has_one_neighbor:
                        break

                # 如果周围八个像素点中没有标签值为1的像素点，则将当前像素点的标签值翻转
                if not has_one_neighbor:
                    labels[y, x] = 0

    # 将聚类结果保存为图像
    seg_img = Image.fromarray(labels.astype(np.uint8) * 255)
    # seg_img.save('E:/DeskTop/photo/bubble/large/result/test_seg.png')
    img_cv = cv2.cvtColor(np.array(seg_img), cv2.COLOR_GRAY2BGR)

    for i in range(kmeans.n_clusters):
        # 获取当前聚类的标签值
        label = 1

        # 找到当前聚类的像素点的坐标
        points = np.where(labels == label)

        # 绘制圆形
        ori_patch = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for x, y in zip(points[1], points[0]):
            cv2.circle(ori_patch, (x, y), 1, (0, 0, 255), 1)

    return ori_patch

result = kmean_bubble('E:/DeskTop/photo/bubble/large/sbf/sbf001.bmp')
cv2.namedWindow("2", 0)
cv2.resizeWindow("2", 600, 600)
cv2.imshow("2", result)
cv2.waitKey(0)

# dir = 'E:/DeskTop/photo/bubble/large'
# dst_path = 'E:/DeskTop/photo/bubble/large/result'
#
# for root, dirs, files in tqdm(os.walk(dir)):
#         for file in files:
#             file_path = os.path.join(root, file)
#             result = kmean_bubble(file_path)
#             cv2.imwrite(os.path.join(root,'result',file),result)