# ===========框架=============#
# 读取txt文件，将内容打印到图像上
# ============================#

import cv2

data_path = 'E:/DeskTop/photo/bubble/small/f1/10_f1.txt'
img_path = 'E:/DeskTop/photo/bubble/small/result2/10_result.jpg'
with open(data_path, 'r') as f:
    data = f.read().splitlines()

img = cv2.imread(img_path)
width, height, _ = img.shape[:]

for textn in range(len(data)):
    text = str(data[textn])
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = int((width - text_size[0])/2)
    text_y = int((height + text_size[1])/2 + (text_size[1]+5)*textn)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, 255, font_thickness, cv2.LINE_AA)

#显示图像
# cv2.imshow('Image with Text', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('E:/DeskTop/photo/bubble/small/result_f1/10_f1.jpg',img)
