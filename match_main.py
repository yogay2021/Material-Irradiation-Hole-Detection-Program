'''
框架主体
sauvola二值化，霍夫变换，模板匹配
初步的确定孔洞位置
'''
# 导入所需库
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola
import math
from tqdm import tqdm
import statistics

# 使用sauvola算法进行图像二值化处理
def findcontours_sobel(imagex):
    # 高斯滤波，滤除部分噪声
    patch = cv2.GaussianBlur(imagex,(5,5),0)

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
    return patch_new

# 霍夫变换检测到的圆，返回圆心半径
def hough_circle_detection(edgeimg,radrange):
    # 霍夫变换圆形检测
    circles1 = cv2.HoughCircles(edgeimg,cv2.HOUGH_GRADIENT,1,27,param1=100,param2=10,minRadius=radrange[0],maxRadius=radrange[1])
    # 找到的所有圆的圆心，半径
    template_circle = np.round(circles1[0,:]).astype("int")
    return template_circle

# 计算所有圆的平均半径
def avgrad_circle(circlelist):
    sumrad = 0  # 初始化所有圆半径的总和为零
    for numcircle0 in range(len(circlelist)):
        # 按顺序取出其中每一个圆
        template0 = circlelist[numcircle0]
        # 提取圆的半径
        template0_rad = template0[2]
        # 求取半径总和
        sumrad = sumrad + template0_rad
    # 计算圆形的平均半径
    aver_rad = sumrad / len(circlelist)
    # 有多少个圆
    numyuan = len(circlelist)
    return aver_rad, numyuan

# 从原图中获取每个圆的模板图像，并做筛选 ,返回模板列表，模板的h，w
def template_img(circlelist,img_ori,avg_rad,expend_r):
    template_list = []  # 霍夫圆模板列表
    for numcircle in range(len(circlelist)):

        template1 = circlelist[numcircle]  # 按顺序取出其中每一个圆
        # 提取圆的圆心坐标，半径
        template1_x = template1[0]
        template1_y = template1[1]

        # 在原图上截取模板
        template = img_ori[template1_y-(int(avg_rad)+expend_r+0):template1_y+(int(avg_rad)+expend_r),
                           template1_x-(int(avg_rad)+expend_r+0):template1_x+(int(avg_rad)+expend_r)] #(y1:y2, x1:x2)
        h,w = template.shape[:2]

        # 去除大小明显离群的模板
        if h == (2*int(expend_r)+2*int(avg_rad)) and w == (2*int(expend_r)+2*int(avg_rad)):
            template_list.append(template)
        else:
            continue
    return template_list,h,w

# 计算出所有的圆心模板在原图中的匹配值矩阵，返回平均值矩阵
# 这样的处理方法并不完善，只能找到和大多数模板匹配的气泡，也就是说对基本难例无效。
def cul_match(tem_list,img_ori):
    # 进行模板匹配，返回值match表征匹配度
    match_list = []  # 建立列表存储每个模板的全图match值矩阵
    # 遍历所有的模板，存储每一个模板在全图的match值矩阵到列表中
    for tem_num in range(len(tem_list)):
        templatex = tem_list[tem_num]
        match = cv2.matchTemplate(img_ori,templatex,cv2.TM_CCOEFF_NORMED)
        match_list.append(match)
    # 求取所有模板的match矩阵元素一一对应的平均值矩阵，这样就运用了所有的圆模板
    stack_match = np.stack(match_list,axis=-1)  # 堆叠
    avg_match = np.mean(stack_match,axis=-1)  # 求平均
    return avg_match

# 计算出满足阈值的点在原图中的坐标
def cul_xy(thresh_matchx,avg_matchx):
    # 记录满足阈值的坐标
    location_match = np.where(avg_matchx>thresh_matchx)
    # print("match均值",np.mean(avg_matchx))
    loc = zip(location_match[::-1])
    x,y = list(loc)
    # 记录坐标
    list_xy = []
    # 计算出满足阈值点的数量
    for point_num in range(len(x[0])):
        list_xy.append((x[0][point_num], y[0][point_num]))  # 坐标保存在列表中
    # list_xy[x][y]表示第x个坐标，y或0或1表示横坐标或纵坐标

    return list_xy,x,y

# 去除重复的密集框
# 思路很原始，间距小于半径的判断重复，nms会是更优解
def delete_box(xa,list_xya,avg_radx):
    list_box = []  # 存储没重复的框
    for j in range(len(xa[0])):
        overlap = False # 定义第j个框的重复状态
        # 计算框与框之间的距离
        for l in range(j+1,len(xa[0])):
            x_pow2 = pow(list_xya[j][0]-list_xya[l][0],2)
            y_pow2 = pow(list_xya[j][1]-list_xya[l][1],2)
            dis_box = pow(x_pow2 + y_pow2, 0.5)
            if dis_box < avg_radx:  # 判定存在重复，直接退出，进入下一个框
                overlap = True
                break
        if not overlap:  # 不存在重复的就存入列表
            list_box.append(list_xy[j])
    return list_box

# 将输入的完整大图平均地分割为若干张，
# 局限性较大，没有做padding，边界上会丢失完整的圆，切割越多影响越大。
# 边界点坐标处理的很僵硬，容易报错
def split_image(image, num):
    # 获取图像尺寸
    widthx, heightx = image.shape[:2]
    # 计算每个子图的宽度和高度
    sub_width = math.ceil(widthx / num)
    sub_height = math.ceil(heightx / num)
    # 定义子图列表
    sub_images = []
    # 定义子图坐标列表
    point_images = []
    # 循环切割子图
    for i in range(num):
        for j in range(num):
            # 计算当前子图的左上角坐标和右下角坐标
            left = i * sub_width
            top = j * sub_height
            right = min(left + sub_width, widthx)
            bottom = min(top + sub_height, heightx)
            point_images.append((top,bottom,left,right))
            # 切割子图并添加到列表
            sub_image = image[top:bottom,left:right] # (y1:y2, x1:x2)
            sub_images.append(sub_image)
    # 返回子图列表
    return sub_images,point_images,widthx,heightx

# 将切割后处理完的图像合并成原图，和split搭配使用
def merge_image(parts,point_images,heighta,widtha,imgx):
    # 创建新图像
    # merged_image = np.zeros((widtha,heighta,3), dtype=np.uint8)
    imgx = cv2.cvtColor(imgx, cv2.COLOR_GRAY2RGB)
    merged_image = np.zeros_like(imgx)
    # 合并图像
    for m, part in enumerate(parts):
        merged_image[point_images[m][0]:point_images[m][1],point_images[m][2]:point_images[m][3]] = part
    # 返回还原后的原大小图
    return merged_image

#
def update_template(img,boxlist,h,w):
    template_update = []
    for boxnum in range(len(boxlist)):
        # (y1:y2, x1:x2)
        template_up = img[boxlist[boxnum][1]:boxlist[boxnum][1]+h,
                      boxlist[boxnum][0]:boxlist[boxnum][0]+w]
        template_update.append(template_up)
    return template_update

# 迭代更新筛选模板的阈值
def update_th(matchx,numyuanx):
    # 求出较为稳定的阈值，用于后续筛选最有效的模板
    avg_match11 = matchx
    for i in range(len(matchx)):
        for j in range(len(matchx[i])):
            avg_match11[i][j] = '{:.5f}'.format(matchx[i][j])
    avg_match11 = [element for row in avg_match11 for element in row]
    # 排序
    avg_match11.sort()
    # 获取列表的长度
    n = len(avg_match11)
    # 求得最优的阈值
    best_th = avg_match11[-numyuanx]
    # print("更新阈值={}".format(best_th))
    return best_th



if __name__ == "__main__":

    '''
    本框架的超参数：
    1.将截取的气泡模板拓展的范围大小expendr（已经优化）
    2.设置每行、列切割的数量num_split,1表示不切割
    3.设置霍夫变换寻找圆的半径阈值范围radrange
    4.findrange在此处更像是一个补丁，匹配阈值的自动寻找往往确定的过于严格了，
      输出的结果丢失了一定数量的对象，需要适当的通过这个参数来多找几组目标。
    '''

    expendr = 0  # 无需修改
    num_split = 1
    radrange = (10, 35)
    findrange = -25

    # 计算运行时间
    t1 = cv2.getTickCount()

    # 导入图像数据
    path = 'E:\\DeskTop\\photo\\bubble\\dataset\\train\\images\\bf010_4.jpg'
    xml_path = "E:\\DeskTop\\photo\\bubble\\dataset\\train\\labels\\bf010_4.txt"
    img_ori = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img_ori = cv2.resize(img_ori,(800,800))
    # print("图像的shape{}".format(img_ori.shape[:2]))
    sizeimg = img_ori.shape
    width_img = sizeimg[0]
    height_img = sizeimg[1]

    #将图像平均切割成若干份
    sub_images,point_images,width_ori,height_ori = split_image(img_ori,num_split)

    imagergb_list = []

    num_bubble = 0  # 初始化气泡数量
    bubble_center = []  # 定义圆形坐标列表
    #遍历处理子图
    for numx in tqdm(range(len(sub_images))):
        img = sub_images[numx]
        image = Image.fromarray(img)
        image = np.array(image)

        #sobel算子计算边缘，返回二值化处理结果
        patch_new = findcontours_sobel(image)
        # cv2.imshow("1",patch_new)
        # cv2.waitKey(0)

        # 霍夫变换检测圆，返回存储每个圆信息的列表
        template_circle = hough_circle_detection(patch_new,radrange)

        #计算气泡圆平均半径
        aver_rad ,numyuan = avgrad_circle(template_circle)
        #print("平均半径{} = {}".format(numx,aver_rad))

        # 从原图中获取每个圆的模板图像，并做筛选 , 返回模板列表，模板的h,w
        template_list,h,w = template_img(template_circle, image, aver_rad, expendr)
        # cv2.imshow("1",template_list[0])
        # cv2.waitKey(0)

        #计算出所有的圆心模板在原图中的匹配值矩阵，返回平均值矩阵
        avg_match = cul_match(template_list, img)


        #更新阈值，取出排序靠前的阈值
        best_th = update_th(avg_match,numyuan)

        #计算出满足match阈值的坐标
        list_xy,x,y = cul_xy(best_th, avg_match)

        #删除重复的boxes
        list_box = delete_box(x, list_xy, aver_rad)

        #循环进行两次模板迭代********************************
        for upnum in range(1):
            # 更新模板---把筛选过后的模板取出，存入列表
            template_update = update_template(img,list_box,h,w)

            # 计算出所有的圆心模板在原图中的匹配值矩阵，返回平均值矩阵
            avg_match = cul_match(template_update,img)

            best_th = update_th(avg_match,numyuan)
            # 计算出满足match阈值的坐标
            list_xy, x, y = cul_xy(best_th, avg_match)

            # 删除重复的boxes
            list_box = delete_box(x,list_xy,aver_rad)
        while( len(list_box) < numyuan + (findrange)):
            best_th = best_th - 0.001  # 迭代降低阈值，直到找出所有圆（理论数量）
            print("best_threshold",best_th)
            # 计算出满足match阈值的坐标
            list_xy, x, y = cul_xy(best_th, avg_match)
            # 删除重复的boxes
            list_box = delete_box(x, list_xy, aver_rad)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 绘制标注框
        for ia in range(len(list_box)):
            num_bubble = num_bubble + 1
            bottom_right = (list_box[ia][0] + w, list_box[ia][1] + h)
            # cv2.rectangle(img, (list_box[ia][0], list_box[ia][1]), bottom_right, (128, 128 , 255), 2)
            radpointx = list_box[ia][0] + int(w/2)
            radpointy = list_box[ia][1] + int(h/2)

            # img_rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            bubble_center.append([radpointx, radpointy, aver_rad])
            cv2.circle(img_rgb,(radpointx,radpointy),int(aver_rad),(0,0,255),2)


        imagergb_list.append(img_rgb)

    # dstimg = merge_image(sub_images,point_images,height_ori,width_ori,img)
    dstimg = merge_image(imagergb_list, point_images, height_ori, width_ori,img_ori)


    # 打开文件并写入数据
    with open(xml_path, "w") as file:
        for x in range(len(bubble_center)):
            px = max(0, bubble_center[x][0]/width_img)
            py = max(0, bubble_center[x][1]/height_img)
            pw = max(0, 2 * bubble_center[x][2]/width_img)
            ph = max(0, 2 * bubble_center[x][2]/height_img)
            file.write('0' + ' ' + str(px) + ' ' + str(py) + ' ' + str(pw) + ' ' + str(ph) +"\n")

    # 计算时间
    t2 = cv2.getTickCount()
    t_demo = (t2 - t1) / cv2.getTickFrequency()
    print("程序耗时 = {}s".format(t_demo))


    cv2.namedWindow("img",0)
    cv2.resizeWindow("img",600,600)
    cv2.imshow('img',dstimg)
    cv2.waitKey(0)

    # print(bubble_center)
    # cv2.imwrite("E:\\DeskTop\\photo\\bubble\\small\\result2\\6_result.jpg", dstimg)