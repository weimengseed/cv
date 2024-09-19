import cv2
import os
import math
import numpy as np


def filter_contours(img_copy, contours, mode,
                    min_area=0, min_arclength=0,
                    max_area=float('inf'), max_arclength=float('inf'),
                    draw_contours=True, print_filter_results=False,
                    padding=False, padding_color=(0, 0, 0), bin_img=None):
    '''
    根据面积和周长过滤检测到的轮廓
    :param img_copy: 在此图像上画轮廓
    :param contours: 轮廓
    :param min_area: 面积阈值，过滤面积小于该阈值的轮廓
    :param min_arclength: 周长阈值，过滤周长小于该阈值的轮廓
    :param max_arclength: 周长阈值，过滤周长大于该阈值的轮廓
    :param max_area: 面积阈值，过滤面积大于该阈值的轮廓
    :param draw_contours: 是否在原图上绘制轮廓
    :param padding: 是否在过滤的轮廓中填充
    :param padding_color: 填充的颜色
    :param bin_img: 二值化图像
    :return: 经过过滤后的轮廓数量和过滤后的轮廓
    '''
    # count = 0
    res = []
    filtered_cnts = []
    for i, cnt in enumerate(contours):         #遍历轮廓
        area = cv2.contourArea(cnt)            #计算轮廓面积
        arclength = cv2.arcLength(cnt, closed=True)  #计算轮廓周长
        if mode == 'min':
            if area < min_area or arclength < min_arclength:
                if padding:
                    cv2.drawContours(bin_img, contours, i, padding_color, -1)   #画轮廓(在哪幅图上画，轮廓索引(画所有轮廓是-1)，轮廓颜色，厚度)
                continue
        elif mode == 'max':
            if area > max_area or arclength > max_arclength:
                if padding:
                    cv2.drawContours(bin_img, contours, i, padding_color, -1)
                continue
        else:
            raise Exception("mode参数错误，请输入max或min")
        filtered_cnts.append(cnt)
        if draw_contours:
            cv2.drawContours(img_copy, contours, i, (0, 0, 255), 2)
        # count += 1
        res.append((area, arclength))
    count = len(res)
    if print_filter_results:
        print(res)
    return count, filtered_cnts



def read_image(img_path, resize=False, scale=0.5):
    '''
    使用openCV读取图片
    :param img_path: 读取的图片路径
    :param resize: 是否缩放
    :param scale: 等比例缩放比例
    :return: 图片对象
    '''
    image = cv2.imread(img_path) #读取照片
    wid = image.shape[0] #shape读取图像的属性 image.shape()返回一个元组，(100,100,3)分别表示图像的长、宽、通道数(彩色是3，灰度是1)
    hei = image.shape[1]
    #print(wid,hei)
    image=image[50:wid-50,50:hei-50]

    # image=image[]
    if resize:
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)  #cv2.resize(图片，dsize图片尺寸，水平方向的缩放系数，数值方向上的缩放系数)
    return image


def show_image(win_name, img_obj):
    '''
    使用openCV展示图片
    :param img_obj: openCV图片对象
    :return:
    '''
    cv2.imshow(win_name, img_obj)   #在窗口打印图片，cv2.imshow(窗口名称，图片对象)
    cv2.waitKey()
    cv2.destroyAllWindows()


def write_image(save_dir, image_path, img_obj, mark, resize=False, scale=1):
    '''
    保存图像
    :param save_dir: 将图片保存到这个目录下
    :param image_path: 图片路径
    :param img_obj: 图片对象
    :param mark: 标明图像类别
    :param resize: 是否等比例缩放
    :param scale: 缩放比例
    :return:
    '''
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    image_dir, image_name = image_path.split(os.sep)[:-1], image_path.split(os.sep)[-1]
    image_name = image_name.split('.')[0]
    new_name = image_name + mark + '.jpg'
    save_path = os.path.join(save_dir, new_name)
    if resize:
        img_obj = cv2.resize(img_obj, (0, 0), fx=scale, fy=scale)
    cv2.imwrite(save_path, img_obj)
    print('图片{}保存完成！'.format(save_path))


def binary_image(img_obj, mode=cv2.THRESH_BINARY, thresh_hold=125, max_value=255, blur=True, kernel_size=(5, 5),
                 sigma=5):
    '''
    将openCV图片对象转灰度图，高斯模糊（可选），二值化
    :param img_obj:图片对象
    :param thresh_hold:二值化阈值
    :param max_value:超过二值化阈值的值置为该值
    :param blur:是否高斯模糊处理
    :param kernel_size:高斯模糊卷积核大小
    :param sigma:高斯模糊强度
    :return:二值化后的图像
    '''
    gray = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)   #图片的灰度处理
    if blur:
        gray = cv2.GaussianBlur(gray, kernel_size, sigma)  #cv2.GaussianBlur进行高斯模糊
    _, binaried = cv2.threshold(gray, thresh_hold, max_value, mode)  # 二值化处理
    return binaried


def extra_color(img, lowerb, upperb):
    '''
    提取颜色
    :param img: 图片对象
    :param lowerb: 低阈值
    :param upperb: 高阈值
    :return: 对应颜色区域的二值化图
    '''
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   #将图片进行颜色空间转换，从RGB转换成HSV
    pos = cv2.inRange(hsv, lowerb, upperb)   #设置阈值，去除背景部分，低于lowerb和高于upperb的图像值变为0，在这之间的变为225
    return pos


def is_white(arr):
    '''
    判断一个区域内是否有轮廓
    :param arr: 数组
    :return:
    '''
    for r in arr:
        for c in r:
            if c == 255:
                return True
    return False


def draw_fruit_contours(img_copy, color_bin, total_cnts, height, width):
    '''
    将指定颜色的果实轮廓绘制在图片上
    :param img_copy: 将轮廓绘制在该图上
    :param color_bin: 提取颜色后得到的二值化图
    :param total_cnts: 所有果实的轮廓
    :param height: 图片高度
    :param width: 图片宽度
    :return:
    '''
    for i, cnt in enumerate(total_cnts):
        (x, y), (w, h), rotate = cv2.minAreaRect(cnt)  # 返回值 (center(x,y), (width, height), angle of rotation ) 返回中心点坐标，宽度高度，角度
        # x, y = int(x), int(y)
        # w, h = int(w), int(h)
        left, right = max(0, int(x - w / 2)), min(width, int(x + w / 2))
        bottom, top = max(0, int(y - h / 2)), min(height, int(y + h / 2))
        rec = color_bin[bottom: top, left: right]
        if is_white(rec):
            cv2.drawContours(img_copy, total_cnts, i, (0, 0, 255), 2) #轮廓填充，cv2.drawContours(目标图像，轮廓组，指明画第几个轮廓，轮廓颜色，线宽)
            # show_image('img_copy', img_copy)


def find_top(cnt):
    '''
    得到轮廓上顶点
    :param cnt: 单个轮廓
    :return: 轮廓上顶点的坐标
    '''
    x, y = cnt[0][0]
    for p in cnt:
        i, j = p[0][0], p[0][1]
        if j < y:
            x, y = i, j
    return x, y


def find_bottom(cnt):
    '''
    得到轮廓底部的顶点
    :param cnt: 单个轮廓
    :return: 底部顶点坐标
    '''
    x, y = cnt[0][0]
    for p in cnt:
        i, j = p[0][0], p[0][1]
        if j > y:
            x, y = i, j
    return x, y


def find_left(cnt):
    '''
    得到最左边的点
    :param cnt: 单个轮廓
    :return: 最左边点的坐标
    '''
    x, y = cnt[0][0]
    for p in cnt:
        i, j = p[0][0], p[0][1]
        if i < x:
            x, y = i, j
    return x, y


def find_right(cnt):
    '''
    得到最右边的点
    :param cnt: 单个轮廓
    :return: 最右边的点坐标
    '''
    x, y = cnt[0][0]
    for p in cnt:
        i, j = p[0][0], p[0][1]
        if i > x:
            x, y = i, j
    return x, y


def get_middle_point(p1, p2):
    '''
    得到两点的中心点
    :param p1: 点1
    :param p2: 点2
    :return: 中心点坐标
    '''
    new_x = int((p1[0] + p2[0]) / 2)
    new_y = int((p1[1] + p2[1]) / 2)
    return new_x, new_y


def get_distance(points):
    '''
    计算距离
    :param points: 列表，每个元素为一个坐标点
    :return: 距离
    '''
    ll = len(points)
    dis = 0
    for i in range(0, ll - 1):
        delta_x = points[i][0] - points[i + 1][0]
        delta_y = points[i][1] - points[i + 1][1]
        dis += math.sqrt(delta_x ** 2 + delta_y ** 2)
    return dis


def px2mm(img_path, resize=False, scale=1):
    '''
    将像素单位转换为毫米单位
    :param img_path: 用于单位转换的图片
    :param resize: 是否缩放
    :param scale: 缩放比例
    :return: 1像素为多少毫米
    '''
    ds = []
    img = read_image(img_path, resize=resize, scale=scale)
    bin_img = binary_image(img, thresh_hold=50, mode=cv2.THRESH_BINARY_INV, blur=False)
    cnts, hie = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in cnts:
        left_point = find_left(cnt)
        right_point = find_right(cnt)
        bottom_point = find_bottom(cnt)
        top_point = find_top(cnt)
        ds.append(get_distance([left_point, right_point]))
        ds.append(get_distance([bottom_point, top_point]))
    mean_px = np.array(ds).mean()
    return 20 / mean_px
    # show_image('bin', bin_img)


def sort_contours(cnts, start=None, end=None):   #对轮廓进行排序，一张图上有多个轮廓，提取的轮廓是乱序的
    first_row = []                               #可以使用imutils中的contours模块中的 sort_contours进行排序
    second_row = []
    results = {}
    for i, cnt in enumerate(cnts):
        cur_y = cnt[0][0][1]
        if cur_y < 300:
            first_row.append(cnt)
        else:
            second_row.append(cnt)
    first = sorted(first_row, key=lambda x: x[0][0][0])
    second = sorted(second_row, key=lambda x: x[0][0][0])
    first.extend(second)
    return first



def get_stem_length(cnt):
    '''
    根据单个茎的轮廓计算茎的长度
    :param cnt: list，单个茎的轮廓
    :return: int，单个茎的长度；list[(x1, y1), (x2, y2)...]，模拟茎长度的一系列坐标点
    '''
    mps = []
    ll = len(cnt) // 2
    left = cnt[: ll][::-1]
    right = cnt[ll: ll * 2]
    for i in range(len(left)):
        lp = left[i][0]
        rp = right[i][0]
        mps.append(get_middle_point(lp, rp))
    return get_distance(mps), mps


def draw_line(img, points):
    '''
    在图片上绘制曲线（直线）
    :param img: 将曲线（直线）绘制在该图片上
    :param points: list[(x1, y1), (x2, y2)...]，坐标点
    :return:
    '''
    ll = len(points)
    if ll < 2:
        raise Exception("少于两个点，无法绘制曲线")
    for i in range(ll - 1):
        cv2.line(img, points[i], points[i + 1], (0, 0, 255), 2)    #画线（图片，起始点坐标，终点坐标，颜色，线的粗细）


def sort_and_split_contours(contours):  
    # 获取每个轮廓的外接矩形  
    bounding_boxes = [cv2.boundingRect(c) for c in contours]  
    
    # 将轮廓与其边界框结合  
    contours_with_boxes = list(zip(contours, bounding_boxes))  
    
    # 根据y坐标排序  
    contours_with_boxes.sort(key=lambda b: b[1][1])  # b[1][1]是y坐标  

    # 将轮廓划分成两行  
    rows = [[]]  
    last_center_y = -1  

    for i, (contour, rect) in enumerate(contours_with_boxes):  
    # 计算轮廓的中心点的y坐标  
        center_y = rect[1] + rect[3] / 2  # rect[1]是y坐标，rect[3]是高度  

    # 比较当前中心点y坐标与最后一个中心点y坐标，确定是否分行  
        if last_center_y == -1 or (center_y - last_center_y > 60):  # 10是行间的阈值  
            rows.append([])  # 新开一行  

        rows[-1].append((contour, rect))  
        last_center_y = center_y  # 更新最后一个中心点y坐标

    # 对每一行进行从左到右排序  
    sorted_contours = []  
    for row in rows:  
        # 对当前行的轮廓按x坐标排序  
        row_sorted = sorted(row, key=lambda b: b[1][0])  # b[1][0]是x坐标  
        sorted_contours.extend([c[0] for c in row_sorted])  # 只保留轮廓  
        
    return sorted_contours  