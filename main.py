import os
import cv2
import copy
from imutils import contours 
from utils import px2mm, read_image, show_image, write_image, binary_image, \
    filter_contours,  get_stem_length, draw_line, sort_and_split_contours #, sort_contours


class CalLength:
    def __init__(self, root, resize, scale, save_dir=None):
        self.test_img = 'qcbj_5X_5Y/5X.png'
        self.root = root
        self.resize = resize
        self.scale = scale
        if save_dir:
            self.save_dir = save_dir
        else:
            if os.path.isdir(root):
                self.save_dir = root
            else:
                self.save_dir = root.split(os.sep)[:-1][0]

    def single_img_process(self, img_path, save_dir=None, start=0, end=None, show_line=False, save_img=False):
        fl_cnts, img_copy = self.get_filtered_contours(img_path)  #返回轮廓和画了轮廓的图片
        #sorted_cnts = sort_contours(cnts=fl_cnts)                 #对轮廓进行排序
        sorted_cnts = sort_and_split_contours(contours=fl_cnts)
        #for method in ("left-to-right","top-to-bottom"):
        #(sorted_cnts, boundingBoxes) = contours.sort_contours(fl_cnts, method="left-to-right")
        # img = read_image(img_path, resize=True, scale=0.5)
        # img_c = copy.copy(img)
        # cv2.drawContours(img_c, sorted_cnts, 0, (0, 0, 255), 2)
        # show_image('img_c', img_c)


        lls = self.cal_length(sorted_cnts, img_copy, show_line)  #计算轮廓长度        
        # print(lls)
        if save_img:
            write_image(self.save_dir, img_path, img_copy, mark='_line', resize=True, scale=2)
        single_res = {}
        for key, val in lls.items():
            k = start + key
            if k in single_res:
                raise Exception('重复检测轮廓！')
            single_res[k] = val
        return single_res

    #def cal_length(self, fl_cnts, img_copy, show_line):
    def cal_length(self, sorted_cnts, img_copy, show_line):
        lls = {}
        #sorted_cnts = sort_contours(fl_cnts)
        for i, cnt in enumerate(sorted_cnts):
            length, mps = get_stem_length(cnt)
            length *= MM_PER_PX
            lls[i] = length
            if show_line:
                draw_line(img_copy, mps)
        return lls

    def get_filtered_contours(self, img_path):
        img = read_image(img_path, self.resize, self.scale)
        img_copy = copy.copy(img)
        bin_img = binary_image(img, thresh_hold=150, mode=cv2.THRESH_BINARY_INV, blur=False)   #将图片二值化处理(包括灰度处理，高斯模糊，二值化)
        opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, (3, 3), iterations=3)               #形态学操作(过滤噪点) (输入图片，模式，使用的内核，迭代次数)
        cnts, hie = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)         #轮廓检测(输入图片，轮廓模式，轮廓方法)   返回（轮廓、每条轮廓属性）
        cnt_count, fl_cnts = filter_contours(img_copy, cnts, mode='max', max_area=2000, draw_contours=False)  #过滤轮廓
        cnt_count, fl_cnts = filter_contours(img_copy, fl_cnts, mode='min', min_area=50, min_arclength=30,    #过滤轮廓
                                             draw_contours=False)
        return fl_cnts, img_copy         #返回轮廓和原图

    def write_text(self, dic_res, file_path):
        total = 0
        max_ll = 0
        min_ll = float('inf')
        with open(file_path, 'w') as fw:
            pass
        with open(file_path, 'a') as fa:
            for k, v in dic_res.items():
                total += v
                min_ll = min(min_ll, v)
                max_ll = max(max_ll, v)
                ss = '{},{:.4f}\n'.format(k, v)
                fa.write(ss)
            mean = total / len(dic_res)
            fa.write('mean,{:.4f}\n'.format(mean))
            fa.write('max,{:.4f}\n'.format(max_ll))
            fa.write('min,{:.4f}\n'.format(min_ll))
        print('文件{}写入成功！'.format(file_path))

    def run(self):               #遍历图片的函数
        if os.path.isdir(self.root):
            txt_name = 'F2_results.txt'
            txt_path = os.path.join(self.save_dir, txt_name)
            dic_total = {}
            images = os.listdir(self.root)
            for image in images:
                if image.split('.')[-1] == 'jpg':
                    if '_' not in image:
                        start = int(image.split('.')[0])
                    else:
                        start = int(image.split('_')[0])
                    image_path = os.path.join(self.root, image)
                    dic_res = self.single_img_process(image_path, save_dir=self.save_dir, start=start, show_line=True, save_img=True)
                    dic_total.update(dic_res)
                    # print(dic_total)
                    # break
            self.write_text(dic_total, txt_path)

        else:
            dic_res = self.single_img_process(self.root, save_dir=self.save_dir, show_line=True, save_img=True)
            img_name = self.root.split(os.sep)[-1].split('.')[0]
            txt_name = img_name + '.txt'
            txt_path = os.path.join(self.save_dir, txt_name)
            self.write_text(dic_res, txt_path)


def cal_mm_per_px(test_image_path, resize, scale):
    return px2mm(test_image_path, resize, scale)


if __name__ == '__main__':
    TEST_IMAGE_PATH = 'qcbj_5X_5Y/5X.png'
    MM_PER_PX = cal_mm_per_px(TEST_IMAGE_PATH, True, 0.5)
    print(MM_PER_PX)

    root = 'qcbj_8X_8Y/F2_image'
    callen = CalLength(root, resize=True, scale=0.5, save_dir='qcbj_8X_8Y')
    callen.run()
