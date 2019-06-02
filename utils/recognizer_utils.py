import cv2
import numpy as np


def getROI(image_np, boxes_roi, margin_ratio):
    (left, right, top, bottom) = boxes_roi[0]
    width = right - left
    height = bottom - top
    margin_x = margin_ratio * width
    margin_y = margin_ratio * height
    img_roi = image_np[
        int(top - margin_y): int(bottom + margin_y),
        int(left - margin_x): int(right + margin_x)]
    return img_roi


def drawBoxOfROI(scores_roi, boxes_roi, margin_ratio, im_width, im_height, image_np):
    try:
        # 扩大识别区域
        (left, right, top, bottom) = boxes_roi[0]  # 取第一个框
        margin_x = margin_ratio * (right - left)
        margin_y = margin_ratio * (bottom - top)
        # 越界处理
        left = int(left - margin_x) if int(left - margin_x) >= 0 else 0
        top = int(top - margin_y) if int(top - margin_y) >= 0 else 0
        right = int(right + margin_x) if int(right + margin_x) <= im_width else im_width
        bottom = int(bottom + margin_y) if int(bottom + margin_y) <= im_height else im_height
        p1 = (left, top)
        p2 = (right, bottom)
        cv2.rectangle(image_np, p1, p2, (0, 0, 255), 2, 1)
        # 给定返回值
        b_have_hand = True
        image_roi = image_np[top:bottom, left:right]
        # 显示得分
        cv2.putText(image_np, str(float('%.2f' % scores_roi[0])),
                    (int(left), int(top)-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 0), 2)
        return b_have_hand, image_roi
    except Exception as e:
        b_have_hand = False
        image_roi = np.zeros((150, 150, 3), np.uint8)
        return b_have_hand, image_roi


def myContList(find_cont, area_ignore):
    l_cont = find_cont # 输入的轮廓列表
    a_min = area_ignore # 判定去除的最小面积
    c_big = []
    for i in range(len(l_cont)):
        cont = l_cont[i]
        a = cv2.contourArea(cont)
        if a > a_min: c_big.append(cont) # 留下符合大小的轮廓们
    return c_big


def extractHand(image_np):
    # 肤色分割
    image_ycrcb = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
    (ch_y, ch_cr, ch_cb) = cv2.split(image_ycrcb)
    ch_cr = cv2.GaussianBlur(ch_cr, (5, 5), 1)
    _, ch_cr_bin = cv2.threshold(
        ch_cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image_bin = ch_cr_bin
    # 高斯降噪
    image_blur = cv2.GaussianBlur(image_np, (7, 7), 1)
    # 开运算
    kernel = np.ones((9, 9),np.uint8)
    image_open = cv2.morphologyEx(image_blur, cv2.MORPH_OPEN, kernel)
    # 背景去除
    list_rgb = cv2.split(image_open)
    for i in range(len(list_rgb)):
        list_rgb[i] = cv2.bitwise_and(list_rgb[i], image_bin)
    image_fore = cv2.merge(list_rgb)
    # 边缘提取
    image_canny = cv2.Canny(image_fore, 100, 200)
    return image_canny

def processROI(b_have_hand, image_np):
    if b_have_hand:
        try:
            image_extract = extractHand(image_np)
            # 找轮廓
            rows, cols, _ = image_np.shape
            cont_all, _ = cv2.findContours(image_extract,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cont_all_big = myContList(cont_all, (rows*cols)/3) # 相当于仅保留最大的轮廓
            cv2.drawContours(image_np, cont_all_big, -1, (0, 255, 0), 3)
            return image_extract
        except Exception as e:
            print('【Exception】', e)
            return image_np
    else:
        return image_np
