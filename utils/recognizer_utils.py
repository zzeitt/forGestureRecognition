# coding=utf-8
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
        right = int(right + margin_x) if int(right +
                                             margin_x) <= im_width else im_width
        bottom = int(bottom + margin_y) if int(bottom +
                                               margin_y) <= im_height else im_height
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
    l_cont = find_cont  # 输入的轮廓列表
    a_min = area_ignore  # 判定去除的最小面积
    c_big = []
    for i in range(len(l_cont)):
        cont = l_cont[i]
        a = cv2.contourArea(cont)
        if a > a_min:
            c_big.append(cont)  # 留下符合大小的轮廓们
    return c_big


def extractHand(rows, cols, image_np):
    # ------------ 高斯降噪 ------------ #
    image_blur = cv2.GaussianBlur(image_np, (7, 7), 1)
    # ------------ 开运算 ------------ #
    kernel = np.ones((10, 10), np.uint8)
    image_open = cv2.morphologyEx(image_blur, cv2.MORPH_OPEN, kernel)
    # ------------ 肤色分割 ------------ #
    image_ycrcb = cv2.cvtColor(image_open, cv2.COLOR_RGB2YCrCb)
    (ch_y, ch_cr, ch_cb) = cv2.split(image_ycrcb)
    ch_cr = cv2.GaussianBlur(ch_cr, (5, 5), 1)
    _, ch_cr_bin = cv2.threshold(
        ch_cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image_bin = ch_cr_bin
    # ------------ 开运算 ------------ #
    kernel = np.ones((5, 5), np.uint8)
    image_bin = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN, kernel)

    return image_bin


def processROI(b_have_hand, image_np):
    if b_have_hand:
        try:
            rows, cols, _ = image_np.shape
            image_extract = extractHand(rows, cols, image_np)
            # ------------ 轮廓提取 ------------ #
            cont_all, _ = cv2.findContours(
                image_extract, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cont_all_big = myContList(cont_all, (rows*cols)/3)  # 仅保留大轮廓
            if len(cont_all_big) > 0:
                image_extract_3ch = np.zeros((rows, cols, 3), dtype='uint8')
                cv2.drawContours(image_extract_3ch, cont_all_big, 0, (255, 255, 255), -1)
                # ------------ 凸缺陷检测 ------------ #
                cnt = cont_all_big[0]
                hull = cv2.convexHull(cnt, returnPoints=False)
                defects = cv2.convexityDefects(cnt, hull)
                num_far = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    d /= 256
                    if d > rows*0.1:
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])
                        cv2.line(image_extract_3ch, start, end, [0, 255, 0], 2)
                        cv2.circle(image_extract_3ch, far, 5, [255, 0, 0], -1)
                        num_far += 1
                if num_far >= 4:
                    str_gesture = '5'
                else:
                    str_gesture = 'NULL'

            else:
                image_extract_3ch = cv2.merge([image_extract, image_extract, image_extract])
                str_gesture = 'NULL'

            image_ret = image_extract_3ch
            # ------------ 打印手势 ------------ #
            cv2.putText(image_ret, str_gesture, (0, cols - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 255, 0), 2)
            return image_ret
        except Exception as e:
            print('【Exception】', e)
            return image_np
    else:
        return image_np
