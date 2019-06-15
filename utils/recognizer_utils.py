# -*- coding:utf-8 -*-
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


def drawBoxOfROI(scores_roi, boxes_roi, padding_ratio,
                 margin_ratio, im_width, im_height, image_np):
    try:
        # 扩大识别区域
        (left, right, top, bottom) = boxes_roi[0]  # 取第一个框
        padding_x = padding_ratio * (right - left)
        padding_y = padding_ratio * (bottom - top)
        margin_x = margin_ratio * (right - left)
        margin_y = margin_ratio * (bottom - top)
        # 越界处理
        padding_left = max(int(left - padding_x), 0)
        padding_top = max(int(top - padding_y), 0)
        padding_right = min(int(right + padding_x), im_width - 1)
        padding_bottom = min(int(bottom + padding_y), im_height - 1)
        # margin_left = max(int(left - margin_x), 0)
        margin_left = padding_left
        margin_top = max(int(top - margin_y), 0)
        # margin_right = min(int(right + margin_x), im_width - 1)
        # margin_bottom = min(int(bottom + margin_y), im_height - 1)
        margin_right = padding_right
        margin_bottom = padding_bottom
        # 给定返回值
        b_have_hand = True
        image_clone_1 = image_np.copy()
        image_clone_2 = image_np.copy()
        image_extend = image_clone_1[margin_top:margin_bottom,
                                     margin_left:margin_right]
        image_roi = image_clone_2[padding_top:padding_bottom,
                                  padding_left:padding_right]
        # 可视化
        cv2.rectangle(image_np, (margin_left, margin_top),
                      (margin_right, margin_bottom), (0, 255, 255), 2, 1)
        cv2.rectangle(image_np, (padding_left, padding_top),
                      (padding_right, padding_bottom), (0, 255, 0), 2, 1)
        cv2.putText(image_np, str(float('%.2f' % scores_roi[0])),
                    (int(margin_left), int(margin_top)-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 100), 2)
        return b_have_hand, image_roi, image_extend
    except Exception as e:
        b_have_hand = False
        image_roi = np.zeros((150, 150, 3), np.uint8)
        return b_have_hand, image_roi, image_roi


def myContList(find_cont, area_ignore):
    l_cont = find_cont
    a_min = area_ignore
    c_big = []
    for i in range(len(l_cont)):
        cont = l_cont[i]
        a = cv2.contourArea(cont)
        if a > a_min:
            c_big.append(cont)
    return c_big


def extractHand(image_np, image_np_extend):
    image_bin_list = []
    for image_iter in (image_np, image_np_extend):
        rows, cols = image_iter.shape[0:2]
        # ------------ 高斯降噪 ------------ #
        image_blur = cv2.GaussianBlur(image_iter, (7, 7), 1)
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
        # ------------ 遮挡部分 ------------ #
        cv2.rectangle(image_bin, (0, int(rows * 0.82)),
                      (cols, rows), 0, thickness=-1)
        image_bin_list.append(image_bin)

    return image_bin_list[0], image_bin_list[1]


def myEllipseFitting(diff_rate, image_np):
    rows, cols = image_np.shape[0:2]
    # ------------ 轮廓提取 ------------ #
    cont_all, _ = cv2.findContours(
        image_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_all_big = myContList(cont_all, (rows*cols)/3)
    if len(cont_all_big) <= 0:
        image_extract_3ch = cv2.merge(
            [image_np, image_np, image_np])
        b_ellipse_fit = False
    else:
        image_extract_3ch = np.zeros((rows, cols, 3), dtype='uint8')
        cv2.drawContours(image_extract_3ch, cont_all_big,
                         0, (127, 127, 127), -1)
        cnt = cont_all_big[0]
        # ------------ 判断“拳头” ------------ #
        ellipse_fit = cv2.fitEllipse(cnt)
        image_temp = np.zeros((rows, cols), dtype='uint8')
        cv2.ellipse(image_temp, ellipse_fit, 255, thickness=-1)
        ellipse_fit_conts, _ = cv2.findContours(
            image_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        diff = cv2.matchShapes(cnt, ellipse_fit_conts[0], 1, 0.0)
        if diff < diff_rate:
            cv2.ellipse(image_extract_3ch, ellipse_fit, (0, 0, 255), 2)
            b_ellipse_fit = True
        else:
            b_ellipse_fit = False

    return image_extract_3ch, b_ellipse_fit


def countFarPoint(far_ratio, image_np_extend):
    rows, cols = image_np_extend.shape[0:2]
    # ------------ 轮廓提取 ------------ #
    cont_all, _ = cv2.findContours(
        image_np_extend, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_all_big = myContList(cont_all, (rows*cols)/3)
    if len(cont_all_big) <= 0:
        image_extract_3ch = cv2.merge(
            [image_np_extend, image_np_extend, image_np_extend])
        i_num_far = -1
    else:
        image_extract_3ch = np.zeros((rows, cols, 3), dtype='uint8')
        cv2.drawContours(image_extract_3ch, cont_all_big,
                         0, (127, 127, 127), -1)
        cnt = cont_all_big[0]        # ------------ 凸缺陷检测 ------------ #
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        i_num_far = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            d /= 256
            if d > rows * far_ratio:
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                cv2.line(image_extract_3ch, start, end, [0, 255, 0], 2)
                cv2.circle(image_extract_3ch, far, 5, [255, 0, 0], -1)
                i_num_far += 1

    return image_extract_3ch, i_num_far


def tellHand(image_np, image_np_extend):
    image_ellipse_fit, b_ellipse_fit = myEllipseFitting(0.03, image_np)
    image_convex, i_num_far = countFarPoint(0.08, image_np_extend)

    if b_ellipse_fit:
        if i_num_far in (-1, 0, 1,):
            image_ret = image_ellipse_fit
            str_gesture = 'Fist'
        elif i_num_far in (2,):
            image_ret = image_convex
            str_gesture = 'Y'
        elif i_num_far in (3,):
            image_ret = image_convex
            str_gesture = '3'
        else:
            image_ret = image_ellipse_fit
            str_gesture = 'NULL'
    else:
        if i_num_far in (2,):
            image_ret = image_convex
            str_gesture = 'Y'
        elif i_num_far in (3,):
            image_ret = image_convex
            str_gesture = '3'
        elif i_num_far in (4,):
            image_ret = image_convex
            str_gesture = '5'
        else:
            image_ret = image_convex
            str_gesture = 'NULL'

    return image_ret, str_gesture


def processROI(b_have_hand, image_np, image_np_extend):
    if b_have_hand:
        try:
            image_extract, image_extract_extend = extractHand(
                image_np, image_np_extend)
            image_ret, str_gesture = tellHand(
                image_extract, image_extract_extend)
            # ------------ 打印手势 ------------ #
            cv2.putText(image_ret, str_gesture, (0, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 255, 0), 2)
            return image_ret, str_gesture
        except Exception as e:
            print('【Exception】', e)
            return image_np, "NULL"
    else:
        return image_np, "NULL"
