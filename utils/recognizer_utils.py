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


def drawBoxOfROI(scores_roi, boxes_roi, margin_ratio, image_np):
    try:
        (left, right, top, bottom) = boxes_roi[0]  # 取第一个框
        margin_x = margin_ratio * (right - left)
        margin_y = margin_ratio * (bottom - top)
        left = int(left - margin_x)
        top = int(top - margin_y)
        right = int(right + margin_x)
        bottom = int(bottom + margin_y)
        p1 = (left, top)
        p2 = (right, bottom)
        cv2.rectangle(image_np, p1, p2, (0, 0, 255), 2, 1)
        b_have_hand = True
        image_roi = image_np[top:bottom, left:right]
        # 显示得分
        cv2.putText(image_np, str(float('%.2f' % scores_roi[0])),
                    (int(left), int(top)-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 0, 0), 2)
        return b_have_hand, image_roi
    except Exception as e:
        b_have_hand = False
        image_roi = np.zeros((200, 200, 3), np.uint8)
        return b_have_hand, image_roi


def processROI(b_have_hand, image_np):
    if b_have_hand:
        # 肤色分割
        image_ycrcb = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
        (ch_y, ch_cr, ch_cb) = cv2.split(image_ycrcb)
        ch_cr = cv2.GaussianBlur(ch_cr, (5, 5), 1)
        # 二值化
        _, image_bin = cv2.threshold(
            ch_cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 开运算
        SIZE = 10
        kernel = np.ones((SIZE,SIZE),np.uint8)
        image_open = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN, kernel)

        return image_open
    else:
        return image_np
