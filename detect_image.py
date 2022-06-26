import argparse
import tensorflow as tf
import cv2
from utils import detector_utils as detector_utils
from utils import recognizer_utils as recognizer_utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 消除警告


score_thresh = 0.1


if __name__ == '__main__':
    # 参数传递
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=1,
        help='Max number of hands to detect.')
    args = parser.parse_args()
    # 载入图像
    img_src = cv2.imread('test_image/image2.jpg')
    # 加载模型
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.compat.v1.Session(graph=detection_graph)
    if img_src is None:
        print('图片加载失败！')
    else:
        print('图片加载成功。')
        boxes, scores = detector_utils.detect_objects(
            img_src, detection_graph, sess)
        print('scores:', scores)
        # draw bounding boxes
        boxes_to_recog, scores_to_show = detector_utils.draw_box_on_image(
            args.num_hands, score_thresh, scores, boxes, 
            img_src.shape[1], img_src.shape[0], img_src)
        b_have_hand, img_roi, image_extend = recognizer_utils.drawBoxOfROI(
            scores_to_show, boxes_to_recog, 0.2, 0.8,
            img_src.shape[1], img_src.shape[0], img_src)
        cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
        cv2.namedWindow('ROI', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Result', img_src)
        cv2.imshow('ROI', img_roi)
        cv2.waitKey(0)

