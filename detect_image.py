import argparse
import tensorflow as tf
import cv2
from utils import detector_utils as detector_utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 消除警告


score_thresh = 0.5


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
    sess = tf.Session(graph=detection_graph)
    if img_src is None:
        print('图片加载失败！')
    else:
        print('图片加载成功。')
        boxes, scores = detector_utils.detect_objects(
            img_src, detection_graph, sess)
        # draw bounding boxes
        detector_utils.draw_box_on_image(
            args.num_hands, score_thresh, scores, boxes, 
            img_src.shape[1], img_src.shape[0], img_src)
        # 显示图片
        cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
        cv2.imshow('Result', img_src)
        cv2.waitKey(0)

