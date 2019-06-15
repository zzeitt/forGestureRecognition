import datetime
from utils.detector_utils import WebcamVideoStream
import time
from multiprocessing import Queue, Pool
import multiprocessing
import tensorflow as tf
import cv2
from utils import detector_utils as detector_utils
from utils import recognizer_utils as recognizer_utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 消除警告

ACCURACY_GAP = 60

frame_processed = 0
score_thresh = 0.5
num_hands = 1
num_workers = 2
queue_size = 3

# Create a worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue
def worker(input_q, output_q, cap_params, frame_processed):
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    while True:
        frame = input_q.get()
        if (frame is not None):
            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)
            # draw bounding boxes
            boxes_to_recog, scores_to_show = detector_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                frame)
            b_have_hand, img_roi, img_extend = recognizer_utils.drawBoxOfROI(
                scores_to_show, boxes_to_recog, 0.2, 0.8,
                cap_params['im_width'], cap_params['im_height'], frame)
            img_roi, str_gesture = recognizer_utils.processROI(b_have_hand, img_roi, img_extend)
            # add frame annotated with bounding box to queue
            output_q.put(frame)
            output_q.put(img_roi)
            output_q.put(str_gesture)
            frame_processed += 1
        else:
            output_q.put(frame)
    sess.close()


if __name__ == '__main__':

    input_q = Queue(maxsize=queue_size)
    output_q = Queue(maxsize=queue_size)

    video_capture = WebcamVideoStream(
        src=0, width=600, height=400).start()

    cap_params = {}
    frame_processed = 0
    cap_params['im_width'], cap_params['im_height'] = video_capture.size()
    cap_params['score_thresh'] = score_thresh
    cap_params['num_hands_detect'] = num_hands

    print(cap_params)

    # spin up workers to paralleize detection.
    pool = Pool(num_workers, worker,
                (input_q, output_q, cap_params, frame_processed))

    start_time = datetime.datetime.now()
    num_frames = 0
    num_gesture_count = 0
    fps = 0
    index = 0

    cv2.namedWindow('Multi-Threaded Detection', cv2.WINDOW_NORMAL)
    cv2.namedWindow('ROI', cv2.WINDOW_AUTOSIZE)

    # ----------------Loop开始---------------- #
    while True:
        try:
            frame = video_capture.read()

            frame = cv2.flip(frame, 1)
            input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            output_frame = output_q.get()
            output_roi = output_q.get()
            str_gesture = output_q.get()  # 获取手势判别输出
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            output_roi = cv2.cvtColor(output_roi, cv2.COLOR_RGB2BGR)

            num_frames += 1
            num_frames_mod = num_frames % ACCURACY_GAP

            if (output_frame is not None):
                    cv2.imshow('Multi-Threaded Detection', output_frame)
                    cv2.imshow('ROI', output_roi)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    if str_gesture == '5':
                        num_gesture_count += 1
                    if num_frames_mod == ACCURACY_GAP - 1:
                        print("Accuracy:", num_gesture_count / ACCURACY_GAP)
                        num_gesture_count = 0                        

            else:
                break
        except Exception as e:
            print('【Exception】: ', e)

    # ----------------Loop结束---------------- #
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time
    print("fps", fps)
    video_capture.stop()
    cv2.destroyAllWindows()
    pool.terminate()
    pool.join()
