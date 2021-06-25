import os
import argparse
from ast import parse
from collections import deque, defaultdict
import pickle

import cv2
import numpy as np

from detector import MRCNN_detector
from tracker import Tracker
# from detectron2.data import MetadataCatalog
# from detectron2.utils.visualizer import Visualizer
from feature import detect_feature, desc_feature
from ttc import ttc_cal
from visualizer import Visualizer


def main(video_path, detector, tracker, save=True):
    cap = cv2.VideoCapture(video_path)
    # get frame meta data of the current video
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS {fps}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # `height`
    # output video path
    out_path = video_path.split('.')[0]+"_output.mp4"
    visualizer = Visualizer(out_path, (width, height), 10)

    idx = 0
    dataDQ = deque(maxlen=2)
    ttc_records = defaultdict(list)
    data_record = []
    data_out_path = video_path.split('.')[0]+"_records.pickle"
    while(cap.isOpened()):
        idx += 1
        print(
            f"====================Read on frame {idx}=========================")
        ret, frame = cap.read()
        if frame is None:
            break
        # run the detecion model on the frame
        # try:
        outputs = detector.inference(frame)
        print(f"Number of detected vehicle {outputs.shape[0]}")
        # update the sort-based tracker
        trks = tracker.update(outputs)

        data_point = {
            'idx': idx,
            'frame': frame,
            'outputs': outputs,
            "tracks": trks
        }
        if save:
            data_record.append(data_point)
            continue
        
        # feature extraction and description
        gray = cv2.cvtColor(data_point[frame], cv2.COLOR_BGR2GRAY)
        keypoints = detect_feature(gray, "BRISK")
        kp, des = desc_feature(keypoints, gray, "BRIEF")
        # result on current frame
        data_point.update(
            {
                'keypoints': kp,
                'descriptor': des,
            })
        

        dataDQ.append(data_point)

        if len(dataDQ) < 2:
            visualizer.add(frame)
            continue
        ttc_dict = ttc_cal(dataDQ, fps)
        for k, v in ttc_dict.items():
            ttc_records[k].append(v)

        # tracks remove due to loss tracks on current frame
        for k in (ttc_records.keys() - ttc_dict.keys()):
            ttc_records.pop(k, None)
        # filter non-stable ttc due to relative speed or inaccurate detection
        ttc_relay = []
        for k, v in ttc_records.items():
            if len(v) > 5:
                if max(v[-5:]) - min(v[-5:]) < 5 and min(v[-5:]) > 0:
                    ttc_relay.append([k, v[-1]])
        # overlay on the output
        visualizer.add(frame, data_point['tracks'], np.array(ttc_relay))

        # except Exception as e:
        #     print("current frame skipped for ttc", e)
        #     continue
    if save:
        with open(data_out_path, 'wb') as handle:
            pickle.dump(data_record, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(detector.cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_instance_predictions(predictions)
    # cv2.imwrite(f"/workspace/src/content/result/res{idx}.jpg", out.get_image()[:, :, ::-1])
    # image = cv2.imread('./content/images/image140.jpg')
    # output = detector.inference(image)

    # cv2.imwrite("./res.jpg", out.get_image()[:, :, ::-1])


def parse_args():
    """"Parse input arguments"""
    parser = argparse.ArgumentParser(description="FCW argument")
    parser.add_argument("--video_path", type=str,
                        default='/workspace/src/content/short_video_crash_rear_end_10sec.mp4')
    parser.add_argument("--model_name", type=str,
                        default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--max_age", type=int, default=1,
                        help="Maximum number of frames to keep alive a track without associated detection")
    parser.add_argument("--min_hits", type=int, default=3,
                        help="Minimum number of assoociated detection before track is initialized")
    parser.add_argument("--iou_th", type=float, default=0.5,
                        help="Minimum IoU for match")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    detector = MRCNN_detector(args.model_name, args.device)
    trker = Tracker(args.max_age, args.min_hits, args.iou_th)

    main(args.video_path, detector, trker)
