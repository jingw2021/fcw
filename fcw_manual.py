import os
import argparse
from ast import parse
from collections import deque, defaultdict
import pickle
import warnings

import cv2
import numpy as np

from detector import MRCNN_detector
from tracker import Tracker
# from detectron2.data import MetadataCatalog
# from detectron2.utils.visualizer import Visualizer
from feature import detect_feature, desc_feature
from ttc import ttc_cal
from visualizer import Visualizer


def main(pickle_path, track_id):
    # read the pickel file 
    with open(pickle_path, 'rb') as f:
        data_records = pickle.load(f)

    # video path
    video_path = pickle_path.split('_records')[0]+".mp4"
    cap = cv2.VideoCapture(video_path)
    # get frame meta data of the current video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # `height`
    # output video path
    out_path = video_path.split('.')[0]+"_output.mp4"
    print(out_path)
    visualizer = Visualizer(out_path, (width, height), fps)

    dataDQ = deque(maxlen=2)
    ttc_records = defaultdict(list)
    for idx in range(len(data_records)):
        print(f"==================frame {idx}============================")
        data_point = data_records[idx]
        # if True:
        #     visualizer.add(data_point['frame'], tracks=data_point["tracks"], frame_no = idx)
        #     continue
        # feature extraction and description
        gray = cv2.cvtColor(data_point["frame"], cv2.COLOR_BGR2GRAY)
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
            visualizer.add(data_point['frame'])
            continue
        ttc_dict = ttc_cal(dataDQ, fps)
        for k, v in ttc_dict.items():
            ttc_records[k].append(v)

        # tracks remove due to loss tracks on current frame
        for k in (ttc_records.keys() - ttc_dict.keys()):
            ttc_records.pop(k, None)
        # filter non-stable ttc due to relative speed or inaccurate detection
        ttc_relay = []
        warning = False
        for k, v in ttc_records.items():
            if len(v) > 5:
                v_temp = [ele for ele in v[-5:] if ele != np.inf]
                if len(v_temp) > 0 and max(v_temp) - min(v_temp) < 5 and min(v_temp) >= 0:
                    if track_id != 0:
                        if k in track_id:
                            ttc_relay.append([k, v_temp[-1]])
                            if v_temp[-1] < 2.7:
                                warning = True
                    else:
                        ttc_relay.append([k, v_temp[-1]])
                if k in track_id:
                    if len(v_temp)>3 and np.mean(np.abs(v_temp[-3])) > 3:
                        warning = False
        # overlay on the output
        visualizer.add(data_point['frame'], tracks=data_point["tracks"], ttc= np.array(ttc_relay), frame_no = idx, warning=warning)

        # except Exception as e:
        #     print("current frame skipped for ttc", e)
        #     continue


    # v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(detector.cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_instance_predictions(predictions)
    # cv2.imwrite(f"/workspace/src/content/result/res{idx}.jpg", out.get_image()[:, :, ::-1])
    # image = cv2.imread('./content/images/image140.jpg')
    # output = detector.inference(image)

    # cv2.imwrite("./res.jpg", out.get_image()[:, :, ::-1])


def parse_args():
    """"Parse input arguments"""
    parser = argparse.ArgumentParser(description="FCW argument")
    parser.add_argument("--pickle_path", type=str,
                        default='/workspace/src/content/video/short_video_crash_rear_end_10sec_records.pickle')
    parser.add_argument("--track_id", type=list, default=[1])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.pickle_path, args.track_id)
