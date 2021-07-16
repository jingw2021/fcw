import os
import sys
import argparse
from collections import deque, defaultdict
import pickle

from numpy.core.records import record
sys.path.append('/workspace/src/ForwardPipeline')

import cv2
import numpy as np
import pandas as pd

from detector import MRCNN_detector
# from detectron2.data import MetadataCatalog
# from detectron2.utils.visualizer import Visualizer
from feature import detect_feature, desc_feature
from ttc import ttc_cal
from visualizer import Visualizer, Scatter_Ploter
from ForwardPipeline import detect_lanes, lane_classifier, vision_tracker


class Pipeline:
    def __init__(self, video_path, models, pipeline_every, detector_every, save_model_output, use_lane, track_ids, force_regenerate_lane_detection=False):
        """ Pipeline for forward collision warning
        params:
            video_path: str, path point to the video
            models: dict,  meta config for the models
            pipeline_every: int, frequence to run the pipeline
            detector_every: int, frequence to run the detector
            save_model_output: bool, run the pipeline on video, save the deep learning model results
            use_lane: bool, whether to use lane detector
            track_ids: list, leading tracks
        """
        self.video_path = video_path
        self.pipeline_every = pipeline_every
        self.detector_every = detector_every
        self.save_model_output = save_model_output
        self.output_path = video_path.split('.')[0]+"_output_vision_trker.mp4"
        self.use_lane = use_lane
        self.track_ids = track_ids
        self.lane_pred_save_path = os.path.join(
            video_path.split('.')[0], 'lane_prediction')
        self.running_frame_idx = 0
        self.detector = MRCNN_detector(
            models.get('detector'), models.get('device'))
        # self.tracker = Tracker(models.get('tracker_max_age'), models.get(
        #     "tracker_min_hits"), models.get("tracker_iou"))

        # vision tracker
        self.trackers = vision_tracker.Trackers(tracker_size=(540, 320))
        self.vision_tracking = True

        # for detector result
        self.inter_data_output = video_path.split('.')[0]+"_records.pickle"
        self.data_records = []

        # for lane result
        if self.use_lane:
            if os.path.exists(self.lane_pred_save_path) and not force_regenerate_lane_detection:
                self.lane_detector = detect_lanes.LaneDetectorFromSave(
                    self.lane_pred_save_path)
                self.save_lane_prediction = False
            else:
                print("Regenerating lane detection")
                os.makedirs(self.lane_pred_save_path, exist_ok=True)
                self.lane_detector = detect_lanes.LaneDetectorRunNetwork()
                self.save_lane_prediction = True
            self.lane_classifier = lane_classifier.LaneClassifier(
                learning_rate=models.get('lane_lr'))

    def run(self):
        cap_in = cv2.VideoCapture(self.video_path)
        fps = cap_in.get(cv2.CAP_PROP_FPS)
        width = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        visualizer = Visualizer(self.output_path, (width, height), fps)

        # if read mode, the detector is already saved
        if not self.save_model_output:
            with open(self.inter_data_output, 'rb') as f:
                self.data_records = pickle.load(f)

        data_dq = deque(maxlen=2)  # cache pre and cur detector
        ttc_records = defaultdict(list)  # hold ttc for all tracks
        data_point = {}
        warning = False
        res = []
        while cap_in.isOpened():
            ret, frame = cap_in.read()

            if not ret:
                print(f"Cannot read from video {self.video_path}")
                break
            # if need save to pickle, else read from pickle
            print(f"====================={self.running_frame_idx}===========================")
            if self.save_model_output:
                output = self.detector.inference(frame)
                data_point = {
                    'idx': self.running_frame_idx,
                    'frame': frame,
                    'outputs': output
                }
                self.data_records.append(data_point)
                self.running_frame_idx += 1
                continue

            data_point.update(self.data_records[self.running_frame_idx])

            if self.vision_tracking:                
                data_point['outputs'][:,0] /= width
                data_point['outputs'][:,1] /= height
                data_point['outputs'][:,2] /= width
                data_point['outputs'][:,3] /= height

                

            if self.use_lane:
                lane_prediction = self.lane_detector.detect_lanes(
                    frame.copy(), self.running_frame_idx)
                if self.save_lane_prediction:
                    self.lane_detector.save_prediction(lane_prediction, os.path.join(
                        self.lane_pred_save_path, "{:05d}.pkl".format(self.running_frame_idx)))
                lanes_full = self.lane_classifier.update(lane_prediction)


            if self.running_frame_idx % self.pipeline_every == 0:
                # update the tracks
                if self.running_frame_idx % self.detector_every == 0 and len(data_point['outputs']) > 0:
                    object_trackers = self.trackers.update(data_point['frame'], data_point['outputs'])
                else:
                    object_trackers = self.trackers.update(data_point['frame'], None)
                
                # format the interfaces
                trks = []
                for t in object_trackers:
                    bb = t['latest_bb'].get_l_t_r_b((width, height))
                    id = t['id']
                    trks.append(list(bb)+[id])
                trks = np.array(trks)


                # extract feature and matching
                gray = cv2.cvtColor(data_point["frame"], cv2.COLOR_BGR2GRAY)
                keypoints = detect_feature(gray, "BRISK")
                kp, des = desc_feature(keypoints, gray, "BRIEF")
                # result on current frame
                data_point.update(
                    {
                        'tracks': trks,
                        'keypoints': kp,
                        'descriptor': des,
                    })

                data_dq.append(data_point.copy())


                if len(data_dq) > 1:
                    ttc_dict = ttc_cal(data_dq, fps)
                    for k, v in ttc_dict.items():
                        ttc_records[k].append(v)

                    # tracks remove due to loss tracks on current frame
                    for k in (ttc_records.keys() - ttc_dict.keys()):
                        ttc_records.pop(k, None)
                    # filter non-stable ttc due to relative speed or inaccurate detection
                    ttc_relay = []
                    for k, v in ttc_records.items():
                        if len(v) > 2:
                            v_temp = [ele for ele in v[-5:] if ele != np.inf]
                            if len(v_temp) > 1 and max(v_temp) - min(v_temp) < 5 and v_temp[-1]>0:
                                if len(self.track_ids) > 0:
                                    if k in self.track_ids:
                                        ttc_relay.append([k, v_temp[-1]])
                                        if v_temp[-1] < 2.7:
                                            warning = True
                                        # add the predicted ttc to res
                                        res.append([self.running_frame_idx, v_temp[-1]])
                                else:
                                    ttc_relay.append([k, v_temp[-1]])
                            if k in self.track_ids:
                                if len(v_temp) > 3 and np.mean(np.abs(v_temp[-3])) > 3:
                                    warning = False
                    data_point.update(
                        {
                            'ttc_relay': np.array(ttc_relay)
                        })
                    visualizer.add(data_point['frame'], data_point.get('tracks', np.array([])), data_point.get('ttc_relay', np.array([])), frame_no=self.running_frame_idx, warning=warning)
            else:
                visualizer.add(data_point['frame'], data_point.get('tracks', np.array([])), data_point.get('ttc_relay', np.array([])), frame_no=self.running_frame_idx, warning=warning)

            self.running_frame_idx += 1


        if self.save_model_output:
            with open(self.inter_data_output, 'wb') as f:
                pickle.dump(self.data_records, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Detection reuslt saved to {self.inter_data_output}")
        
        return res, fps


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description="FCW argument")
    parser.add_argument("--video_path", type=str,
                        default='/workspace/src/content/video/short_video_crash_rear_end_10sec.mp4')
    parser.add_argument("--det_model_name", type=str,
                        default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--pipeline_every", type=int, default=3,
                        help="frequency for pipeline running")
    parser.add_argument("--detector_every", type=int, default=15,
                        help="frequency for detector running")
    parser.add_argument("--max_age", type=int, default=2,
                        help="Maximum number of frames to keep alive a track without associated detection")
    parser.add_argument("--min_hits", type=int, default=3,
                        help="Minimum number of assoociated detection before track is initialized")
    parser.add_argument("--iou_th", type=float, default=0.25,
                        help="Minimum IoU for match")
    parser.add_argument("--save_model_output", type=bool, default=False, help="whether to save detection result only")
    parser.add_argument("--use_lane", type=bool, default=False, help="whether to use detection result only")

    parser.add_argument("--track_ids", type=list, default=[])
    parser.add_argument("--force_regenerate_lane_detection", type=bool, default=False, help="whether to rerun lane detection model")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    models = {
        "detector": args.det_model_name, 
        "device": args.device,
        "tracker_max_age": args.max_age,
        "tracker_min_hits": args.min_hits,
        "tracker_iou": args.iou_th,
    }
    sp = Scatter_Ploter('./scatter.png')

    # read video list
    df = pd.read_csv("/workspace/src/content/video/result/ttc_scatter/july8_1/videos.csv")

    records = []

    # for idx, row in df.iterrows():
    #     tracks = [int(ele) for ele in row['tracks'].split('|')]

    #     pipeline = Pipeline(row['videos'],
    #                         models, 
    #                         args.pipeline_every, 
    #                         args.detector_every, 
    #                         False,
    #                         args.use_lane, 
    #                         tracks, 
    #                         args.force_regenerate_lane_detection)
    #     res, fps = pipeline.run()
    #     records.append({
    #         "idx": idx,
    #         "ttc_records": res, 
    #         "fps" : fps, 
    #         "actual" : row['actual']
    #     })
    
    # with open("/workspace/src/content/video/result/ttc_scatter/july8_1/recods.pickle", 'wb') as f:
    #     pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open("/workspace/src/content/video/result/ttc_scatter/july8_1/recods.pickle", 'rb') as f:
        records = pickle.load(f)
    
    sp.plot(records)
