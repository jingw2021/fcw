import numpy as np
import cv2
import os

import detect_lanes
import lane_classifier
import object_detection
import tracker
from visualizer import Visualizer

class Pipeline:
    def __init__(self, video_name, pipeline_every, detector_every, learning_rate, force_regenerate_lane_detection, force_regenerate_object_detection, save_object_detection_viz_path, visualize_lane_classification):
        self.video_name = video_name
        self.visualize_lane_classification = visualize_lane_classification
        self.lane_pred_save_path = os.path.join('results', video_name, 'lane_prediction')
        self.pipeline_every = pipeline_every
        self.detector_every = detector_every

        self.running_frame_idx = 0

        if os.path.exists(self.lane_pred_save_path) and not force_regenerate_lane_detection:
            self.lane_detector = detect_lanes.LaneDetectorFromSave(self.lane_pred_save_path)
            self.save_lane_prediction = False
        else:
            print("Regenerating lane detection")
            os.makedirs(self.lane_pred_save_path, exist_ok=True)
            self.lane_detector = detect_lanes.LaneDetectorRunNetwork()
            self.save_lane_prediction = True
        

        self.lane_classifier = lane_classifier.LaneClassifier(learning_rate=learning_rate)

        if visualize_lane_classification:
            self.lane_classification_viz_save_path =  os.path.join('results', video_name, 'lane_classification_viz')
            os.makedirs(self.lane_classification_viz_save_path, exist_ok=True)


        # Object detection # 
        obj_save_path = os.path.join('results', video_name, 'object_detector_pred')
        if os.path.exists(obj_save_path) and not force_regenerate_object_detection:
            self.object_detector = object_detection.ObectDetectorFromSave(
                base_path=obj_save_path,
                save_viz_path=save_object_detection_viz_path
            )
        else:
            self.object_detector = object_detection.ObjectDetector(
                save_pred_path=obj_save_path,
                save_viz_path=save_object_detection_viz_path
            )

        # Tracker
        self.trackers = tracker.Trackers(tracker_size=(540, 320))

    def run(self):
        video_path = 'data/{}.mp4'.format(self.video_name)
        cap_in = cv2.VideoCapture(video_path)

        width  = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
        height = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

        visualizer = Visualizer(os.path.join('results', self.video_name, 'output.mp4'), (width, height))
        if self.visualize_lane_classification:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            cap_out = cv2.VideoWriter('output.mp4',fourcc, 10, (width,height))
        while cap_in.isOpened():
            self.running_frame_idx +=1
            ret, frame = cap_in.read()
                # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if not self.running_frame_idx % self.pipeline_every == 0:
                continue
            
            lane_prediction = self.lane_detector.detect_lanes(frame.copy(), self.running_frame_idx)
            
            if self.running_frame_idx % self.detector_every == 0:
                object_predictions = self.object_detector.infer(frame, self.running_frame_idx)
                tracked_objects = self.trackers.update(frame, object_predictions)
            else:
                object_predictions = None
                tracked_objects = self.trackers.update(frame, None)
            # Tracker

            #import pdb; pdb.set_trace()
            if lane_prediction is None:
                print("There was a problem") 
                break
            
            if self.save_lane_prediction:
                self.lane_detector.save_prediction(lane_prediction, os.path.join(self.lane_pred_save_path, "{:05d}.pkl".format(self.running_frame_idx)))
            
            print("Got frame {} lane_prediction".format(self.running_frame_idx))
            lanes_full = self.lane_classifier.update(lane_prediction)

            if self.visualize_lane_classification:
                save_path = os.path.join(self.lane_classification_viz_save_path, "{:05d}.jpg".format(self.running_frame_idx))
                viz = self.lane_classifier.overlay_visualization(frame.copy(), lanes_full['prediction'], lanes_full['left_lane_idx'], lanes_full['right_lane_idx'], lanes_full['middle_lane_idx'], save_path)
                cap_out.write(viz)

            visualizer.add(frame, prediction=object_predictions, tracked_objects=tracked_objects, lanes=lanes_full)

            #prediction = ldrn.detect_lanes(frame.copy())
            
        
if __name__ == "__main__":
    video_name = "processed-video-forward-1620834900000"
    video_name = "processed-video-forward-1621266360000"
    video_name = "processed-video-forward-1621256982469"
    video_name = "processed-video-forward-1615393200000"
    video_name = "processed-video-forward-1621170600000"
    #video_name = "tu_simple_1"
    pipeline_every = 3
    detector_every = 15
    save_object_detection_viz_path=os.path.join('results', video_name, 'object_detector_viz')
    pipeline = Pipeline(
        video_name, 
        pipeline_every,
        detector_every,
        learning_rate=0.05,
        force_regenerate_lane_detection=False, 
        force_regenerate_object_detection=False,
        visualize_lane_classification=True,
        save_object_detection_viz_path=None,
    )
    pipeline.run()

