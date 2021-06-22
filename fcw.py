import argparse
from ast import parse
from collections import deque

import cv2
import detectron2
from detectron2.engine.hooks import PeriodicCheckpointer
from detector import MRCNN_detector
from tracker import Tracker
from feature import detect_feature, desc_feature, match_descriptors
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

def main(video_path, detector, tracker):
    cap = cv2.VideoCapture(video_path)
    # get frame per second of the current video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    idx = 0
    dataDQ = deque(maxlen=2)
    while(cap.isOpened()):
        idx += 1
        print(f"====================Read on frame {idx}=========================")
        ret, frame = cap.read()        
        if frame is None:
            break
        # run the detecion model on the frame
        try: 
            outputs = detector.inference(frame)
            print(f"Number of detected vehicle {outputs.shape[0]}")
            # update the sort-based tracker 
            trks = tracker.update(outputs)      
            
            # feature extraction and description
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints = detect_feature(gray, "BRISK")
            kp, des = desc_feature(keypoints, gray, "BRIEF")

            data_point = {
                'idx': idx, 
                'frame': frame, 
                'outputs': outputs,
                "tracks" : trks,
                'keypoints': kp,
                'descriptor': des,
            }

            dataDQ.append(data_point)

            


            for d in trks:
                print(d)
            

        except Exception as e:
            print("current frame skipped for ttc", e)
            continue


        


    # v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(detector.cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_instance_predictions(predictions)
    # cv2.imwrite(f"/workspace/src/content/result/res{idx}.jpg", out.get_image()[:, :, ::-1])
    # image = cv2.imread('./content/images/image140.jpg')
    # output = detector.inference(image)

    # cv2.imwrite("./res.jpg", out.get_image()[:, :, ::-1])

def parse_args():
    """"Parse input arguments"""
    parser = argparse.ArgumentParser(description="FCW argument")
    parser.add_argument("--video_path", type=str, default='/workspace/src/content/processed-video-forward-1612980546861.mp4')
    parser.add_argument("--model_name", type=str, default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--max_age", type=int, default=1, help="Maximum number of frames to keep alive a track without associated detection")
    parser.add_argument("--min_hits", type=int, default=3, help="Minimum number of assoociated detection before track is initialized")
    parser.add_argument("--iou_th", type=float, default=0.5, help="Minimum IoU for match")
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = parse_args()
    detector = MRCNN_detector(args.model_name, args.device)
    trker = Tracker(args.max_age, args.min_hits, args.iou_th)

    main(args.video_path, detector, trker)
