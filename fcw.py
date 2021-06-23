import argparse
from ast import parse
from collections import deque

import cv2
from matplotlib import rc_context
import numpy as np
from scipy.spatial import distance
import detectron2
from detectron2.engine.hooks import PeriodicCheckpointer
from numpy.ma.core import inner
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
            # result on current frame
            data_point = {
                'idx': idx, 
                'frame': frame, 
                'outputs': outputs,
                "tracks" : trks,
                'keypoints': kp,
                'descriptor': des,
            }
            dataDQ.append(data_point)
            
            # match before previous frame and current frame
            prev_frame, cur_frame = dataDQ[0], dataDQ[1]
            matches = match_descriptors(prev_frame['descriptor'], cur_frame['descriptor'], "FLANN")
            # keep only the common tracks
            common_trks = np.intersect1d(prev_frame['tracks'][:,-1], cur_frame['tracks'][:,-1])
            prev_idx = [True if ele in common_trks else False for ele in prev_frame['tracks'][:,-1]]
            cur_idx = [True if ele in common_trks else False for ele in cur_frame['tracks'][:,-1]]
            prev_trks = prev_frame['tracks'][prev_idx, :]
            cur_trks = cur_frame['tracks'][cur_idx,:]
            # obtain the matching features within each track
            records = []
            for mt_idx, mt in enumerate(matches):
                prev_idx = mt[0].queryIdx
                cur_idx = mt[0].trainIdx
                prev_loc = prev_frame['keypoints'][prev_idx].pt
                cur_loc = cur_frame['keypoints'][cur_idx].pt

                for bb_prev, bb_cur in zip(prev_trks, cur_trks):
                    if (prev_loc[0] >= bb_prev[0] and prev_loc[0] <= bb_prev[2] and prev_loc[1] >= bb_prev[1] and prev_loc[1] <= bb_prev[3] ) and \
                        (cur_loc[0] >= bb_cur[0] and cur_loc[0] <= bb_cur[2] and cur_loc[1] >= bb_cur[1] and cur_loc[1] <= bb_cur[3]):
                        records.append([bb_prev[-1], mt_idx])

            # calculate ttc for different tracks
            records = np.array(records)
            ttc_records = []
            for trk_index in common_trks:
                trk_matches = [ele for ele in records if ele[0] == trk_index]
                dist_ratio = []
                for outer_idx in range(len(trk_matches)-1):
                    kp_outer_prev = prev_frame['keypoints'][matches[int(trk_matches[outer_idx][1])][0].queryIdx].pt
                    kp_outer_cur = cur_frame['keypoints'][matches[int(trk_matches[outer_idx][1])][0].trainIdx].pt
                    for inner_idx in range(outer_idx+1, len(trk_matches)):
                        kp_inner_prev = prev_frame['keypoints'][matches[int(trk_matches[inner_idx][1])][0].queryIdx].pt
                        kp_inner_cur = cur_frame['keypoints'][matches[int(trk_matches[inner_idx][1])][0].trainIdx].pt

                        dist_cur = distance.euclidean(kp_outer_cur, kp_inner_cur)
                        dist_prev = distance.euclidean(kp_outer_prev, kp_inner_prev)
                        if dist_prev > 1 and dist_cur > 50:
                            dist_ratio.append(dist_cur*1.0/dist_prev)
                if len(dist_ratio) > 0:
                    ttc = (1.0/fps)/(np.median(dist_ratio)-1)
                else:
                    ttc = -1
                ttc_records.append([trk_index, ttc])
            
            print(ttc_records)
                        

                


            




            


            

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
