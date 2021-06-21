import argparse
from ast import parse

import cv2
import detectron2
from detector import MRCNN_detector
from tracker import Tracker
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

def main(video_path, detector, tracker):
    cap = cv2.VideoCapture(video_path)
    # get frame per second of the current video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # idx = 0
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     print(frame.shape)
    #     if frame is None:
    #         break
         
    #     cv2.imwrite(f"./content/images/image{idx}.jpg", frame)
    #     idx += 1
    #     outputs = detector.inference(frame)
    #     print(outputs)
    image = cv2.imread('./content/images/image140.jpg')
    output = detector.inference(image)
    # v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(detector.cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_instance_predictions(output["instances"])
    # cv2.imwrite("./res.jpg", out.get_image()[:, :, ::-1])

def parse_args():
    """"Parse input arguments"""
    parser = argparse.ArgumentParser(description="FCW argument")
    parser.add_argument("--video_path", type=str, default='./content/processed-video-forward-1612980546861.mp4')
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
