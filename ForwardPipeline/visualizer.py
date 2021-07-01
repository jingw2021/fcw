import cv2
import os
from detectron2.data import MetadataCatalog

LEFT_LANE_COLOR = (0, 0, 255)
RIGHT_LANE_COLOR = (0, 255, 0)
MIDDLE_LANE_COLOR = (0, 255, 255)
OTHER_LANE_COLOR =  (255, 0, 0)

CAR_COLOR = (0, 0, 255)
TRACKED_CAR_COLOR = (0, 204, 0)

class Visualizer():
    def __init__(self, video_outpath, size, fps=10):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.out_width, self.out_height = size
        self.cap_out = cv2.VideoWriter(video_outpath,fourcc, fps, size)
        self.class_names = MetadataCatalog.get('coco_2017_val').thing_classes
    def __del__(self):
        self.cap_out.release()

    def add(self, frame, prediction=None, tracked_objects=None, lanes=None):
        if prediction:
            filtered_preds = self.transform_object_pred_to_list(prediction)
            for prediction in filtered_preds:
                cv2.rectangle(frame, (prediction['bounding_box']['left'], prediction['bounding_box']['top']), (prediction['bounding_box']['right'], prediction['bounding_box']['bottom']), CAR_COLOR, 2)
                cv2.putText(frame, prediction['pred_class'], (prediction['bounding_box']['left'] - 3, prediction['bounding_box']['top'] - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, CAR_COLOR, 2)
        
        if tracked_objects:
            for tracked_obj in tracked_objects:
                p1, p2 = tracked_obj['latest_bb'].get_p1_p2(size=(self.out_width, self.out_height), as_int=True)
                cv2.rectangle(frame, (p1[0] + 2, p1[1]+2), (p2[0] + 2, p2[1]+2), TRACKED_CAR_COLOR, 2)
                cv2.putText(frame, "{}".format(tracked_obj['id']), (p1[0], p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.51, TRACKED_CAR_COLOR, 3)
        
        if lanes:
            self.overlay_lanes(frame, lanes['prediction'], lanes['left_lane_idx'], lanes['right_lane_idx'])

        self.cap_out.write(frame)

    def overlay_lanes(self, frame, lane_prediction, ego_left_lane_idx, ego_right_lane_idx):
        height, width, _ = frame.shape

        for idx, lane in enumerate(lane_prediction):
            points = lane.points.copy()
            points[:, 0] *= width
            points[:, 1] *= height
            points = points.round().astype(int)

            if idx == ego_left_lane_idx:
                color = LEFT_LANE_COLOR
            elif idx == ego_right_lane_idx:
                color = RIGHT_LANE_COLOR
            else:
                color = OTHER_LANE_COLOR
            for curr_p, next_p in zip(points[:-1], points[1:]):
                frame = cv2.line(frame, tuple(curr_p), tuple(next_p), color=color, thickness=3)

    def transform_object_pred_to_list(self, prediction, car_only=True):
        filtered_predictions = []
        instances = prediction['instances']
        for pred_box, pred_class, score in zip(instances.pred_boxes, instances.pred_classes, instances.scores):
            pred_box = pred_box.numpy()
            left, top, right, bottom = int(pred_box[0]), int(pred_box[1]), int(pred_box[2]), int(pred_box[3])
            pred_class = pred_class.numpy()
            score = score.cpu().numpy()
            class_name = self.class_names[pred_class]
            if class_name in ["car"]:
                filtered_predictions.append(
                    {
                        'pred_class': class_name,
                        'score': score,
                        'bounding_box': {'top': top, 'left': left, 'bottom': bottom, 'right': right},
                    }
                )
        return filtered_predictions

    def overlay_lanes(self, img, prediction, ego_left_lane_idx, ego_right_lane_idx):
        height, width, _ = img.shape

        for idx, lane in enumerate(prediction):
            points = lane.points.copy()
            points[:, 0] *= width
            points[:, 1] *= height
            points = points.round().astype(int)

            if idx == ego_left_lane_idx:
                color = LEFT_LANE_COLOR
            elif idx == ego_right_lane_idx:
                color = RIGHT_LANE_COLOR
            else:
                color = OTHER_LANE_COLOR
            for curr_p, next_p in zip(points[:-1], points[1:]):
                img = cv2.line(img, tuple(curr_p), tuple(next_p), color=color, thickness=3)

    


