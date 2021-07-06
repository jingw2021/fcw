import cv2



from forward_pipeline_utils import Rect, detectron_prediction_to_rect_list

class Trackers():
    def __init__(self, tracker_size, iou_threshold=0.5):
        self.tracker_size = tracker_size
        self.tracker_width, self.tracker_height = tracker_size
        self.iou_threshold = iou_threshold
        self.last_object_id = 0
        self.objects = []
    def rect_to_p1_p2(self, bounding_box):
        p1 = \
            bounding_box[0]/self.tracker_width, \
            bounding_box[1]/self.tracker_height
        p2 = \
            (bounding_box[0]+bounding_box[2])/self.tracker_width, \
            (bounding_box[1]+bounding_box[3])/self.tracker_height
        return p1, p2

    def iou_x_y_w_h(self, rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        inter_left = max(x1, x2)
        inter_top = max(y1, y2)
        inter_right = min(x1 + w1, x2, + w2)
        inter_bottom = min(y1 + h1, y2, + h2)

        inter = (inter_bottom - inter_top)*(inter_right - inter_left)
        union = w1*h1 + w2*h2 - inter
        return inter/union
    
    def iou_l_t_r_b(self, rect1, rect2):
        l1, t1, r1, b1 = rect1
        l2, t2, r2, b2 = rect2

        inter_left = max(l1, l2)
        inter_top = max(t1, t2)
        inter_right = min(r1, r2)
        inter_bottom = min(b1, b2)

        inter = max(0, inter_bottom - inter_top)*max(0, inter_right - inter_left)
        union = (b1 - t1)*(r1 - l1) + (b2 - t2)*(r2 - l2) - inter
        return inter/union

    def update(self, frame, predictions):
        image_tracker = cv2.resize(frame, self.tracker_size)

        for object in self.objects:
            ok, bounding_box = object['tracker'].update(image_tracker)
            object['latest_bb'] = Rect(
                bounding_box[0]/self.tracker_width,
                bounding_box[1]/self.tracker_height,
                (bounding_box[0]+bounding_box[2])/self.tracker_width,
                (bounding_box[1]+bounding_box[3])/self.tracker_height
            )

        if predictions is None:
            return self.objects
        else:
            # predictions = detectron_prediction_to_rect_list(predictions)
            next_objects = []
            for prediction in predictions:
                bb = Rect(prediction[0], prediction[1], prediction[2], prediction[3])
                object_id = None
                for idx, object in enumerate(self.objects):
                    iou = self.iou_l_t_r_b(
                            object['latest_bb'].get_l_t_r_b(),
                            bb.get_l_t_r_b()
                        )
                    if iou > self.iou_threshold:
                        object = self.objects.pop(idx)
                        object_id = object['id']
                        break

                tracker = cv2.TrackerCSRT_create()

                tracker.init(image_tracker, bb.get_x_y_w_h((self.tracker_width, self.tracker_height), as_int=True))
                if object_id is None:
                    object_id = self.last_object_id
                    self.last_object_id +=1

                next_objects.append({'id': object_id, 'tracker': tracker, 'latest_bb': bb})
            

            self.objects = next_objects
            return self.objects