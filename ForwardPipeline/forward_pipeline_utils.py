from detectron2.data import MetadataCatalog

class Rect():
    def __init__(self, l, t, r, b):
        self.left = l
        self.top = t
        self.right = r
        self.bottom = b

    def get_l_t_r_b(self, size=None):
        if size:
            width, height = size
            return self.left*width, self.top*height, self.right*width, self.bottom*height
        else:
            return self.left, self.top, self.right, self.bottom

    def get_x_y_w_h(self, size=None, as_int=False):
        if size:
            width, height = size
            x, y, w, h = self.left*width, self.top*height, (self.right-self.left)*width, (self.bottom-self.top)*height
        else:
            x, y, w, h = self.left, self.top, self.right-self.left, self.bottom-self.top
        
        if as_int:
            return int(x), int(y), int(w), int(h)
        else:
            return x, y, w, h
    

    def get_p1_p2(self, size=None, as_int=False):
        if size:
            width, height = size
            p1 = self.left*width, self.top*height
            p2 = self.right*width, self.bottom*height
        else:
            p1 = self.left, self.top
            p2 = self.right, self.bottom
        if as_int:
            p1 = int(p1[0]), int(p1[1])
            p2 = int(p2[0]), int(p2[1])
        return p1, p2
    

class_names = MetadataCatalog.get('coco_2017_val').thing_classes
def detectron_prediction_to_rect_list(prediction, car_only=True):
    filtered_predictions = []
    instances = prediction['instances']
    height, width = instances.image_size
    for pred_box, pred_class, score in zip(instances.pred_boxes, instances.pred_classes, instances.scores):
        pred_box = pred_box.numpy()
        left, top, right, bottom = int(pred_box[0]), int(pred_box[1]), int(pred_box[2]), int(pred_box[3])
        pred_class = pred_class.numpy()
        score = score.cpu().numpy()
        class_name = class_names[pred_class]
        if car_only and class_name in ["car"]:
            filtered_predictions.append(
                {
                    'pred_class': class_name,
                    'score': score,
                    'bounding_box': Rect(left/width, top/height, right/width, bottom/height),
                }
            )
    return filtered_predictions