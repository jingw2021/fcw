import cv2
import numpy as np
import torch
from torchvision.ops import nms

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

from detectron2.utils.logger import setup_logger

from feature import detect_feature
setup_logger()


class MRCNN_detector: 
    def __init__(self, model_name, device='cuda'):
        self.cfg = get_cfg()

        ## OOB ##
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file(model_name))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
        self.cfg.MODEL.DEVICE = device
        self.predictor = DefaultPredictor(self.cfg)
    
    def inference(self, image):
        predictions =  self.predictor(image).get('instances')
        vechile_types = {2,3,5,6,7}

        classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
        if len(classes) < 1:
            return np.empty((0, 5))

        classes = np.array([True if ele in vechile_types else False for ele in classes])
        boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores.numpy() if predictions.has("scores") else None
        boxes, scores = boxes[classes], scores[classes]

        kept_idx = nms(torch.tensor(boxes), torch.tensor(scores), 0.5)
        boxes, scores = boxes[kept_idx, :], scores[kept_idx]
        
        res = np.zeros((boxes.shape[0], boxes.shape[1]+1))
        res[:,:-1] = boxes
        res[:,-1] = scores
        return res


if __name__ == "__main__":
        detector = MRCNN_detector("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", 'cpu')
        image = cv2.imread('/workspace/src/content/images/image116.jpg')

        print(detector.inference(image))


