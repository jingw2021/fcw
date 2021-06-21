import numpy as np
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

from detectron2.utils.logger import setup_logger
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
        vechile_types = {3,4,6,7,8}

        classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
        if classes is None:
            return

        classes = np.array([True if ele in vechile_types else False for ele in classes])
        boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores.numpy() if predictions.has("scores") else None
        boxes, scores = boxes[classes], scores[classes]
        
        res = np.zeros((boxes.shape[0], boxes.shape[1]+1))
        res[:,:-1] = boxes
        res[:,-1] = scores
        
        return res
