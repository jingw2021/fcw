import cv2
import numpy as np
import os
import pickle
import torch
from torchvision import transforms
# importing module
import sys

from torchvision.transforms.transforms import ToPILImage
  
# appending a path
sys.path.append('/workspace/src/ForwardPipeline/LaneATT')

from lib.config import Config
from lib.runner import Runner
from lib.experiment import Experiment
from lib.lane import Lane

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class LaneDetectorInterface():
    def __init__(self):
        return

    def detect_lanes():
        raise NotImplementedError
    
    def save_prediction(self, prediction, save_path):
        pickle.dump(prediction, open(save_path, "wb"))
        return

    def overlay_image_prediction(self, img, prediction, save_path):        
        red = (0, 0, 255)

        height, width, _ = img.shape

        for lane in prediction:
            points = lane.points.copy()
            points[:, 0] *= width
            points[:, 1] *= height
            points = points.round().astype(int)

            for curr_p, next_p in zip(points[:-1], points[1:]):
                img = cv2.line(img, tuple(curr_p), tuple(next_p), color=red, thickness=3)

        cv2.imwrite(save_path, img)      


class LaneDetectorRunNetwork(LaneDetectorInterface):
    def __init__(self):
        super().__init__()
        exp_name = "laneatt_r18_culane"
        args = {
            "exp_name": exp_name,
            "mode": "eval",
            "resume": False,
            "deterministic": False,
        }
        args = dotdict(args)
        exp = Experiment(args.exp_name, args, mode=args.mode, exps_basedir='laneATT/experiments')
        cfg_path = "config.yaml".format(exp_name)
        cfg = Config(cfg_path)
        exp.set_cfg(cfg, override=False)
        device = torch.device('cpu') if not torch.cuda.is_available() or args.cpu else torch.device('cuda')
        self.runner = Runner(cfg, exp, device, view=args.view, resume=args.resume, deterministic=args.deterministic)

        self.trans = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((360, 640)),
                    transforms.ToTensor(),
                ]
            )
        self.device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
        self.model = cfg.get_model()
        self.test_parameters = cfg.get_test_parameters()
        print(self.test_parameters)

        epoch = exp.get_last_checkpoint_epoch()
        model_path = exp.get_checkpoint_path(epoch)
        print(model_path)
        self.model.load_state_dict(exp.get_epoch_model(epoch))
        self.model = self.model.to(self.device)
        self.model.eval()

    def detect_lanes(self, img, frame_idx):
        with torch.no_grad():
            input = torch.unsqueeze(self.trans(img), 0)
            input = input.to(self.device)
            output =  self.model(input, **self.test_parameters)
            prediction = self.model.decode(output, as_lanes=True)
            return prediction[0]  

class LaneDetectorFromSave(LaneDetectorInterface):
    def __init__(self, save_path):
        super().__init__()
        self.base_path = save_path

    def detect_lanes(self, frame, frame_idx):
        next_candidate = os.path.join(self.base_path,  "{:05d}.pkl".format(frame_idx))
        try:
            prediction = pickle.load(open(next_candidate, 'rb'))
            assert isinstance(prediction, list)
            for lane in prediction:
                assert isinstance(lane, Lane)
            return prediction
        except:
            print("Could not find file {}. This is the end of the sequence.".format(next_candidate))
            return None
                


if __name__ == "__main__":
    ldrn = LaneDetectorRunNetwork()

