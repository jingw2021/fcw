"""
range_detector.py
"""
from enum import Enum

from numpy.core.fromnumeric import shape

class RStates(Enum):
    DETECTING = 1
    WARNING = 2

import numpy as np
import pandas as pd
import pickle



class RangeDetector: 
    """
    The range detector is used to detect distraction sequence from a ttc value of listed frames
    """

    def __init__(self, **kwargs):
        self.row_logs = []
        self.initialize()
        # Constants 
        self.TTC_THRESHOLD = kwargs.get("TTC_THRESHOLD", 2.7)
        self.MIN_WARNING_INTERVAL = kwargs.get("MIN_WARNING_INTERVAL", 0.3)
        self.MAX_WARNING_INTERVAL = kwargs.get("MAX_WARNING_INTERVAL", 4)
        self.RECOVERY_DURATION = kwargs.get("RECOVERY_DURATION", 0.2)
        self.FPS = kwargs.get("fps", 30)


    def initialize(self):
        """
        Re-initialize the range detector, which clear all history from any previous detections
        """
        self.reset_detector()
        self.detected_intervals = []
    
    @staticmethod
    def get_time_delta(frame_idx_start, frame_idx_end, fps=30):
        """ Time between two frames
        Params:
            frame_idx_start: int, start frame index
            frame_idx_end: int, end frame index
            fps: int: frame per second
        Returns: 
            time delta in seconds
        """
        return (frame_idx_end-frame_idx_start)/fps

    
    def reset_detector(self, new_state=RStates.DETECTING):
        """ Resets the state of the range detector
        Params:
            new_state: enum RStates, the state at t+1
            clear_history: bool, whether to clear warning history
        """
        self.state = new_state
        self.warning_start_frame = None
        self.last_warning_frame = None
    
    def detect_warning_rt(self, ttc_record):
        """ Detect the ttc warning intervals in real time
        Params: 
            ttc_record: list, [frame idx, ttc value]
        """
        frame_idx, ttc = ttc_record[0], ttc_record[1]
        if self.state == RStates.DETECTING:
                if ttc != -1 and ttc < self.TTC_THRESHOLD:
                    self.state = RStates.WARNING
                    self.warning_start_frame = frame_idx
                    self.last_warning_frame = frame_idx
        elif self.state == RStates.WARNING:
            time_delta_start = RangeDetector.get_time_delta(
                    self.warning_start_frame,
                    frame_idx, 
                    self.FPS)
            if ttc > 0 and ttc < self.TTC_THRESHOLD:
                self.last_warning_frame = frame_idx  

                if time_delta_start > self.MAX_WARNING_INTERVAL:
                    print(f"DEBUG: {time_delta_start} since warning started")
                    self.detected_intervals.append(
                        [
                            self.warning_start_frame, 
                            frame_idx
                        ]
                    )

                    self.reset_detector()
            else:
                time_delta_last = RangeDetector.get_time_delta(
                    self.last_warning_frame, 
                    frame_idx, 
                    self.FPS
                )
                if time_delta_last > self.RECOVERY_DURATION:
                    if time_delta_start > self.MIN_WARNING_INTERVAL:
                        self.detected_intervals.append(
                            [
                                self.warning_start_frame, 
                                frame_idx
                            ]
                        )
                    self.reset_detector()

    def detect_warning_interval(self, ttc_records):
        """
        Detect ttc warning intervals
        Params: 
            ttc_records: 2d-like array, the first column stores the frame idx, the second column ttc
                            
        """
        if ttc_records.shape[0] < 1:
            print("range_detector: there is no ttc records")
            return

        for idx, row in enumerate(ttc_records):
            self.detect_warning_rt(row)
        
        return self.detected_intervals
            


if __name__ == "__main__":
    df = pd.read_csv("/workspace/src/content/video/result/ttc_scatter/july8_2/videos.csv")

    with open("/workspace/src/content/video/result/ttc_scatter/july8_2/recods.pickle", 'rb') as f:
        records = pickle.load(f)
    
    for idx, row in df.iterrows():
        range_detector = RangeDetector()
        length = row['length']
        ttc_records = np.ones(shape=(length+1,2))
        ttc_records[:, 0] = np.arange(length+1)
        ttc_records[:,1]=-1
        for row in records[idx]['ttc_records']:
            ttc_records[row[0], 1] = row[1]
        print(range_detector.detect_warning_interval(ttc_records))
                    


                    







        