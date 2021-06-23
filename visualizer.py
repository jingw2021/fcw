import os

import cv2
import numpy as np

class Visualizer():
    def __init__(self, video_outpath, size, fps=10):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.out_width, self.out_height = size
        self.cap_out = cv2.VideoWriter(video_outpath,fourcc, fps, size)
    def __del__(self):
        self.cap_out.release()

    def add(self, frame, tracks=np.array([]), ttc = np.array([])):
       
        if tracks.shape[0] > 0:
            for tracked_obj in tracks:
                cv2.rectangle(frame, (int(tracked_obj[0]+ 2), int(tracked_obj[1]+2)), (int(tracked_obj[2] + 2), int(tracked_obj[3]+2)), (0,0,255), 2)
                cv2.putText(frame, f"Track {tracked_obj[4]}", (int(tracked_obj[0]), int(tracked_obj[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (0,0,255), 1)
        
        if ttc.shape[0] > 0:
            dy =50
            for t in ttc:
                cv2.putText(frame, f"Track {t[0]} TTC: {t[1]:.3}", (50, dy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
                dy += 50
        self.cap_out.write(frame)

    