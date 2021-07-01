import os

import cv2
import numpy as np


class Visualizer():
    def __init__(self, video_outpath, size, fps=10):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.out_width, self.out_height = size
        self.cap_out = cv2.VideoWriter(video_outpath, fourcc, fps, size)

    def __del__(self):
        self.cap_out.release()

    def add(self, frame, tracks=np.array([]), ttc=np.array([]), predictions=np.array([]), frame_no=0, warning=False):

        if tracks.shape[0] > 0:
            for tracked_obj in tracks:
                cv2.rectangle(frame, (int(tracked_obj[0] + 2), int(tracked_obj[1]+2)), (int(
                    tracked_obj[2] + 2), int(tracked_obj[3]+2)), (0, 0, 255), 2)
                cv2.putText(frame, f"Track {tracked_obj[4]}", (int(tracked_obj[0]), int(
                    tracked_obj[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (0, 0, 255), 1)

        if ttc.shape[0] > 0:
            dy = 50
            for t in ttc:
                cv2.putText(frame, f"Track {t[0]} TTC: {t[1]:.3}", (
                    50, dy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                dy += 50
        if predictions.shape[0] > 0:
            for pred in predictions:
                cv2.rectangle(frame, (int(
                    pred[0] + 2), int(pred[1]+2)), (int(pred[2] + 2), int(pred[3]+2)), (0, 0, 255), 2)
                cv2.putText(frame, f"Conf {pred[4]}", (int(pred[0]), int(
                    pred[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (0, 0, 255), 1)

        if warning:
            cv2.putText(frame, f"Warning!", (800, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(frame, f"Frame {frame_no}", (800, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        self.cap_out.write(frame)
