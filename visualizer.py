import os

import cv2
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


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


class Scatter_Ploter():
    def __init__(self, fig_path):
        self.fig_path = fig_path
        self.pairs = []
    
    def _process_batch(self, records):
        """ obtain the predicted ttc and ground truth ttc 
        params: 
            records: list(dict), each dict with key
                "idx", int, video index
                "ttc_records", 2D array, the first column is the frame number, the second column is the predicted ttc 
                "actual", int, the NO. of actual collision frame
                "fps", float, the fps of the video
        returns:
            2d array, [ground truth ttc, predicted ttc, video index]
        """
        for rec in records: 
            idx = rec.get('idx')
            ttc_records = rec.get('ttc_records')
            fps = rec.get('fps', 30)
            actual = rec.get('actual')

            for ttc_rec in ttc_records:
                frame_no = ttc_rec[0]
                ttc = ttc_rec[1]
                gt_ttc = (actual - frame_no)/fps
                if gt_ttc > 0:
                    self.pairs.append([gt_ttc, ttc, idx])


    def plot(self, records):
        # get the pairs
        self._process_batch(records)
        # check whether to plot
        if len(self.pairs) < 1:
            print("There is no ttc prediction from the videos")
            return
        
        df = pd.DataFrame(self.pairs, columns=['actual', 'pred', 'video'])
        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        sn.scatterplot(x=df['actual'], y=df['pred'])
        fig.axes[0].set_xlabel("Actual TTC (seconds)")
        fig.axes[0].set_ylabel("Predicted TTC (seconds)")
        fig.savefig(self.fig_path, dpi=250)



