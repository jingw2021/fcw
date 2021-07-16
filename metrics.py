import numpy as np


def precision_recall_ones(predictions, actual, fps, time_win=[-3.4, -2.0]):
    """"Compute precision and recall for collision warning 
    Precision is tp/(tp+fp), recall is tp/(tp+fn)

    Params:
        predictions: list(((start, end),...., (start, end))), a list to host the warning intervals for 
                        each video
        actual: list(int), the actuall frame No. of the collision, if no collision, mark with -1
        fps: float, the frame per second 
        time_win: list[float], the time inteval [start, end] as the window for ground truth interval, 
                    Note: from end to actual, there is still warning, but it would be too late, thus 
                    those warning were not considered as FP. would be just ignored. 
    Returns:
        precision/recall
    """
    if len(predictions) != len(actual):
        print(
            f"DEBUG: predictions {len(predictions)} and actual {len(actual)} does not match")
        return
    tp, fp, fn = 0, 0, 0
    for idx, col_time in enumerate(actual):
        # there is no collision in the video
        if col_time == -1:
            fp += sum(1 for ele in predictions[idx] if len(ele) == 2)
            continue
        window_frames = [actual - fps*ele for ele in time_win]
        
        hit_flag = False
        for interval in predictions[idx]:
            if not hit_flag and interval[0] >= window_frames[0] and interval[0] <= window_frames[1]:
                tp += 1
                hit_flag = True
            # if prediction interval started is before window, FP
            elif interval[0] < window_frames[0]:
                fp += 1
        
        if not hit_flag:
            fn += 1
    pred_p, actual_p = tp+fp, tp+fn
    if pred_p == 0:
        print("DEBUG: no warning interval for all videos")
        pred_p = 0.1
    if actual_p == 0:
        print("DEBUG: no collision for all videos")
        actual_p = 0.1

    return tp/pred_p, tp/actual_p
        
    

    






if __name__ == "__main__":
    predictions = [[[87.0, 169.0]],
                   [[90.0, 154.0]],
                   [[108.0, 178.0]],
                   [[99.0, 112.0], [123.0, 154.0]],
                   [[120.0, 217.0]],
                   [[180.0, 241.0]]]

    for x in predictions[3]:
        print(x)
