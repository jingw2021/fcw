import numpy as np

def precision_recall_ones(predictions, actual, fps, time_win=[-3.4, -2.0]):
    """"Compute precision and recall for collision warning 
    Precision is tp/(tp+fp), recall is tp/(tp+fn)

    Params:
        predictions: 2d array-like to indicate the frame No. and ttc
        actual: int, the actuall frame No. of the collision
        fps: float, the frame per second 
        time_win: list[float], the time inteval [start, end] as the window for ground truth interval, 
                    Note: from end to actual, there is still warning, but it would be too late, thus 
                    those warning were not considered as FP. would be just ignored. 
    """

    if actual is None: 
        pass



