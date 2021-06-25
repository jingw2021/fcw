import numpy as np
from feature import match_descriptors
from scipy.spatial import distance


def ttc_cal(dataDQ, fps):
    """ Use the information from two consecutive frames to calculate the ttc
    
    params:
        dataDQ: deque for store info about two frames
        fps: used to caculate the \delta t between frames 
    returns:
        ttc_dict: a dict to store track id: ttc value
    """

    # match before previous frame and current frame
    prev_frame, cur_frame = dataDQ[0], dataDQ[1]
    matches = match_descriptors(prev_frame['descriptor'], cur_frame['descriptor'], "FLANN")
    # keep only the common tracks
    common_trks = np.intersect1d(prev_frame['tracks'][:,-1], cur_frame['tracks'][:,-1])
    prev_idx = [True if ele in common_trks else False for ele in prev_frame['tracks'][:,-1]]
    cur_idx = [True if ele in common_trks else False for ele in cur_frame['tracks'][:,-1]]
    prev_trks = prev_frame['tracks'][prev_idx, :]
    cur_trks = cur_frame['tracks'][cur_idx,:]
    # obtain the matching features within each track
    records = []
    for mt_idx, mt in enumerate(matches):
        prev_idx = mt[0].queryIdx
        cur_idx = mt[0].trainIdx
        prev_loc = prev_frame['keypoints'][prev_idx].pt
        cur_loc = cur_frame['keypoints'][cur_idx].pt

        for bb_prev, bb_cur in zip(prev_trks, cur_trks):
            if (prev_loc[0] >= bb_prev[0] and prev_loc[0] <= bb_prev[2] and prev_loc[1] >= bb_prev[1] and prev_loc[1] <= bb_prev[3] ) and \
                (cur_loc[0] >= bb_cur[0] and cur_loc[0] <= bb_cur[2] and cur_loc[1] >= bb_cur[1] and cur_loc[1] <= bb_cur[3]):
                records.append([bb_prev[-1], mt_idx])

    # calculate ttc for different tracks
    records = np.array(records)
    ttc_dict = {}
    for trk_index in common_trks:
        trk_matches = [ele for ele in records if ele[0] == trk_index]
        dist_ratio = []
        for outer_idx in range(len(trk_matches)-1):
            kp_outer_prev = prev_frame['keypoints'][matches[int(trk_matches[outer_idx][1])][0].queryIdx].pt
            kp_outer_cur = cur_frame['keypoints'][matches[int(trk_matches[outer_idx][1])][0].trainIdx].pt
            for inner_idx in range(outer_idx+1, len(trk_matches)):
                kp_inner_prev = prev_frame['keypoints'][matches[int(trk_matches[inner_idx][1])][0].queryIdx].pt
                kp_inner_cur = cur_frame['keypoints'][matches[int(trk_matches[inner_idx][1])][0].trainIdx].pt

                dist_cur = distance.euclidean(kp_outer_cur, kp_inner_cur)
                dist_prev = distance.euclidean(kp_outer_prev, kp_inner_prev)
                if dist_prev > 1 and dist_cur > 50:
                    dist_ratio.append(dist_cur*1.0/dist_prev)
        if np.median(dist_ratio) == 1:
            ttc = np.inf
        elif len(dist_ratio) > 0:
            ttc = (1.0/fps)/(np.median(dist_ratio)-1)
        else:
            ttc = -1
        ttc_dict[trk_index] = ttc
    return ttc_dict