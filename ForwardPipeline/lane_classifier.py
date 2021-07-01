import cv2

LEFT_LANE_COLOR = (0, 0, 255)
RIGHT_LANE_COLOR = (0, 255, 0)
MIDDLE_LANE_COLOR = (0, 255, 255)
OTHER_LANE_COLOR =  (255, 0, 0)

class LaneClassifier:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.state = {
            'running_x_offset_avg_left_lane': None,
            'running_x_offset_avg_right_lane': None
        }
    
    def find_ego_lanes(self, prediction):
        avg_x_offset = [{'idx': idx, 'avg_x_pos': lane.points[:, 0].mean()} for idx, lane in enumerate(prediction)]


        left_lanes = [val for val in avg_x_offset if val['avg_x_pos'] < 0.5]  
        right_lanes = [val for val in avg_x_offset if val['avg_x_pos'] > 0.5]  

        middle_left_lane = max(left_lanes, key=lambda x: x['avg_x_pos'])
        middle_right_lane = min(right_lanes, key=lambda x: x['avg_x_pos'])

        middle_lane = middle_left_lane if abs(middle_left_lane['avg_x_pos'] - 0.5) < abs(middle_right_lane['avg_x_pos'] - 0.5) else middle_right_lane

        return middle_left_lane, middle_right_lane, middle_lane

    def update_running_avg(self, field, new_value):
        if self.state[field] is None:
            self.state[field] = new_value
        else: 
            self.state[field] = (1 - self.learning_rate)*self.state[field] + self.learning_rate*new_value

    def update(self, prediction):
        ego_left_lane, ego_right_lane, ego_middle_lane = self.find_ego_lanes(prediction)

        self.update_running_avg('running_x_offset_avg_left_lane', ego_left_lane['avg_x_pos'])
        self.update_running_avg('running_x_offset_avg_right_lane', ego_right_lane['avg_x_pos'])

        
        lanes_full = {
            'prediction': prediction,
            'left_lane_idx': ego_left_lane['idx'],
            'right_lane_idx': ego_right_lane['idx'],
            'middle_lane_idx': ego_middle_lane['idx'],
        }
        return lanes_full

    
    def overlay_visualization(self, img, prediction, ego_left_lane_idx, ego_right_lane_idx, ego_middle_lane_idx, save_path):
        self.overlay_lanes(img, prediction, ego_left_lane_idx, ego_right_lane_idx)
        self.overlay_running_avg(img)
        self.overlay_middle_lane_stats(img, prediction, ego_middle_lane_idx)
        cv2.imwrite(save_path, img)

        return img

    def overlay_lanes(self, img, prediction, ego_left_lane_idx, ego_right_lane_idx):

        height, width, _ = img.shape

        for idx, lane in enumerate(prediction):
            points = lane.points.copy()
            points[:, 0] *= width
            points[:, 1] *= height
            points = points.round().astype(int)

            if idx == ego_left_lane_idx:
                color = LEFT_LANE_COLOR
            elif idx == ego_right_lane_idx:
                color = RIGHT_LANE_COLOR
            else:
                color = OTHER_LANE_COLOR
            for curr_p, next_p in zip(points[:-1], points[1:]):
                img = cv2.line(img, tuple(curr_p), tuple(next_p), color=color, thickness=3)
    
    def overlay_middle_lane_stats(self, img, prediction, ego_middle_lane_idx):
        height, width, _ = img.shape

        middle_lane = prediction[ego_middle_lane_idx]
        mid_lane_avg_x = middle_lane.points[:,0].mean()
        in_between = (self.state['running_x_offset_avg_left_lane'] + self.state['running_x_offset_avg_right_lane'])/2
        half_distance_left_right = (self.state['running_x_offset_avg_right_lane'] - self.state['running_x_offset_avg_left_lane'])/2
        rel_dist_to_middle = abs(mid_lane_avg_x - in_between)/half_distance_left_right

        cv2.putText(img, 
            "distance of middle lane to side: {:.0%}".format(rel_dist_to_middle), 
            (10, 40), 
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            MIDDLE_LANE_COLOR,
            2
        )
        if rel_dist_to_middle<0.5:
            cv2.putText(img, 
                "Crossing lane", 
                (500, 500), 
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                MIDDLE_LANE_COLOR,
                6
            )
        
        print("{:.0%}".format(rel_dist_to_middle))

    def overlay_running_avg(self, img):
        height, width, _ = img.shape

        left_moving_avg = self.state['running_x_offset_avg_left_lane'] * width
        right_moving_avg = self.state['running_x_offset_avg_right_lane'] * width

        cv2.circle(img, (int(left_moving_avg), int(height/2)), 6, color=LEFT_LANE_COLOR, thickness= -1)
        cv2.circle(img, (int(right_moving_avg), int(height/2)), 6, color=RIGHT_LANE_COLOR, thickness= -1)

         
