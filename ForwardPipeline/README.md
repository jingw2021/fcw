# Lane Change detection
This project demoes a proof of concept for a lane change detection algorithm.

Lane change detection can be turned into a powerful safety feature. Coupled with turn signal information, such a system can be used to alert a driver who is changing lane without their signal turned on. A driver falling asleep could hence be notified preventing possible accidents. Additionaly, this information could be added to driver safety scores by checking that every lane change was previously notified by a turn signal.

This lane change detection is made of 3 different modules:

- Lane Detection: The first one role is to detect lane for a given frame
- Lane classification: The second one has to classify the lanes predicted by module 1 in different types. More specifically, it outputs the left and right ego lanes.
- Lane Change detection: The third module takes the left and right ego lanes as an input for each franme and predicts if vehicle is changing lane.

## Module 1: Lane Detection

Summary:
- Input: Frame
- Output: Lanes
- State: Stateless

This first module uses LaneATT model. LaneATT is currently the lane detection model with the best speed/accuracy trade-off. One can refer to the [paper](https://arxiv.org/abs/2004.10924) and [availabe code](https://github.com/lucastabelini/LaneATT) for additional details.

Summary of the paper: This paper proposes an aproach to lane detection using anchors. For this problem, anchors are not boxes are they usually are for object detection but lane anchors. THe networks uses some attention mechanism and predicts offset to the anchors. As always with anchor based network, an non maxima suppression layer is necessary to filter out redundant predictions. This NMS layer is customed as it has to operate on lanes.

Available code only implements a GPU version of the NMS layer. To run on the edge, this NMS layer would probably need to run on the CPU. I implemented a CPU version  of that NMS layer. See `override_nms` function.

The model used in this demo was trained on TUSimple dataset which is the only lane dataset available for commercial use.

Without retraining, this model will only work on a small subset of our data that is close to the data distriution of the TUSimple dataset. 
We will have to retrain this dataset to build a production feature.

TODO: Add more performance benchmark 

This first module is just a forward pass throught the model. It is stateless and outputs a set of lanes for each new image that it is fed.

TODO: Add visualization

## Module 2: Lane Classification

Summary:
- Input: Lanes output from module 1
- Output: Left ego lane, right ego lane
- State: Stateless

This second module is a simple heuristic.
It takes lanes output from previous module and output the left and right ego lane.

For each lane, the average x position of the lane is computed (the average lateral position). Right lanes are lanes with an average x position greater than .5 and left lanes with an average x position smaller than .5. The left ego lane is the left lane closest to center (`left_ego_idx = argmax(avg_x_pos(lane) for lane in left_lanes `). The left ego lane is the right lane closest to center (`right_ego_idx = argmin(avg_x_pos(lane) for lane in right_lanes `).
The intuition behind this heuristic is that dash camera views are generally centered and ego lanes are left annd right of the middle of the camera view.

This module runs on every frame lane detection output and is stateless.

TODO: Add visualization + code snippet

## Module 3: Lane Change detection

Summary
- Input: Left ego lane, right ego lane
- Output: Is changing lane or not
- State: Statefull

This third module takes output from output from stage 2 and predicts if the vehicle is crossing lane. This module is statefull. It keeps in its state the average x pos of the left lane and right lane. For each new output, it updates its state using some learning rate.
Then the center most lane is found (avg x pos closest to 0.5).

A single "centerness" value of the center lane is computed as the relative distance of the center lane to the middle of the average left and right lane.

A threshold is then used (0.5) to decide wether the vehicle is crossing a lane or not.

Intuitively, what this heuristic says if that the vehicle can move a quarter of the lane left or right. If it moves laterally more than that, it will trigger a lane change signal.

THe 2 hyper parameters that can be tweaked in that module are:
- learning rate (set to 0.1)
- threshold (set to 0.5)


TODO: Add visualization + code snippet