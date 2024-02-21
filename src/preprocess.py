import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import json
import util
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from tqdm import tqdm


def get_joint_trajectories(pose_dict: dict):
    """
    Get joint trajectories from pose_dict.

    :param pose_dict: dict
    pose_dict: {
      "pose_sequence": [{
          "frame_id": id,
          "people": [
                  {  
                      "person_id": id
                      "pose_keypoints_2d": [x1, y1, c1, x2, y2, c2, ...]
                  }
                  ]
         }]
    }
    :return: dict
    joint_trajectories: {
        "person_id": {
            "x": [x1, x2, ...],
            "y": [y1, y2, ...],
            "c": [c1, c2, ...]
        }
    }
    where x1, y1, c1 are the x, y, c of all joints in the first frame.
    """
    sequence = pose_dict['pose_sequence']
    joint_trajectories = {'num_frames': len(sequence)}
    trajectories = {}
    for frame in sequence:
        frame_id = frame['frame_id']
        people = frame['people']
        for person in people:
            person_id = person['person_id']
            pose_keypoints_2d = person['pose_keypoints_2d']
            x = pose_keypoints_2d[0::3]
            y = pose_keypoints_2d[1::3]
            c = pose_keypoints_2d[2::3]
            if person_id not in trajectories:
                trajectories[person_id] = {
                    "x": [],
                    "y": [],
                    "c": []
                }
            trajectories[person_id]['x'].append(x)
            trajectories[person_id]['y'].append(y)
            trajectories[person_id]['c'].append(c)

    for person_id in trajectories:
        trajectories[person_id]['x'] = np.array(trajectories[person_id]['x'])
        trajectories[person_id]['y'] = np.array(trajectories[person_id]['y'])
        trajectories[person_id]['c'] = np.array(trajectories[person_id]['c'])

    joint_trajectories['trajectories'] = trajectories

    return joint_trajectories


def smooth_joint_trajectories(joint_trajectories: dict):
    """
    Smooth joint trajectories using Savitzky-Golay filter.

    :param joint_trajectories: dict
    joint_trajectories: {
        "person_id": {
            "x": [x1, x2, ...],
            "y": [y1, y2, ...],
            "c": [c1, c2, ...]
        }
    }
    :return: dict
    smoothed_joint_trajectories: {
        "person_id": {
            "x": [x1, x2, ...],
            "y": [y1, y2, ...],
            "c": [c1, c2, ...]
        }
    }
    """
    num_frames = joint_trajectories['num_frames']
    smoothed_joint_trajectories = {'x': np.zeros((num_frames,18)), 'y': np.zeros((num_frames,18)), 'c': np.zeros((num_frames,18))}
    for person_id in joint_trajectories['trajectories']:
        # only save joint trajectoreis of person if they are present in all frames
        if joint_trajectories['trajectories'][person_id]['x'].shape[0] == joint_trajectories['num_frames']:
            for joint in range(18):
                allX = joint_trajectories['trajectories'][person_id]['x'][:, joint]
                allY = joint_trajectories['trajectories'][person_id]['y'][:, joint]
                allC = joint_trajectories['trajectories'][person_id]['c'][:, joint]

                mean_confidence = np.mean(allC, axis=0)

                smooth_idx = np.where(allC > mean_confidence - 0.1)[0]

                # if smooth_idx[0] > 10:
                #     smooth_idx = np.concatenate((np.array([0]), smooth_idx))

                int_x = CubicSpline(smooth_idx, allX[smooth_idx], bc_type='natural')(range(len(allX)))
                smoothed_x = savgol_filter(int_x, 5, 3)

                int_y = CubicSpline(smooth_idx, allY[smooth_idx], bc_type='natural')(range(len(allY)))
                smoothed_y = savgol_filter(int_y, 5, 3)

                int_c = CubicSpline(smooth_idx, allC[smooth_idx], bc_type='natural')(range(len(allC)))
                smoothed_c = savgol_filter(int_c, 5, 3)

                smoothed_joint_trajectories['x'][:, joint] = smoothed_x
                smoothed_joint_trajectories['y'][:, joint] = smoothed_y
                smoothed_joint_trajectories['c'][:, joint] = int_c

    return smoothed_joint_trajectories

def main():

    with open('pose_keypoints.json') as f:
        pose_dict = json.load(f)

    # get joint trajectories
    joint_trajectories = get_joint_trajectories(pose_dict)

    # smooth joint trajectories
    smoothed_joint_trajectories = smooth_joint_trajectories(joint_trajectories)

    # display joint trajectories
    test_video = 'ID01_fullterm_hypotermi_HINE21_MR djup asfyxi_13w F- (Nemo)_anon.mp4'
    cap = cv2.VideoCapture(test_video)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_ann_frames = joint_trajectories['num_frames']
    # out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
    out_smooth = cv2.VideoWriter('out_smooth.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    try:
        print("Drawing joint trajectories on video...")
        frame_id = 0
        # while cap.isOpened():
        for _ in tqdm(range(num_ann_frames)):
            if not cap.isOpened():
                print("Capture closed.")
                break
            ret, frame = cap.read()
            if frame is None or frame_id > joint_trajectories['num_frames']:
                break
            jTrajectory = joint_trajectories['trajectories'][0]
            # canvas = frame.copy()
            # frame_joints = util.draw_bodypose_from_jointTrajectory(canvas, jTrajectory, frame_id)
            canvas = frame.copy()
            frame_smoothjoints = util.draw_bodypose_from_jointTrajectory(canvas, smoothed_joint_trajectories, frame_id)

            frame_id += 1

            # out.write(frame_joints)
            out_smooth.write(frame_smoothjoints)

    except KeyboardInterrupt:
        print("Interrupted, cleaning up...")

    finally:
        print("Cleaning up...")
        cap.release()
        # out.release()
        out_smooth.release()
        print("Done.")
    

if __name__ == '__main__':
    main()

