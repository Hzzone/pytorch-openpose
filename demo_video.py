import copy
import numpy as np
import cv2
from glob import glob
import os

# openpose setup
from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

def process_frame(frame):
    candidate, subset = body_estimation(frame)
    canvas = copy.deepcopy(frame)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    # detect hand
    hands_list = util.handDetect(candidate, subset, frame)

    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(frame[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        all_hand_peaks.append(peaks)

    return util.draw_handpose(canvas, all_hand_peaks)

# writing video with ffmpeg
# https://stackoverflow.com/questions/61036822/opencv-videowriter-produces-cant-find-starting-number-error
import ffmpeg

def to8(img):
    return (img/256).astype('uint8')

# open the first video file found in .videos/
video_file = next(iter(glob("videos/*")))
cap = cv2.VideoCapture(video_file)

# pull video file info
# don't know why this is how it's defined https://stackoverflow.com/questions/52068277/change-frame-rate-in-opencv-3-4-2
input_fps = cap.get(5) 

# define a writer object to write to a movidified file
assert len(video_file.split(".")) == 2, \
        "file/dir names must not contain extra ."
output_file = video_file.split(".")[0]+".processed.avi"
# fourcc = cv2.VideoWriter_fourcc(*'XVID')


class Writer():
    def __init__(self, output_file, input_fps, input_framesize, gray=False):
        if os.path.exists(output_file):
            os.remove(output_file)
        self.ff_proc = (
            ffmpeg
            .input('pipe:',
                   format='rawvideo',
                   pix_fmt='gray' if gray else 'rgb24',
                   s='%sx%s'%(input_framesize[1],input_framesize[0]))
            .output(output_file, pix_fmt='yuv420p')
            .run_async(pipe_stdin=True)
        )

    def __call__(self, frame):
        self.ff_proc.stdin.write(frame.tobytes())

    def close(self):
        self.ff_proc.stdin.close()
        self.ff_proc.wait()


# why isn't this a with statement??
writer = None
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break

    if writer is None:
        input_framesize = frame.shape[:2]
        writer = Writer(output_file, input_fps, input_framesize)
    posed_frame = process_frame(frame)

    # cv2.imshow('frame', gray)

    # write the frame
    # writer(frame)
    writer(posed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.close()
cv2.destroyAllWindows()
