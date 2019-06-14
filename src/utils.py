import cv2 as cv
import dlib
import numpy as np
from imutils import face_utils

# face_utils
(l_s_idx, l_e_idx) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(r_s_idx, r_e_idx) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
get_eyes = lambda marks: (marks[l_s_idx:l_e_idx], marks[r_s_idx:r_e_idx])
dp_idx = 33

# helper fuctions to cast cv2 among dlib
rect2pts = lambda rect: ((rect.left(), rect.top()), (rect.right(), rect.bottom()))
shape2pts = lambda shape: list(map(lambda i: (shape.part(i).x, shape.part(i).y), range(68)))
pts2rect = lambda pts: dlib.rectangle(*pts[0], *pts[1])

# pts functions
pix = ((0,0), (0,0))
safe = lambda rect, shape: tuple([tuple([min(max(x, 0), b-1) for x, b in zip(p, shape[1::-1])]) for p in rect])
pts_area = lambda ps: (ps[1][0] - ps[0][0]) * (ps[1][1] - ps[0][1])
leftup = lambda pt2: all([(x1 < x2) for x1, x2 in zip(*pt2)])
crop = lambda img, pts: img[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0]]
def preserve(img, pts):
    ret = np.zeros(img.shape, np.uint8)
    pts = safe(pts, img.shape)
    ret[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0]] = img[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0]]
    return ret

# colors
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
white = (255, 255, 255)
