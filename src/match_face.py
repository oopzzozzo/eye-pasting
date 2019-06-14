import cv2 as cv
import numpy as np
from blink import avg_ear
from align import *
from utils import *
from imutils import face_utils

replace_threshold = 0.75
pts_dist = lambda p1, p2: sum([(d[0] - d[1])**2 for d in zip(p1, p2)])

# not used
def marks_overlap_ratio(mk1, mk2):
    bb1, bb2 = (tp2pts(cv.boundingRect(mk)) for mk in [mk1, mk2])
    op = ((max, max), (min, min))
    over = tuple(map(lambda sub: tuple([f(p, d) for (f, p, d) in zip(*sub)]), zip(op, bb1, bb2)))
    return pts_area(over)/(pts_area(bb1)+pts_area(bb2)) if leftup(over) else 0

def matchfaces(img, face_mks, material, face_detect, mark_detect):
    homo = get_homo(material, img)
    patches = [face_utils.shape_to_np(mark_detect(material, f)) for f in face_detect(material, 1)]
    if len(face_mks) == 0 or len(patches) == 0:
        return []
    dps = np.array([[p[dp_idx] for p in patches]], dtype='float32')
    dps = cv.perspectiveTransform(dps, homo).reshape(-1, 2)

    matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = matcher.match(np.array(face_mks, dtype='float32')[:,dp_idx], dps)
    matches = [(face_mks[m.queryIdx], patches[m.trainIdx]) for m in matches]
    matches = list(filter(lambda m: avg_ear(m[0]) < avg_ear(m[1]) * replace_threshold, matches))

    return matches
