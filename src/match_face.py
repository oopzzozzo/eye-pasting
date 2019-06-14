import cv2 as cv
import numpy as np
from align import *
from utils import *
from imutils import face_utils

rect_match_ratio = 0.25

pts_dist = lambda p1, p2: sum([(d[0] - d[1])**2 for d in zip(p1, p2)])

def pts_overlap_ratio(pts, divisor):
    op = ((max, max), (min, min))
    over = tuple(map(lambda sub: tuple([f(p, d) for (f, p, d) in zip(*sub)]), zip(op, pts, divisor)))
    return pts_area(over)/(pts_area(divisor)+pts_area(pts)) if leftup(over) else 0

def matchfaces(img, face_mks, material, face_detect, mark_detect):
    homo = get_homo(material, img)
    abstract = align(material, img, homo=homo)
    patches = [face_utils.shape_to_np(mark_detect(material, f)) for f in face_detect(material, 1)]
    if len(patches) == 0:
        return []
    dps = np.array([[p[dp_idx] for p in patches]], dtype='float32')
    dps = cv.perspectiveTransform(dps, homo).reshape(-1, 2)
    closest = lambda gnd: np.linalg.norm(dps - gnd[dp_idx], axis=1).argmin()
    matches = map(lambda f: (f, patches[closest(f)]), face_mks)
    #matches = map(lambda f: (f, dp2patch(min(patches, default=pix, key=lambda p:closeness(f, p)))), face_mks)
    return matches
    #matches = list(filter(lambda p: pts_dist(*map(mid_point, p)) < 0.5*pts_area(p[0])**0.5, matches))
    # match faces overlapping > threshold
    # matches = map(lambda f: (f, max(patches, default=pix, key=lambda p:pts_overlap_ratio(f, p))), faces)
    # matches = list(filter(lambda p: pts_overlap_ratio(p[0], p[1]) > 0.25, matches))
