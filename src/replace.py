import cv2 as cv
import numpy as np
from utils import *

care_idx = 45, 36, 42, 39, 6, 10

def get_center(img):
    rect = cv.boundingRect(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    return (rect[0]+rect[2]//2, rect[1]+rect[3]//2)

def match68(img_mks, pat_mks, pat, shape):
    mkss = (pat_mks, img_mks)
    mkss = [[mks[idx] for idx in care_idx] for mks in mkss]
    mkss = tuple(map(np.float32, mkss))
    h, mask = cv.findHomography(*mkss, cv.RANSAC)
    return h, cv.warpPerspective(pat, h, shape[1::-1])

def scale(hulls, img):
    intensity, thres = 0.2, 5
    
    # draw the hull 
    mask = np.zeros(img.shape[0:2], img.dtype)
    cv.drawContours(mask, hulls, -1, white, -1)

    # blur the hull it according to size
    rect = cv.boundingRect(mask)
    odd = lambda x: int(x*intensity)*2+1
    mask = cv.GaussianBlur(mask,(odd(rect[2]), odd(rect[3]*3)),0)

    # threshold the hull to enlarge
    _, mask = cv.threshold(mask,thres,255,cv.THRESH_BINARY)
    hulls, _ = cv.findContours(mask, 1, 2)
    return hulls

def replace(img, img_mk, pat, pat_mk):
    h, pat = match68(img_mk, pat_mk, pat, img.shape)
    open_eyes = get_eyes(cv.perspectiveTransform(np.array([pat_mk], dtype='float32'), h).reshape(-1, 2).astype('int'))
    hulls = list(map(cv.convexHull, open_eyes))
    # scale hull while hull too thin
    hulls = scale(hulls, img)
    while sum([cv.arcLength(h, True) for h in hulls]) * 3 > sum([cv.contourArea(h) for h in hulls]):
        hulls = scale(hulls, img)
    # draw on mask
    mask = np.zeros(img.shape, img.dtype)
    cv.drawContours(mask, hulls, -1, white, -1)
    # clone with mask
    ret = cv.seamlessClone(pat, img, mask, get_center(mask), cv.NORMAL_CLONE)
    return ret, mask
    

