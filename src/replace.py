import cv2 as cv
import numpy as np
from utils import *

care_idx = 45, 36, 42, 39, 6, 10
#li_corner_idx, ri_corner_idx, chin_idx = care_idx

def get_center(img):
    rect = cv.boundingRect(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    return (rect[0]+rect[2]//2, rect[1]+rect[3]//2)

def match68(img_mks, pat_mks, pat, shape):
    mkss = (pat_mks, img_mks)
    mkss = [[mks[idx] for idx in care_idx] for mks in mkss]
    mkss = tuple(map(np.float32, mkss))
    #print(*mkss)
    h, mask = cv.findHomography(*mkss, cv.RANSAC)
    return h, cv.warpPerspective(pat, h, shape[1::-1])

def scale(mask):
    intensity, thres = 0.3, 10
    odd = lambda x: int(x*intensity)*2+1
    rect = cv.boundingRect(cv.cvtColor(mask, cv.COLOR_BGR2GRAY))
    mask = cv.GaussianBlur(mask,(odd(rect[2]), odd(rect[3]*3)),0)
    _, mask = cv.threshold(mask,thres,255,cv.THRESH_BINARY)
    #mask = cv.GaussianBlur(mask,(3, 3),0)
    return mask

def replace(img, img_mk, pat, pat_mk):
    h, pat = match68(img_mk, pat_mk, pat, img.shape)
    mask = np.zeros(img.shape, img.dtype)
    big_eyes = get_eyes(cv.perspectiveTransform(np.array([pat_mk], dtype='float32'), h).reshape(-1, 2).astype('int'))
    hulls = list(map(cv.convexHull, big_eyes))
    #hulls = list(map(cv.convexHull, get_eyes(img_mk)))
    cv.drawContours(mask, hulls, -1, white, -1)
    mask = scale(mask)
    ret = cv.seamlessClone(pat, img, mask, get_center(mask), cv.NORMAL_CLONE)
    return ret, mask
    

