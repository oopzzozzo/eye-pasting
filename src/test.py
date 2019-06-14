import cv2 as cv
import numpy as np
import argparse
from utils import *

# parse args
ap = argparse.ArgumentParser()
ap.add_argument('-b', '--base', default='../in/report_c.jpg', help='base image path')
ap.add_argument('-e', '--eye', default='../in/report_o.jpg', help='open eye image path')
ap.add_argument('-o', '--out', default='../out/out.jpg', help='output image path')
args = vars(ap.parse_args())

base = cv.imread(args['base'])
eye = cv.imread(args['eye'])

base = cv.imread(args['base'])
material = cv.imread(args['eye'])
mask = np.zeros(base.shape, base.dtype)
cv.rectangle(mask, (1300, 500), (1500, 600), white, -1)
cv.imshow('mask', mask)
M = cv.moments(cv.cvtColor(mask, cv.COLOR_BGR2GRAY))
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
#print(cX, cY)
cv.imshow('out', cv.seamlessClone(material, base, mask, (cX, cY), cv.NORMAL_CLONE))
cv.waitKey()
