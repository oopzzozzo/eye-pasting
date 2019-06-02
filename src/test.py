import cv2 as cv
import numpy as np
import argparse

# parse args
ap = argparse.ArgumentParser()
ap.add_argument('-b', '--base', default='../in/report_c.jpg', help='base image path')
ap.add_argument('-e', '--eye', default='../in/report_o.jpg', help='open eye image path')
ap.add_argument('-o', '--out', default='../out/out.jpg', help='output image path')
args = vars(ap.parse_args())

base = cv.imread(args['base'])
eye = cv.imread(args['eye'])

cv.createGaussianFilter()
