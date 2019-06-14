import numpy as np
import cv2 as cv
import imutils
import dlib
import argparse

from utils import *
from blink import *
from match_face import *
from replace import replace

# a landmark predictor that should be downloaded from dlib
MARK_DETECT_PATH = 'shape_predictor_68_face_landmarks.dat'

# load detectors
face_detect = dlib.get_frontal_face_detector()
mark_detect = dlib.shape_predictor(MARK_DETECT_PATH)

# parse args
ap = argparse.ArgumentParser()
ap.add_argument('-b', '--base', default='../in/report_c.jpg', help='base image path')
ap.add_argument('-e', '--eye', default='../in/report_o.jpg', help='open eye image path')
ap.add_argument('-o', '--out', default='../out/report_out.jpg', help='output image path')
args = vars(ap.parse_args())


# mark all faces with 68 markers
def mark(img, marks):
    img = img.copy()
    #faces = face_detect(img, 1)
    #for rect in faces:
        #pts = rect2pts(rect)
        #cv.rectangle(img, *pts, red, 1) # draw rectangle on img
        #marks = mark_detect(img, rect)
    for m68 in marks:
        for m in m68:
            cv.circle(img, tuple(m), 1, green) # draw circle on img
    return img
            
def main():
    base = cv.imread(args['base'])
    material = cv.imread(args['eye'])
    bad = blink_faces(base, face_detect, mark_detect)
    cv.imwrite(args['base'].replace('in/','out/'), mark(base, bad))
    matches = matchfaces(base, bad, material, face_detect, mark_detect)
    for match in matches:
        base, mask = replace(base, match[0], material, match[1])
        cv.imwrite(args['base'].replace('in/','out/').replace('_c', '_mask'), mask)
    cv.imwrite(args['out'], base)
    #cv.imwrite(args['eye'].replace('in/','out/'), eye)

    

if __name__ == '__main__':
    main()
