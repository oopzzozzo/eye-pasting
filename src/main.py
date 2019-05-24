import numpy as np
import cv2 as cv
import imutils
import dlib
import argparse

# a landmark predictor that should be downloaded from dlib
MARK_DETECT_PATH = 'shape_predictor_68_face_landmarks.dat'

# load detectors
face_detect = dlib.get_frontal_face_detector()
mark_detect = dlib.shape_predictor(MARK_DETECT_PATH)

# parse args
ap = argparse.ArgumentParser()
ap.add_argument('-b', '--base', default='../in/report_c.jpg', help='base image path')
ap.add_argument('-e', '--eye', default='../in/report_o.jpg', help='open eye image path')
ap.add_argument('-o', '--out', default='../out/out.jpg', help='output image path')
args = vars(ap.parse_args())

# helper fuctions to cast cv2 among dlib
rect2pts = lambda rect:((rect.left(), rect.top()), (rect.right(), rect.bottom()))
shape2pts = lambda shape: map(lambda i: (shape.part(i).x, shape.part(i).y), range(68))

# colors
green = (0, 255, 0)
red = (0, 0, 255)

# mark all faces with 68 markers
def mark(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # argument "1" below upsamples the gray image to improve accuracy
    # lower -> not precise, higher -> time consuming
    faces = face_detect(gray_img, 1)
    for rect in faces:
        pts = rect2pts(rect)
        cv.rectangle(img, *pts, red, 1) # draw rectangle on img
        marks = mark_detect(gray_img, rect)
        for mark in shape2pts(marks):
            cv.circle(img, mark, 1, green) # draw circle on img
    return img

def main():
    base = cv.imread(args['base'])
    cv.imwrite('../out/close_out.jpg', mark(base))
    eye = cv.imread(args['eye'])
    cv.imwrite('../out/open_out.jpg', mark(eye))

if __name__ == '__main__':
    main()
