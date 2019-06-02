'''
code from https://docs.opencv.org/3.3.0/d4/dc6/tutorial_py_template_matching.html
'''
import cv2 as cv

def search(pat, img):
    res = cv.matchTemplate(pat, img, cv.TM_CCOEFF_NORMED)
    _, _, _, tl = cv.minMaxLoc(res)

    return tl
