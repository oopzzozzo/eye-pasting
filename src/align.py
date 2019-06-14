'''
cloned from https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
'''

import cv2 as cv
import numpy as np


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def get_homo(src, dest):
    # Detect ORB features and compute descriptors.
    orb = cv.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(src, None)
    keypoints2, descriptors2 = orb.detectAndCompute(dest, None)

    # Match features.
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv.drawMatches(src, keypoints1, dest, keypoints2, matches, None)
    cv.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    try:
        return cv.findHomography(points1, points2, cv.RANSAC)[0]
    except:
        return None
        
def align(src, dest, **kwargs):
    homo = kwargs.get('homo', get_homo(src, dest))
    height, width, channels = dest.shape
    return cv.warpPerspective(src, homo, (width, height)) if homo is not None else src
