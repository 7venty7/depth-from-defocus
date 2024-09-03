import numpy as np
import cv2

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.1


def align_images(im1, im2):
    # convert images to greyscale
    im1gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # orb = cv2.ORB_create(MAX_FEATURES)

    # keypoints1, descriptors1 = orb.detectAndCompute(im1gray, None)
    # keypoints2, descriptors2 = orb.detectAndCompute(im2gray, None)

    # matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    # matches = list(matcher.match(descriptors1, descriptors2, None))
    # matches.sort(key=lambda x: x.distance, reverse=False)

    # n_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    # matches = matches[:n_good_matches]

    # im_matches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite("mathes.jpg", im_matches)

    # points1 = np.zeros((len(matches), 2), dtype=np.float32)
    # points2 = np.zeros((len(matches), 2), dtype=np.float32)

    # for i, match in enumerate(matches):
    #   points1[1, :] = keypoints1[match.queryIdx].pt
    #   points2[1, :] = keypoints2[match.queryIdx].pt

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(im1gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im2gray, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []

    for m, n in matches:
        if m.distance < n.distance * 0.7:
            good_matches.append(m)

    p1 = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    p2 = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

    im_matches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", im_matches)

    h, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    height, width, channels = im2.shape
    im1_reg = cv2.warpPerspective(im1, h, (width, height))

    return im1_reg, h


if __name__ == '__main__':
    im01_file = "./dataset/05/01.JPG"
    im02_file = "./dataset/05/02.JPG"

    im01 = cv2.imread(im01_file, cv2.IMREAD_COLOR)
    im02 = cv2.imread(im02_file, cv2.IMREAD_COLOR)

    im_reg, h = align_images(im02, im01)
    cv2.imwrite("aligned.jpg", im_reg)
