import cv2
import numpy as np


def detect_feature(img, detector_type):
    """ detect the keypoints from image

    Params:
        img: input image
        detector_type: type of the detector (FAST)
    Returns:
        keypoints 
    """
    if detector_type == "FAST":
        detector = cv2.FastFeatureDetector_create()
    elif detector_type == "BRISK":
        detector = cv2.BRISK_create()
    elif detector_type == "ORB":
        detector = cv2.ORB_create()
    elif detector_type == "SIFT":
        detector = cv2.xfeatures2d.SIFT_create()
    else:
        raise(f"Unsupported detector type {detector_type}")

    keypoints = detector.detect(img, None)
    return keypoints


def desc_feature(keypoints, img, descriptor_type):
    """ describe the feature
    Params:
        keypoints: interest points detected from feature detector
        img: image
        descriptor_type: 
    """
    if descriptor_type == "BRIEF":
        descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    elif descriptor_type == "ORB":
        descriptor = cv2.ORB.create()
    else:
        raise(f"Unsupported descriptor type {descriptor_type}")
    kp, des = descriptor.compute(img, keypoints)

    return kp, des


def match_descriptors(des1, des2, matcher_type):
    """ match the feature descriptors from two images
    Params:
        des1: feature descriptor for image 1 
        des2: feature descriptor for image 2
        matcher_type: matcher type (brutal force and FLANN)
    Returns:
        matches: matached featured descriptors
    """
    if matcher_type == "BF":
        matcher = cv2.BFMatcher()
    elif matcher_type == "FLANN":
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        # type conversion for FLANN matching calcu
        des1 = np.float32(des1) 
        des2 = np.float32(des2)
    
    matches = matcher.knnMatch(des1, des2, k=1)
    return matches


if __name__ == "__main__":
    image1 = cv2.imread('/workspace/src/content/images/image140.jpg')
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    keypoints1 = detect_feature(gray1, "BRISK")
    # outImage = np.copy(image)
    # cv2.drawKeypoints(image, keypoints, outImage, color=(
    #     0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp1, des1 = desc_feature(keypoints1, image1, "BRIEF")

    image2 = cv2.imread('/workspace/src/content/images/image141.jpg')
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    keypoints2 = detect_feature(gray2, "BRISK")
    # outImage = np.copy(image)
    # cv2.drawKeypoints(image, keypoints, outImage, color=(
    #     0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp2, des2 = desc_feature(keypoints2, image2, "BRIEF")

    # cv2.imwrite("./out.png", outImage)
    # outImage2 = np.copy(image)
    # cv2.drawKeypoints(image, kp, outImage2, color=(0, 255, 0),
    #                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite("./out2.png", outImage)
    matches = match_descriptors(des1, des2, "FLANN")

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(image1,kp1,image2,kp2,matches,None,**draw_params)
    cv2.imwrite("match.jpg", img3)

