import cv2
import numpy as np


def crop_image(x1, x2, y1, y2, image):
    return image[x1:x2, y1:y2]


# Extract specific area from image to check if car is between the lines or not
def crop_between_line(image):
    x1, x2 = 241, 445
    y1, y2 = 0, 511

    return crop_image(x1, x2, y1, y2, image)


def create_trackbars():
    cv2.namedWindow('trackbar')
    cv2.createTrackbar('l_h', 'trackbar', 0, 255, lambda x: x)
    cv2.createTrackbar('l_s', 'trackbar', 0, 255, lambda x: x)
    cv2.createTrackbar('l_v', 'trackbar', 200, 255, lambda x: x)
    cv2.createTrackbar('u_h', 'trackbar', 255, 179, lambda x: x)
    cv2.createTrackbar('u_s', 'trackbar', 50, 255, lambda x: x)
    cv2.createTrackbar('u_v', 'trackbar', 255, 255, lambda x: x)


def threshold_using_trackbars(image):
    # Image Thresholding
    hsv_transformed = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # trackbar to find the best values for the mask
    l_h = cv2.getTrackbarPos('l_h', 'trackbar')
    l_s = cv2.getTrackbarPos('l_s', 'trackbar')
    l_v = cv2.getTrackbarPos('l_v', 'trackbar')
    u_h = cv2.getTrackbarPos('u_h', 'trackbar')
    u_s = cv2.getTrackbarPos('u_s', 'trackbar')
    u_v = cv2.getTrackbarPos('u_v', 'trackbar')

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv_transformed, lower, upper)

    return mask


def get_best_threshold_values_for_lines():
    l_h = 0
    l_s = 0
    l_v = 108
    u_h = 179
    u_s = 255
    u_v = 255

    return l_h, l_s, l_v, u_h, u_s, u_v


def threshold_image(image):
    l_h, l_s, l_v, u_h, u_s, u_v = get_best_threshold_values_for_lines()
    hsv_transformed = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv_transformed, lower, upper)

    return mask


def save_image(image, path):
    cv2.imwrite(path, image)


def resize_image(image):
    image = cv2.resize(image, (125, 50))
    return image


def preprocess_image(image):
    image = cv2.resize(image, (125, 50))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

