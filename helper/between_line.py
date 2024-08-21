import cv2
import numpy as np
import torch
from torch import nn


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


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(49600, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, 1)
        self.act4 = nn.Sigmoid()

    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        x = self.pool2(x)
        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 10
        x = self.fc4(x)
        x = self.act4(x)
        return x


def resize_image(image):
    image = cv2.resize(image, (125, 50))
    return image


def preprocess_image(image):
    image = cv2.resize(image, (125, 50))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def is_between_lines(model, image):
    image_t = torch.tensor(image, dtype=torch.float).unsqueeze(0)

    output = model(image_t).item()
    output = output >= 0.5
    return output


def calculate_distance_from_lines(image):
    # middle point is (262, 141)
    middle_point = (262, 120)
    left_point = (0, -1)
    right_point = (510, -1)

    for i in range(262, 511):
        if image[120][i] == 255:
            right_point = (i, 120)
            break
    for i in range(262, 0, -1):
        if image[120][i] == 255:
            left_point = (i, 120)
            break

    return middle_point[0] - left_point[0], right_point[0] - middle_point[0]


def is_done(model, image1):
    is_in_line = is_between_lines(model, image1)
    return not is_in_line


