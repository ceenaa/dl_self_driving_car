"""
@ 2023, Copyright AVIS Engine
- An Example Compatible with AVISEngine version 2.1.1 / 1.2.4 (ACL Branch) or higher
"""
import os

import numpy as np
import torch

from helper import between_line
from avis import avisengine, config
import time
import cv2

from supervised.model import CNN

# create trackbar for steering wheel angle
# cv2.namedWindow("Trackbars")
# cv2.createTrackbar("Steering Wheel Angle", "Trackbars", 3, 7, lambda x: x)
# cv2.setTrackbarMin("Steering Wheel Angle", "Trackbars", 0)
# cv2.setTrackbarMax("Steering Wheel Angle", "Trackbars", 6)

# Creating an instance of the Car class
car = avisengine.Car()

# Connecting to the server (Simulator)
car.connect(config.SIMULATOR_IP, config.SIMULATOR_PORT)

# Counter variable
counter = 0

# race_drive.create_trackbars()

debug_mode = False

model = CNN(7)
model.load_state_dict(torch.load("supervised/sp_model.pth"))

# Sleep for 3 seconds to make sure that client connected to the simulator
time.sleep(3)

try:
    while True:
        counter = counter + 1
        car.getData()
        car.setSpeed(70)

        image = car.getImage()
        between_line_image = between_line.crop_between_line(image)
        threshold_image = between_line.threshold_image(between_line_image)
        resized_image = between_line.resize_image(threshold_image)

        # save resized image with PIL

        # angle = cv2.getTrackbarPos("Steering Wheel Angle", "Trackbars")

        angle = model(torch.tensor(resized_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)).argmax().item()

        angles = [-15, -30, -60, 0, 15, 30, 60]

        angle = angles[angle]

        # cv2.imwrite(f"supervised/images/{angle}/{counter}.jpg", resized_image)

        if abs(angle) > 40:
            car.setSpeed(car.getSpeed() / 2)
        elif abs(angle) > 50:
            car.setSpeed(car.getSpeed() / 4)
        else:
            car.setSpeed(70)

        car.setSteering(angle)

        if cv2.waitKey(10) == ord('q'):
            break

finally:
    car.stop()
