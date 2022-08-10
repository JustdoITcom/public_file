import cv2
import os
import sys
import time
import numpy as np
import threading
import json
import natsort
import requests
import copy
import ffmpeg

from collections import deque
from datetime import datetime

os.environ['TZ'] = 'Asia/Tokyo'
# os.environ['TZ'] = 'Asia/Singapore'

IMAGE_SAVE_TIME = 0.1
CAMERA_DEBUG_TIME = 60
WRITE_FPS = 5
JSON_CHECK_TIME = 60
PING_SEND_TIME = 10
MAINTAIN_TIME = 0.2

IS_PRINT_DEBUG = True
IS_WRITER = True

DETECT_CNT = 5

IS_DISPLAY = False

LABEL_COLOR = {
    'fire': (0, 0, 255),
    'smoke': (0, 255, 0)
}

USE_AREA_COLOR = False
AREA_COLOR = [(131, 233, 127), (37, 96, 217), (143, 27, 235), (156, 146, 59)]

USE_RTSP_TIME = True

COCO_CLASSES = ['fire', 'smoke']
USE_TRT = True

RESOLUTION = {
        'VGA': [640, 480],
        'DVD': [720, 480],
        'HD': [1280, 720],
        'FHD': [1920, 1080]
}

CAMERA_SETTING_FOLDER = '/fourind/fire_and_smoke/workspace/c2d'
IMAGE_FOLDER = '/fourind/fire_and_smoke/workspace/d2c'
VIDEO_FOLDER = '/fourind/fire_and_smoke/workspace/d2c'
JSON_FOLDER = '/fourind/fire_and_smoke/workspace/d2c'
