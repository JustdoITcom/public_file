from lib.vidgear.gears import WriteGear
import cv2

import os
# print(os.stat('server_150.zip').st_uid)
print(getpwuid(os.stat('/bin/bash').st_uid).pw_name)

output_params = {
    "-input_framerate": 5,
    "-vcodec": "libx264",
    "-preset": "superfast",
    "-crf": 28,
    "-pix_fmt": "yuv420p",
    "-vsync": 1,
    "-framerate": 5,
}

writer = WriteGear(
    '/workspace/d2c/test.mp4',
    compression_mode=True,
    logging=False,
    **output_params)


frame = cv2.imread('nosignal.jpg')

writer.write(frame)
writer.close()
