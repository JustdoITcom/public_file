from lib.vidgear.gears import WriteGear


import os
print(os.stat('server_150.zip').st_uid)


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
    'test.mp4',
    compression_mode=True,
    logging=False,
    **output_params)


frame = cv2.imread('nosignal.jpg')

writer.write(frame)
writer.close()
