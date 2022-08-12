from lib.init import *
from lib.func import *
from lib.tracker import ObjectTracker
from lib.yolox_model import Yolox
from lib.vidgear.gears import WriteGear

class YoloxDetector(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.buffer = deque()
        self.detect_model = None
        self.models = {}

    def run(self):
        while True:
            if system.is_system_stopped:
                break

            if len(self.buffer) > 0:
                analyzer, writer, frame, read_cnt = self.buffer.popleft()

                analyzer.results = self.models['fire'].detect(frame)

                if len(analyzer.buffer) < 10:
                    analyzer.buffer.append([writer, frame, read_cnt])
            time.sleep(0.0001)

    def init_model(self, model, filename, thresh):
        self.models[model] = Yolox(
            exp_file = 'model/{}'.format(filename),
            trt_file = 'model/{}'.format(filename.replace('.py', '.pth')),
            conf=thresh,
            size=800,
        )


class DetectAnalyzer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.buffer = deque()
        self.video_save = []

        self.tracker = ObjectTracker()
        self.fps = 0
        self.parent = None

        self.read_cnt = 1
        self.results = []

        self.is_stop = False
        self.is_write = False

        self.confidenc_dic = {}

        self.event_occur = False

        self.now = None
        self.detection_time = None

    def run(self):
        while True:
            if self.is_stop:
                self.is_stop = False
                break

            if len(self.buffer) > 0:
                writer, frame, read_cnt = self.buffer.popleft()
                results = self.tracker.update(self.results)

                new_results = []
                for result in results:
                    label = result['label']
                    per = result['per']

                    is_inside = False
                    if len(self.parent.area) > 0:
                        for i, area in enumerate(self.parent.area):
                            i += 1
                            if len(area) > 0:
                                if not result['is_in_area']:
                                    is_inside = is_inside_polygon(result['bbox_pos'], area)
                                    result['is_in_area'] = is_inside
                                    if is_inside:
                                        result['area_id'] = i
                    else:
                        result['is_in_area'] = True
                        result['area_id'] = '0'

                    if int(result['area_id']) > -1\
                        and result['is_in_area']:
                        is_apply = False
                        if time.time() - result['maintain_time'] >= MAINTAIN_TIME:
                            if self.parent is not None:
                                self.confidenc_dic = {'fire': self.parent.confidence1, 'smoke': self.parent.confidence2}

                            if per >= self.confidenc_dic[label]:
                                if IS_PRINT_DEBUG:
                                    print('Camera{} {} Confidence: {}'.format(self.parent.cam_number, label, round(per.item(),2)))
                                self.parent.area_id = result['area_id']
                                new_results.append(result)

                self.parent.results = new_results
                if len(new_results) > 0:
                    if len(self.parent.before_buffer) >= (WRITE_FPS*2):
                        if (not self.parent.is_writer
                                and not self.parent.is_save
                                and time.time()-self.parent.dt_interval >= self.parent.interval):
                            self.parent.is_writer = True
                            if self.parent.event_type == '':
                                self.parent.event_type = new_results[0]['label']

                dt_frame = frame.copy()
                if len(self.parent.area) > 0:
                    for area in self.parent.area:
                        pts = np.array(area, np.int32)
                        if len(pts) > 0:
                            cv2.polylines(dt_frame, [pts], True, (133, 233, 127), 2)

                for result in self.parent.results:
                    xmin, ymin, xmax, ymax = result['bbox']
                    cv2.rectangle(dt_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

                dt_frame = cv2.resize(dt_frame, (1280, 720))
                frame = cv2.resize(frame, (1280, 720))

                if not self.parent.is_writer:
                    self.parent.before_buffer.append([frame.copy(), dt_frame.copy()])
                    if len(self.parent.before_buffer) > round(WRITE_FPS*2):
                        tmp_frame, result_frame = self.parent.before_buffer.popleft()
                        del tmp_frame
                        del result_frame
                else:
                    if self.parent.ai_frame is None:
                        self.parent.ai_frame = dt_frame.copy()
                        self.parent.ori_frame = frame.copy()
                        self.dt_fire_alarm()
                        time.sleep(0.001)

                    if self.parent.movie == 'On':
                        self.parent.frame_buffer.append([frame.copy(), dt_frame.copy()])

                        if len(self.parent.frame_buffer) >= (WRITE_FPS*8):
                            merge_buffer = copy.deepcopy(self.parent.before_buffer)
                            merge_buffer.extend(self.parent.frame_buffer)

                            if len(writer.buffer) < 1:
                                writer.buffer.append(merge_buffer)
                            self.parent.is_save = True
                            self.parent.frame_buffer = deque()
                            del merge_buffer
                            self.parent.dt_interval = time.time()

                    del frame
                    del dt_frame
            time.sleep(0.0001)

    def dt_fire_alarm(self):
        # patlite send
        if self.parent.patlite_url != '':
            if IS_PRINT_DEBUG:
                print('PATLITE send {}'.format(self.parent.patlite_url))
            try:
                requests.get(self.parent.patlite_url, timeout=0.1)
            except:
                pass

        self.now = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.detection_time = datetime.now().astimezone().isoformat()

        camera_id = self.parent.camera_id
        cam_number = self.parent.cam_number
        area_id = self.parent.area_id
        event_type = self.parent.event_type

        if IS_PRINT_DEBUG:
            print('One Event Detected CAMERA={}, AREA={}, CLASS={}, TIME={}'.format(
                                                        cam_number,
                                                        area_id,
                                                        event_type,
                                                        self.now
                                                        ))

        media_filename = '{}_{}_{}_{}'.format(self.now, system.device_id, cam_number, area_id)
        json_filename = '{}_{}_{}_{}'.format(self.now, system.device_id, camera_id, area_id)

        self.parent.media_filename = media_filename
        self.parent.json_filename = json_filename

        ''' meta data json '''
        if event_type == 'fire':
            value = '炎'
        else:
            value = '煙'

        with open('{}.json'.format(os.path.join(JSON_FOLDER, json_filename)), 'w') as  json_file:
            meta_json ={
                "message_type": "d2c",
                "data": {
                    "payload":{
                    "service_id": self.parent.server_info['service_id'],
                    "tenant_id": self.parent.server_info['tenant_id'],
                    "organization_id": self.parent.server_info['organization_id'],
                    "data_type": "D2C_FNC_0003",
                    "event_time": self.detection_time,
                    "detection_time": self.detection_time,
                    "device_id": system.device_id,
                    "camera_id": camera_id,
                    "detection_type": value,
                    "detection_image_name": '{}.jpg'.format(media_filename),
                    "detection_video_name": '{}.mp4'.format(media_filename),
                    "error_message": ""
                    }
                }
            }
            json.dump(meta_json, json_file, indent=4, ensure_ascii=False)
        self.parent.event_type = ''
        file_upload_list = ['jpg', 'ai_jpg']
        for file in file_upload_list:
            file_upload_json = '{}_{}.json'.format(media_filename, file)

            if file == 'ai_jpg':
                file = '_ai.jpg'
            elif file == 'jpg':
                file = '.jpg'

            save_frame = None
            if file == '.jpg':
                save_frame = self.parent.ori_frame.copy()
            else:
                save_frame = self.parent.ai_frame.copy()
            cv2.imwrite('{}{}'.format(os.path.join(IMAGE_FOLDER, media_filename), file), save_frame)

            with open(os.path.join(JSON_FOLDER, file_upload_json), 'w') as json_file:
                upload_json ={
                    "message_type": "blob",
                    "service_id": self.parent.server_info['service_id'],
                    "tenant_id": self.parent.server_info['tenant_id'],
                    "organization_id": self.parent.server_info['organization_id'],
                    "local_file_name": '{}{}'.format(media_filename, file),
                    "remote_file_name": '{}{}'.format(media_filename, file),
                }
                json.dump(upload_json, json_file, indent=4)


class VideoWriter(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.buffer = deque()
        self.is_stop = False

    def run(self):
        while True:
            if self.is_stop:
                self.is_stop = False
                break

            if len(self.buffer) > 0:
                frame_buffer = self.buffer.popleft()

                dt = time.time()
                cam_number = self.parent.cam_number
                media_filename = self.parent.media_filename
                json_filename = self.parent.json_filename

                # time.sleep(IMAGE_SAVE_TIME)
                output_params = {
                    "-input_framerate": 5,
                    "-vcodec": "libx264",
                    "-preset": "ultrafast",
                    "-crf": 28,
                    "-pix_fmt": "yuv420p",
                    "-vsync": 1,
                    "-framerate": 5,
                }

                try:
                    ori_video_name = '{}{}'.format(os.path.join(VIDEO_SUB_FOLDER, media_filename), '.mp4')
                    ai_video_name = '{}{}'.format(os.path.join(VIDEO_SUB_FOLDER, media_filename), '_ai.mp4')

                    writer = WriteGear(
                        ori_video_name,
                        compression_mode=True,
                        logging=False,
                        **output_params)

                    ai_writer = WriteGear(
                        ai_video_name,
                        compression_mode=True,
                        logging=False,
                        **output_params)

                    for data in frame_buffer:
                        frame, ai_frame = data
                        writer.write(frame)
                        ai_writer.write(ai_frame)
                        self.parent.dt_interval = time.time()

                    writer.close()
                    ai_writer.close()
                    del writer
                    del ai_writer

                    if os.path.isfile(ori_video_name):
                        shutil.move(ori_video_name, '{}{}'.format(os.path.join(VIDEO_FOLDER, media_filename), '.mp4'))
                        if IS_PRINT_DEBUG:
                            print('CAMERA{} >> {} move'.format(self.parent.cam_number, ori_video_name))
                    
                    if os.path.isfile(ai_video_name):
                        shutil.move(ai_video_name, '{}{}'.format(os.path.join(VIDEO_FOLDER, media_filename), '_ai.mp4'))
                        if IS_PRINT_DEBUG:
                            print('CAMERA{} >> {} move'.format(self.parent.cam_number, ai_video_name))
                    
                    file_upload_list = ['mp4', 'ai_mp4']
                    for file in file_upload_list:
                        file_upload_json = '{}_{}.json'.format(media_filename, file)
                        if file == 'ai_mp4':
                            file = '_ai.mp4'
                        else:
                            file = '.mp4'

                        with open(os.path.join(JSON_FOLDER, file_upload_json), 'w') as json_file:
                                upload_json ={
                                    "message_type": "blob",
                                    "service_id": self.parent.server_info['service_id'],
                                    "tenant_id": self.parent.server_info['tenant_id'],
                                    "organization_id": self.parent.server_info['organization_id'],
                                    "local_file_name": '{}{}'.format(media_filename, file),
                                    "remote_file_name": '{}{}'.format(media_filename, file),
                                }
                                json.dump(upload_json, json_file, indent=4)

                    now = datetime.now().strftime('%Y%m%d_%H%M%S')
                    # if IS_PRINT_DEBUG:
                    #     print('camera : {}, Writing the all File Done : {}\n'.format(cam_number, now))

                    self.parent.is_writer = False
                    self.parent.is_save = False
                    del frame_buffer
                    self.parent.ori_frame = None
                    self.parent.ai_frame = None

                except Exception as ex:
                    if IS_PRINT_DEBUG:
                        print('ERROR >> CAMERA{} VideoWriter not'.format(self.parent.cam_number), )
                    self.parent.is_writer = False
                    del frame_buffer
                    self.parent.ori_frame = None
                    self.parent.ai_frame = None

            time.sleep(0.1)


class BomWorker(threading.Thread):
    def __init__(self, url, work_name, cam_number, server_info, meta, detector=None, analyzer=None, writer=None):
        threading.Thread.__init__(self)

        self.meta = meta
        self.work_name = work_name
        # server meta information
        self.server_info = server_info

        # save
        self.before_buffer = deque()
        self.frame_buffer = deque()
        self.video_buffer = deque()

        self.dt_interval = time.time()
        # camera meta information
        self.url = url
        self.alert = ''
        self.movie = ''
        self.area = []
        self.resolution = ''
        self.interval = 0
        self.confidence1 = 0
        self.confidence2 = 0
        self.schedule_flag = ''
        self.schedule = {}
        self.tenant_id = ''
        self.device_id = ''
        self.area_id = ''
        self.cam_number = cam_number
        self.event_type = ''

        self.is_writer = False
        self.is_save = False

        self.cap = None

        self.detector = detector
        self.analyzer = analyzer
        self.writer = writer

        self.disp_frame = None
        self.ori_frame = None
        self.ai_frame = None
        # self.frame = None

        self.is_detect = True
        self.read_idxes = None
        self.model_list = {}
        self.model_idxes = {}

        self.write_idxes = []

        # Frame, State
        self.fps = 0
        self.frame_cnt = 1
        self.read_cnt = 1
        self.frame_w = 0
        self.frame_h = 0

        self.week = ''
        self.hour = 0
        self.minute = 0

        self.results = []

        self.ping_time = time.time()
        self.recon_debug_time = time.time()
        self.reconn_time = time.time()
        self.calc_date = time.time()
        self.json_reference_time = time.time()
        self.debug_time = time.time()

        self.is_stop = False

        ### ffmpeg
        self.packet_size = -1
        self.is_play = False
        self.reconn_delay = 5
        self.RECONN_SECONDS = 10
        self.proc = None

        self.init_time()

        # self.now = None
        # self.detection_time = None
        self.media_filename = None
        self.json_filename = None

    def run(self):
        self.clear(self.meta)
        self.play()

        while True:
            if system.is_system_stopped:
                break

            if self.is_stop:
                self.analyzer.is_stop = True
                self.writer.is_stop = True
                self.is_stop = True
                break

            if time.time() - self.calc_date > 10:
                self.init_time()
                self.calc_date = time.time()

            if self.is_play:
                ret, frame = self.cap.read()

                if ret:
                    if self.fps > 0:
                        if self.read_cnt > round(self.fps)-1:
                            self.read_cnt = 1
                        self.read_cnt += 1

                    if self.schedule_flag == 'On':
                        if len(self.schedule[self.week]) > 0:
                            self.is_detect = self.is_in_schedule()

                    if self.is_detect\
                        and self.detector is not None\
                        and self.alert == 'On':
                        try:
                            if len(self.detector.buffer) < 1:
                                if len(self.model_idxes[self.read_cnt]) > 0:
                                    self.detector.buffer.append([self.analyzer, self.writer, frame, self.read_cnt])
                        except Exception as ex:
                            print('model_idxes error >> ', ex)

                            if self.detector is not None:
                                self.detector.fps = self.fps

                            self.model_list[self.work_name] = DETECT_CNT
                            self.on_state(is_play=self.is_play)

                    else:
                        if time.time() - self.debug_time > 5:
                            print('\nCurrently, detection is not in progress. Please check the schedule time or whether it is detected.\n')
                            self.debug_time = time.time()
                            self.results =[]

                    self.disp_frame = frame.copy()
                else:
                    self.clear(self.meta, is_play=True)
                    # self.play()
            else:
                self.results = []
                if self.url != '':
                    if (time.time() - self.reconn_time) > 5:
                        try:
                            self.clear(self.meta, is_play=True)
                            self.play()

                            if self.is_play:
                                self.reconn_delay = 1
                                if IS_PRINT_DEBUG:
                                    print('the camera reconnected CAMERA : {}, URL : {}'.format(self.cam_number, self.url))
                                self.reconn_time = time.time()
                            else:
                                self.reconn_delay *= 2
                                if self.reconn_delay >= self.RECONN_SECONDS:
                                    self.reconn_delay = 1

                                    if time.time() - self.recon_debug_time > 5:
                                        if IS_PRINT_DEBUG:
                                            print('the camera reconnecting CAMERA : {}, URL : {}'.format(self.cam_number, self.url))
                                        self.recon_debug_time = time.time()

                        except Exception as ex:
                            print('reconnect error >> ', ex)
                            pass
                else:
                    time.sleep(5)

            try:
                if USE_RTSP_TIME:
                    if 'rtsp://' in self.url:
                        time.sleep(0.0001)
                    else:
                        time.sleep(1/(self.fps))
                else:
                    time.sleep(1/(self.fps))
            except Exception as ex:
                # print('camera number {} error >> {}'.format(self.cam_number, ex))
                time.sleep(0.0001)

    def get_metadata(self, url):
        try:
            probe = ffmpeg.probe(url, loglevel='panic')
            return next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        except ffmpeg.Error as e:
            return {}

    def play(self):
        self.dt_reconn = time.time()

        try:
            if self.cap is not None:
                self.cap = None

            if self.url != '':
                self.cap = cv2.VideoCapture(self.url)
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)

            if self.fps > 999:
                meta = self.get_metadata(self.url)

                if IS_PRINT_DEBUG:
                    print('CAMERA{} INFO'.format(self.cam_number))
                    print('{}'.format(meta))
                try:
                    if 'r_frame_rate' in meta:
                        self.fps = int(meta['r_frame_rate'].split('/')[0])
                    else:
                        self.fps = 30
                except:
                    self.fps = 30
            print('fps: {}\n'.format(self.fps))

            if self.fps > 0:
                self.is_play = True
            else:
                self.is_play = False
                self.fps = 0
        except Exception as ex:
            if IS_PRINT_DEBUG:
                print('CAMERA : {}, PLAY ERROR >> {}'.format(self.cam_number, ex))
            self.is_play = False
            self.fps = 0
            time.sleep(1)

        if self.is_play:
            if IS_PRINT_DEBUG:
                print('the camera connect success CAMERA : {}, URL : {}'.format(self.cam_number, self.url))
        # else:
        #     if IS_PRINT_DEBUG:
        #         print('the camera connect fail CAMERA : {}, URL : {}'.format(self.cam_number, self.url))

        self.model_list[self.work_name] = DETECT_CNT
        self.on_state(is_play=self.is_play)

    def clear(self, meta, is_play=False, is_url_change=False):
        self.meta = meta

        self.frame_cnt = 1

        if is_url_change:
            if self.url != meta['rtsp_url']:
                # self.cap = cv2.VideoCapture(self.url)
                if meta['rtsp_url'] != '':
                    self.url = meta['rtsp_url']
                    self.cap = cv2.VideoCapture(self.url)

        
        if IS_PRINT_DEBUG:
            print('###################')
            print('CAMERA{} RECONN'.format(self.cam_number))

        self.is_writer = False
        self.is_save = False
        self.before_buffer = deque()
        self.frame_buffer = deque()
        self.video_buffer = deque()

        # if is_play:
        self.is_play = False

        self.alert = meta['alert']
        self.movie = meta['movie']

        self.area = []
        for i, key in enumerate(meta):
            if 'area' in key:
                area = []
                for pt in meta[key]:
                    if 'p' in pt:
                        area.append(meta[key][pt])
                self.area.append(area)
            elif 'schedule' in key:
                if '_' not in key:
                    for value in meta[key]:
                        self.schedule[value] = meta[key][value]
        self.resolution = meta['resolution']
        self.interval = meta['interval']

        try:
            self.confidence1 = meta['confidence1']
        except Exception as ex:
            self.confidence1 = 0.4
            print('confidence1 error >> ', ex)
            print('confidence1 >> 0.4')

        try:
            self.confidence2 = meta['confidence2']
        except Exception as ex:
            self.confidence2 = 0.4
            print('confidence2 error >> ', ex)
            print('confidence2 >> 0.4')

        self.schedule_flag = meta['schedule_flag']

        self.device_id = meta['device_id']
        self.camera_id = meta['camera_id']
        self.patlite_url = meta['patlite']

        self.model_list[self.work_name] = DETECT_CNT
        self.on_state(is_play=self.is_play)

    def init_analyzer(self):
        self.analyzer.parent = self
        self.analyzer.start()

    def init_writer(self):
        self.writer.parent = self
        self.writer.start()

    def init_idxes(self, model, idxes):
        self.model_list[model] = idxes

    def on_state(self, is_play):
        if is_play:
            read_idxes = np.arange(1, self.fps+1)
            model_idxes = dict([(int(i), []) for i in np.arange(1, self.fps+1)])
            for model, idxes in self.model_list.items():
                for i in conv_spaced_list(read_idxes, idxes, is_end=False):
                    if model not in model_idxes[int(i)]:
                        model_idxes[int(i)].append(model)
            self.model_idxes = model_idxes

            write_idxes = np.arange(1, self.fps+1)
            self.write_idxes = conv_spaced_list(write_idxes, WRITE_FPS, is_end=False)

    def init_time(self):
        date = datetime.now().strftime('%Y-%m-%d')
        datetime_date = datetime.strptime(date, '%Y-%m-%d')
        week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        r = datetime_date.weekday()

        now_time = datetime.now().strftime('%H:%M:%S').split(':')

        self.week = week[r]
        self.hour = int(now_time[0])
        self.minute = int(now_time[1])

    def is_in_schedule(self):
        start_time = self.schedule[self.week][0].split(':')
        end_time = self.schedule[self.week][1].split(':')

        h1, m1 = int(start_time[0]), int(start_time[0])
        h2, m2 = int(end_time[0]), int(end_time[0])

        is_in_hour = int(start_time[0]) <= self.hour <= int(end_time[0])
        is_in_time = False
        if is_in_hour:
            is_start = False
            if self.hour == int(start_time[0]):
                if self.minute >= m1:
                    is_start = True
            else:
                is_start = True

            is_end = False
            if self.hour == int(end_time[0]):
                if self.minute <= m2:
                    is_end = True
            else:
                is_end = True
            if is_start and is_end:
                is_in_time = True
        return is_in_time


class SystemControl():
    def __init__(self):
        self.is_system_stopped = False

        self.message_type = ''
        self.data_info = {}
        self.camera_info = {'camera1': {}, 'camera2': {}, 'camera3': {}, 'camera4': {}}

        self.rtsp_url = {}

        self.is_first = True
        self.file = None

        self.is_load = False
        self.device_id = ''

    def init_setting(self):
        try:
            files = os.listdir(CAMERA_SETTING_FOLDER)

            json_list = []
            for file in files:
                ext = os.path.splitext(file)[-1]
                if ext == '.json':
                    json_list.append(file)

            if len(json_list) > 0:
                self.file = natsort.natsorted(json_list)[-1]
        except:
            pass

        if self.file is not None:
            self.file_split = self.file.split('-')

            self.device_id = ''
            for i in range(4):
                self.device_id += self.file_split[i]
                if i < 3:
                    self.device_id += '-'
            try:
                with open(os.path.join(CAMERA_SETTING_FOLDER, self.file)) as json_file:
                    json_data = json.load(json_file)

                    for key1 in json_data:
                        if key1 == 'message_type' and self.is_first :
                            self.message_type = json_data[key1]
                        elif key1 == 'data':
                            json_data2 = json_data[key1]['payload']
                            for key2 in json_data2:
                                if 'camera' not in key2:
                                    self.data_info[key2] = json_data2[key2]
                                else:
                                    self.camera_info[key2] = json_data2[key2]
                self.is_load = True
            except Exception as ex:
                if IS_PRINT_DEBUG:
                    print('JSON READ ERROR >> ', ex)
                pass

while True:
    system = SystemControl()
    system.init_setting()

    if system.is_load:
        break
    print('Could not find JSON file with camera information.')
    print('Please check the "{}" PATH'.format(CAMERA_SETTING_FOLDER))
    time.sleep(1)

camera_info = system.camera_info

yolox_detector = YoloxDetector()
yolox_detector.init_model(model='fire', filename='model.py', thresh=0.01)
yolox_detector.start()

# video_writer = DetectWriter()
# video_writer.start()

is_json_time = time.time()
name_list = []
works = {}
is_json_check = False
debug_time = time.time()
ping_time = time.time()

while True:
    dt = time.time()
    if time.time() - is_json_time >= JSON_CHECK_TIME:
        system.camera_info = {'camera1': {}, 'camera2': {}, 'camera3': {}, 'camera4': {}}
        system.init_setting()
        try:
            camera_info = system.camera_info
            is_json_check = True
        except Exception as ex:
            print('json file format error >> ', ex)
            time.sleep(1)
        is_json_time = time.time()
        if IS_PRINT_DEBUG:
            print('C2D FILE NAME : {}\n'.format(system.file))

    if is_json_check:
        for name, streamer in list(camera_info.items()):
            if name in name_list:
                if len(streamer) > 0:
                    if camera_info[name]['rtsp_url'] == '':
                        if name in works:
                            print('The ID with the name of {} disappeared from the camera information JSON file\n'.format(name))
                            works[name].is_stop = True
                            time.sleep(1)

                            name_list.remove(name)
                            works.pop(name)
                        else:
                            name_list.remove(name)
                    else:
                        if name in works:
                            works[name].clear(camera_info[name], is_play=False, is_url_change=True)
                            works[name].meta = camera_info[name]
                        else:
                            name_list.remove(name)
        if IS_PRINT_DEBUG:
            print('The system setting has updated\n')
        is_json_check = False

    for name, streamer in list(camera_info.items()):
        if len(streamer) > 0:
            if name not in name_list:
                print('Find camera ID {} and start connecting.'.format(streamer['camera_id']))
                print('{}\n'.format(camera_info[name]))
                if camera_info[name]['rtsp_url'] != '':
                    name_list.append(name)
                    works[name] = BomWorker(
                        url = camera_info[name]['rtsp_url'],
                        work_name = name,
                        cam_number = name[-1],
                        server_info = system.data_info,
                        meta = camera_info[name],
                        detector = yolox_detector,
                        analyzer = DetectAnalyzer(),
                        writer = VideoWriter(),
                    )
                    works[name].init_analyzer()
                    works[name].init_writer()
                    works[name].init_idxes(name, DETECT_CNT)
                    works[name].start()
                    time.sleep(1)
        else:
            if name in name_list:
                print('The ID with the name of {} disappeared from the camera information JSON file\n'.format(name))
                works[name].is_stop = True
                time.sleep(1)

                name_list.remove(name)
                works.pop(name)

    if IS_DISPLAY:
        for name, streamer in list(works.items()):
            if streamer.disp_frame is not None:
                disp_frame = streamer.disp_frame.copy()
                dt_frame = disp_frame.copy()

                if len(streamer.area) > 0:
                    for i, area in enumerate(streamer.area):
                        pts = np.array(area, np.int32)
                        if len(pts) > 0:
                            if USE_AREA_COLOR:
                                cv2.polylines(dt_frame, [pts], True, AREA_COLOR[i], 2)
                            else:
                                cv2.polylines(dt_frame, [pts], True, (133, 233, 127), 2)

                if len(streamer.results) > 0:
                    for result in streamer.results:
                        xmin, ymin, xmax, ymax = result['bbox']
                        cv2.rectangle(dt_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

                dt_frame  = cv2.resize(dt_frame, (1280, 720))
                disp_frame  = cv2.resize(disp_frame, (1280, 720))


                cv2.imshow(name, dt_frame)
                cv2.waitKey(1)

    time.sleep(0.0001)

    if time.time() - ping_time > PING_SEND_TIME:
        print('system ping')
        ping_time = time.time()
