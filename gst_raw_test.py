import os
import re
import threading
import gi
import time
import collections
import ffmpeg

from constant import (
    ACL_MEMCPY_DEVICE_TO_HOST,
    check_ret,
    EncdingType,
)

# l = "/home/tigran/trash/Monster_acl/core/test/acl_test/pipe_debug"
# log_place = "/home/tigran/trash/Monster_acl/test_dir/test/video/in_out_264"
# try:
#     import acl

#     l = "/app/test_dir/test/acl_test/pipe_debug"
#     log_place = "/app/test_dir/test/video/in_out_264/log264.log"
# except:
#     print("home comp")
#     pass
# os.environ["GST_DEBUG_DUMP_DOT_DIR"] = l
gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)
Gst.debug_set_active(True)
Gst.debug_set_default_threshold(1)

H264_PROFILES = {
    "Baseline": EncdingType.H264_BASELINE_LEVEL.value,
    "High": EncdingType.H264_HIGH_LEVEL.value,
    "Main": EncdingType.H264_MAIN_LEVEL.value,
    "Main264": EncdingType.H265_MAIN_LEVEL.value,
}


class GstBase:
    def _runGstPipeline(self):
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            # logger.warning("Can`t start player")
            return False
        return True

    def _stopGstPipeline(self):
        ret = self.pipeline.set_state(Gst.State.PAUSED)
        if ret == Gst.StateChangeReturn.FAILURE:
            # logger.warning("Can`t stop player")
            return False
        return True

    def _connect_bus(self):
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.enable_sync_message_emission()
        self.bus.connect("message", self.bus_messages)
        self.bus.connect("sync-message::element", self.bus_messages)

    def bus_messages(self, bus, message):
        if message:
            if message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                print(
                    "Error received from element %s: %s" % (message.src.get_name(), err)
                )
                print("Debugging information: %s" % debug)
                # logger.debug(
                #     f"Error received from element {message.src.get_name()}: {err}\n Debugging inf {debug}"
                # )
            elif message.type == Gst.MessageType.EOS:
                # logger.info("End-Of-Stream reached")
                print("End-Of-Stream reached")

            elif message.type == Gst.MessageType.STATE_CHANGED:
                if isinstance(message.src, Gst.Pipeline):
                    old_state, new_state, pending_state = message.parse_state_changed()
                    s = old_state.value_nick
                    s2 = new_state.value_nick
                    print(
                        "Pipeline state changed from %s to %s."
                        % (old_state.value_nick, new_state.value_nick)
                    )
                    # logger.info(f"Pipeline state changed from {s} to {s2}")
            else:
                print("Unexpected message received.")
                # logger.info(f"messageType= {message.type}")
                if message.type == Gst.MessageType.STREAM_STATUS:
                    # logger.debug(
                    #     f"Stream Status msg ={message.get_stream_status_object().get_state()}"
                    # )
                    pass

    def get(self):
        """return buffer"""
        pass


class GstRtspReciver(GstBase):
    def __init__(self, video_path, force_fps="") -> None:
        # super().__init__()
        # if len(video_path) < 4:
        #     mock_stream = "rtsp://admin:admin123@192.168.0.108:554"
        #     video_path = mock_stream
        self.video_path = video_path
        if "rtsp://" in video_path:
            self.pipeline = Gst.parse_launch(
                f"rtspsrc location={video_path} latency=100 name=m_src ! "
                "rtph264depay name=rtph ! "
                "appsink emit-signals=true name=m_appsink"
            )
        else:
            self.pipeline = Gst.parse_launch(
                f"filesrc location={video_path} ! "
                "qtdemux name=demux demux.video_0 ! queue ! h264parse ! "
                "appsink emit-signals=true name=m_appsink"
            )
        sink = self.pipeline.get_by_name("m_appsink")
        caps = Gst.caps_from_string(
            "video/x-h264, stream-format=(string)byte-stream, alignment=au"
        )
        # caps = Gst.caps_from_string("video/x-h264, stream-format=(string)avc")
        sink.set_property("caps", caps)
        sink.connect("new-sample", self.new_buffer, sink)
        self._connect_bus()
        count = 0
        while True:
            try:
                count += 1
                probe = ffmpeg.probe(video_path)
                video_info = next(
                    s for s in probe["streams"] if s["codec_type"] == "video"
                )
                fps = video_info["avg_frame_rate"]
                self.fps = float(fps.split("/")[0])
                self.FRAME_WIDTH = int(video_info["width"])
                self.FRAME_HEIGHT = int(video_info["height"])
                print(f"self.FRAME_WIDTH {self.FRAME_WIDTH}")
                try:
                    # check if codec 265
                    if video_info["codec_name"] == "hevc":
                        self.CODEC_PROFILE = H264_PROFILES["Main264"]
                    else:
                        self.CODEC_PROFILE = H264_PROFILES[video_info["profile"]]
                except:
                    self.CODEC_PROFILE = H264_PROFILES["Main"]

                    # logger.info(f"FPS was set forcefull {settings.FORCE_FPS}")
                if force_fps != "":
                    self.fps = int(force_fps)
                if 1 < self.fps < 60:
                    # logger.info(f"fps = video_info['r_frame_rate'] {self.fps}")
                    break
                fps = video_info["r_frame_rate"]
                self.fps = float(fps.split("/")[0])
                if 1 < self.fps < 60:
                    break
            except ffmpeg._run.Error:
                if count > 3:
                    self.fps = 12
                    break
            except:
                self.fps = 12
                break
        # settings.INPUT_FRAMERATE = settings.OUTPUT_FRAME = self.fps

        print(f"BatchController2 fps= {self.fps}")

        self.last_data = time.time()
        self.drain = collections.deque(maxlen=60)
        # self.emit_buffer = collections.deque(maxlen=60)
        # self.emit_timings = collections.deque(maxlen=60)
        self._data = []
        self._timings = []
        # self._batch_id = 0
        self._stopped = threading.Event()
        # self._has_data = threading.Event()

    def new_buffer(self, sink, _):
        sample = sink.emit("pull-sample")
        t = time.time()
        buffer = sample.get_buffer()
        self.drain.append((buffer, t))
        return Gst.FlowReturn.OK

    def get(self):
        if len(self.drain) > 0:
            return self.drain.popleft()
        else:
            return None, None


class GstEater:
    def __init__(self, video_path, mode=1) -> None:
        self.fps, self.WIDTH, self.HEIGHT, self.CODEC = self.__get_stream_metadata(
            video_path
        )
        self.rtsp = "rtsp://171.25.232.232:554/57a748fba81a45e0a9fba696c3264d07"
        self.rtsp2 = "rtsp://171.25.232.45:554/oBAAMizlVADtXYyJvLim01mRyifKfi"
        mock_vid = "/app/test_dir/test/video/face-demographics-walking-and-pause.mp4"
        self.mode = mode
        self.video_path = video_path
        if mode == 1:
            self.pipeline = Gst.parse_launch(
                f"filesrc location={video_path} name=m_src ! "
                "appsink emit-signals=true name=m_appsink"
            )
        elif mode == 2:
            # pass
            self.pipeline = Gst.parse_launch(
                f"rtspsrc location={video_path} do-rtsp-keep-alive=true tcp-timeout=0 name=m_src ! "
                "rtph264depay name=rtph ! "
                "appsink emit-signals=true name=m_appsink"
            )
        elif mode == 3:
            self.pipeline = Gst.parse_launch(
                f"rtspsrc location={video_path} do-rtsp-keep-alive=true tcp-timeout=0 name=m_src ! "
                "rtph264depay name=rtph ! h264parse name=parser ! "
                "appsink emit-signals=true name=m_appsink"
            )
        elif mode == 4:
            self.pipeline = Gst.parse_launch(
                f"filesrc location={mock_vid} ! "
                "qtdemux name=demux demux.video_0 ! queue ! h264parse ! "
                "appsink emit-signals=true name=m_appsink"
            )
        elif mode == 6:
            self.pipeline = Gst.parse_launch(
                f"filesrc location={mock_vid} ! "
                "appsink emit-signals=true name=m_appsink"
            )
        elif mode == 5:
            # not working
            self.pipeline = Gst.parse_launch(
                f"filesrc location={mock_vid} ! "
                "h264parse ! "
                "appsink emit-signals=true name=m_appsink"
            )
        # self.pipeline = Gst.parse_launch(
        #         f"rtspsrc location={self.rtsp} do-rtsp-keep-alive=true tcp-timeout=0 name=m_src ! "
        #         "rtph264depay name=rtph ! "
        #         "appsink emit-signals=true name=m_appsink")
        sink = self.pipeline.get_by_name("m_appsink")
        caps = Gst.caps_from_string(
            "video/x-h264, stream-format=(string)byte-stream, alignment=au"
        )
        # caps = Gst.caps_from_string("video/x-h264, stream-format=(string)avc")
        sink.set_property("caps", caps)
        sink.connect("new-sample", self.new_buffer, sink)
        self.video_name = self.video_path[self.video_path.rfind("/") + 1 :]
        name_file = f"RTSP_init_{self.mode}_{self.video_name}"
        Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, name_file)
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.enable_sync_message_emission()
        self.bus.connect("message", self.bus_messages)
        self.bus.connect("sync-message::element", self.bus_messages)
        # Gst.debug_bin_to_dot_file((self.pipeline), Gst.DebugGraphDetails.ALL, "/home/tigran/work_from_home/Monster_acl/core/test/acl_test/pipeline")
        # code.InteractiveConsole(vars).interact()
        self.end = False
        self.emit_buffer = collections.deque(maxlen=120)
        self.emit_timings = collections.deque(maxlen=120)

        # ret = self.pipeline.set_state(Gst.State.PLAYING)
        # if ret == Gst.StateChangeReturn.FAILURE:
        #     print("ERROR !!!!!\nCAN't start play")

        # print(ret)
        # cur_time = time.time()
        # print(cur_time)
        # out_time = cur_time + 1 * 60
        # while out_time > cur_time:
        #     cur_time = time.time()

    def get_buffer(self):
        return self.emit_buffer

    def new_buffer(self, sink, _):
        sample = sink.emit("pull-sample")
        t = time.time()
        buffer = sample.get_buffer()
        self.emit_buffer.append(buffer)
        self.emit_timings.append(t)
        # self.have_data.set()
        # print("Sample type:{}\nBuffer:{}\ntime:{}".format(type(sample), buffer, str(t)))
        # print("Gst buffer_size:{}".format(buffer.get_size()))
        return Gst.FlowReturn.OK

    def get_try(self):
        try:
            p = self.emit_buffer.popleft(), self.emit_timings.popleft()
            return p
        except:
            return None, None

    def bus_messages(self, bus, message):
        if message:
            if message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                print(
                    "Error received from element %s: %s" % (message.src.get_name(), err)
                )
                print("Debugging information: %s" % debug)
            elif message.type == Gst.MessageType.EOS:
                print("End-Of-Stream reached.")
                self.end = True
            elif message.type == Gst.MessageType.STATE_CHANGED:
                if isinstance(message.src, Gst.Pipeline):
                    old_state, new_state, pending_state = message.parse_state_changed()
                    s = old_state.value_nick
                    s2 = new_state.value_nick
                    print(
                        "Pipeline state changed from %s to %s."
                        % (old_state.value_nick, new_state.value_nick)
                    )
            else:
                print("Unexpected message received.")

    def _runGstPipeline(self):
        print("set to play")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Can`t set gst pipepline to PLAYING\n")
            return False
        name_file = f"RTSP_start_{self.mode}_{self.video_name}"
        Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, name_file)

        return True

    def _stopGstPipeline(self):
        print("set to stop")
        ret = self.pipeline.set_state(Gst.State.PAUSED)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Can`t set gst pipepline to PLAYING\n")
            return False
        Gst.debug_bin_to_dot_file(
            self.pipeline, Gst.DebugGraphDetails.ALL, "pipelineRTSP_stop"
        )
        return True

    def __get_stream_metadata(self, video_path):
        DATA_WAITING_TIMEOUT = 2
        fps = None
        FRAME_WIDTH = None
        FRAME_HEIGHT = None
        CODEC_PROFILE = None
        while True:
            try:
                probe = ffmpeg.probe(video_path)
                video_info = next(
                    s for s in probe["streams"] if s["codec_type"] == "video"
                )
                fps = video_info["avg_frame_rate"]
                fps = float(fps.split("/")[0])
                FRAME_WIDTH = int(video_info["width"])
                FRAME_HEIGHT = int(video_info["height"])
                try:
                    CODEC_PROFILE = H264_PROFILES[video_info["profile"]]
                except:
                    CODEC_PROFILE = H264_PROFILES["Main"]
                if 1 < fps < 100:
                    return fps, FRAME_WIDTH, FRAME_HEIGHT, CODEC_PROFILE
                    break
                fps = video_info["r_frame_rate"]
                fps = float(fps.split("/")[0])
                if 1 < fps < 100:
                    return fps, FRAME_WIDTH, FRAME_HEIGHT, CODEC_PROFILE
                    break
            except ffmpeg._run.Error:
                # logger.warning("Warning BatchController ffprobe error")
                time.sleep(DATA_WAITING_TIMEOUT)


class GstEaterVideo:
    def __init__(self, video_path, name="") -> None:
        # threading.Thread.__init__(self, name=name)
        self.fps, self.WIDTH, self.HEIGHT, self.CODEC = self.__get_stream_metadata(
            video_path
        )
        if len(video_path) < 4:
            mock_vid = (
                "/app/test_dir/test/video/face-demographics-walking-and-pause.mp4"
            )
            video_path = mock_vid
        self.video_path = video_path
        self.pipeline = Gst.parse_launch(
            f"filesrc location={video_path} ! "
            "qtdemux name=demux demux.video_0 ! queue ! h264parse ! "
            "appsink emit-signals=true name=m_appsink"
        )
        sink = self.pipeline.get_by_name("m_appsink")
        caps = Gst.caps_from_string(
            "video/x-h264, stream-format=(string)byte-stream, alignment=au"
        )
        # caps = Gst.caps_from_string("video/x-h264, stream-format=(string)avc")
        sink.set_property("caps", caps)
        sink.connect("new-sample", self.new_buffer, sink)

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.enable_sync_message_emission()
        self.bus.connect("message", self.bus_messages)
        self.bus.connect("sync-message::element", self.bus_messages)

        self.last_data = time.time()
        self.emit_buffer = collections.deque()
        self.emit_timings = collections.deque()
        self._data = []
        self._timings = []
        # self._batch_id = 0
        self._stopped = threading.Event()
        self.have_data = threading.Event()

    def _runGstPipeline(self):
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("run gst don't run")
            return False
        return True

    def _stopGstPipeline(self):
        ret = self.pipeline.set_state(Gst.State.PAUSED)
        if ret == Gst.StateChangeReturn.FAILURE:
            # logger.warning('Can`t stop player after probe')
            return False
        return True

    def new_buffer(self, sink, _):
        sample = sink.emit("pull-sample")
        t = time.time()
        buffer = sample.get_buffer()
        self.emit_buffer.append(buffer)
        self.emit_timings.append(t)
        self.have_data.set()
        # print("Sample type:{}\nBuffer:{}\ntime:{}".format(type(sample), buffer, str(t)))
        # print("Gst buffer_size:{}".format(buffer.get_size()))
        return Gst.FlowReturn.OK

    def bus_messages(self, bus, message):
        if message:
            if message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                print(
                    "Error received from element %s: %s" % (message.src.get_name(), err)
                )
                print("Debugging information: %s" % debug)
                # logger.debug(
                #     f"Error received from element {message.src.get_name()}: {err}\n Debugging inf {debug}"
                # )
            elif message.type == Gst.MessageType.EOS:
                # logger.info("End-Of-Stream reached")
                print("End-Of-Stream reached")

            elif message.type == Gst.MessageType.STATE_CHANGED:
                if isinstance(message.src, Gst.Pipeline):
                    old_state, new_state, pending_state = message.parse_state_changed()
                    s = old_state.value_nick
                    s2 = new_state.value_nick
                    print(
                        "Pipeline state changed from %s to %s."
                        % (old_state.value_nick, new_state.value_nick)
                    )
                    # logger.info(f"Pipeline state changed from {s} to {s2}")
            else:
                print("Unexpected message received.")
                # logger.info(f"messageType= {message.type}")
                if message.type == Gst.MessageType.STREAM_STATUS:
                    # logger.debug(
                    #     f"Stream Status msg ={message.get_stream_status_object().get_state()}"
                    # )
                    pass

    def get_buffer(self):
        return self.emit_buffer.popleft()

    def get_times(self):
        return self.emit_timings.popleft()

    def get(self):
        if len(self.emit_buffer) > 0 and len(self.emit_timings) > 0:
            return self.emit_buffer.popleft(), self.emit_timings.popleft()
        else:
            return None, None

    def get_try(self):
        try:
            p = self.emit_buffer.popleft(), self.emit_timings.popleft()
            return p
        except:
            return None, None

    def __get_stream_metadata(self, video_path):
        DATA_WAITING_TIMEOUT = 2
        fps = None
        FRAME_WIDTH = None
        FRAME_HEIGHT = None
        CODEC_PROFILE = None
        while True:
            try:
                probe = ffmpeg.probe(video_path)
                video_info = next(
                    s for s in probe["streams"] if s["codec_type"] == "video"
                )
                fps = video_info["avg_frame_rate"]
                fps = float(fps.split("/")[0])
                FRAME_WIDTH = int(video_info["width"])
                FRAME_HEIGHT = int(video_info["height"])
                try:
                    CODEC_PROFILE = H264_PROFILES[video_info["profile"]]
                except:
                    CODEC_PROFILE = H264_PROFILES["Main"]
                if 1 < fps < 100:
                    return fps, FRAME_WIDTH, FRAME_HEIGHT, CODEC_PROFILE
                    break
                fps = video_info["r_frame_rate"]
                fps = float(fps.split("/")[0])
                if 1 < fps < 100:
                    return fps, FRAME_WIDTH, FRAME_HEIGHT, CODEC_PROFILE
                    break
            except ffmpeg._run.Error:
                # logger.warning("Warning BatchController ffprobe error")
                time.sleep(DATA_WAITING_TIMEOUT)


def test_mp4_vid(args):
    gst_proc = GstEater("", mode=4)
    gst_proc._runGstPipeline()
    from vdec import Vdec

    out_time = time.time() + 0.2 * 60
    while out_time > time.time():
        pass
    import acl  # type: ignore

    import numpy as np

    ar_buffer = gst_proc.get_buffer()
    ar = ar_buffer[0]
    size = ar.get_size()

    img = np.ndarray((size,), buffer=ar.extract_dup(0, size), dtype=np.uint8)
    test_out_name = "venc_out/test.yuv"
    img.tofile(test_out_name)
    acl.init()
    acl.rt.set_device(0)
    context, ret = acl.rt.create_context(0)

    vdec_proc = Vdec(context=context, width=768, height=432)
    v_dict = {"img": test_out_name, "w": 768, "h": 432, "dtype": np.uint8}
    vdec_proc.run_stock(v_dict)
    img_desc = vdec_proc.get_image_buffer()
    img_desc = img_desc[0]
    np_output = np.zeros(img_desc["size"], dtype=np.byte)
    np_output_ptr = acl.util.numpy_to_ptr(np_output)
    ret = acl.rt.memcpy(
        np_output_ptr,
        img_desc["size"],
        img_desc["buffer"],
        img_desc["size"],
        ACL_MEMCPY_DEVICE_TO_HOST,
    )
    check_ret("acl.rt.memcpy", ret)
    out_file = "venc_out/test2.yuv"
    np_output.tofile(out_file)
    conv_img()
    return 0


def conv_img(img_file="venc_out/test2.yuv", w=768, h=432):
    import cv2
    import numpy as np

    np_out = np.fromfile(img_file, dtype=np.uint8)
    yuv = np_out.reshape((int(h * 1.5), w))
    yuv = yuv.astype("uint8")
    bgr_to_save = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    bgr_name = f"venc_out/2bgr.jpg"
    cv2.imwrite((bgr_name), bgr_to_save)

    # code.InteractiveConsole(vars).interact()
