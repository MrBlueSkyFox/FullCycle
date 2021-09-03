import os
import re
import sys
import gi
import time

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
# # os.environ["GST_DEBUG_FILE"] = log_place
# os.putenv("GST_DEBUG_DUMP_DIR_DIR", l)
from constant import ACL_MEMCPY_DEVICE_TO_HOST, check_ret
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject


"""
1)
gst-launch-1.0 filesrc location=test.h264 ! video/x-h264 ! h264parse ! mp4mux ! filesink location=video.mp4
gst-launch-1.0 filesrc location=test.h264 ! video/x-h264,framerate=5/1 ! \
    h264parse ! mp4mux ! filesink location=video.mp4


gst-launch-1.0 -e appsrc emit-signals=True is-live=True \
 caps=video/x-raw,format=RGB,width=640,height=480,framerate=30/1 ! \
 queue max-size-buffers=4 ! mp4mux name=m_mp4mux ! filesink location=video.mp4
"""


def ndarray_to_gst_buffer(array: np.ndarray) -> Gst.Buffer:
    """Converts numpy array to Gst.Buffer"""
    return Gst.Buffer.new_wrapped(array.tobytes())


import code

# video writter
# gst_string = (
#             f'appsrc ! videoconvert n-threads=4 ! '
#             f'video/x-raw,width={w},height={h} ! queue ! '
#             # f'vaapih264enc rate-control=cbr bitrate={self.BITRATE} quality-level=6 ! '
#             f'x264enc bitrate={self.BITRATE} speed-preset=fast ! '
#             'video/x-h264,profile=main ! queue ! '
#             'h264parse ! mpegtsmux ! queue ! '
#             f'hlssink location={os.path.join(self.LIVE_PATH, "segment-%d.ts")} '
#             f'max-files={self.LIVE_VIDEOS_LIMIT} '
#             # f'max-files=1000 '
#             f'playlist-length={self.LIVE_PLAYLIST_LIMIT} '
#             f'playlist-location={os.path.join(self.LIVE_PATH, "playlist.m3u8")} '
#             f'target-duration={settings.SEC_INTERVAL}'
#         )


# format_save 
# 1 HSL type of save,multiply files 2 sec each
# other mp4 one file
class GstConvertVideo:
    def __init__(self, out_place="", format_save = 1, dict_data={}) -> None:
        self.SEC_INTERVAL = 2
        self.LIVE_VIDEOS_LIMIT = 20 // self.SEC_INTERVAL
        self.LIVE_PLAYLIST_LIMIT = self.LIVE_VIDEOS_LIMIT // 2
        self.fps = dict_data["fps_out_test"]
        if int(format_save) == 1:
            self.pipeline = Gst.parse_launch(
                f"appsrc emit-signals=True is-live=True name=m_appsrc ! "
                "video/x-h264,stream-format=(string)byte-stream,alignment=au ! "
                "h264parse ! mpegtsmux name=m_mux ! queue ! "
                f"hlssink location={os.path.join(out_place, 'segment-%d.ts')} "
                f"max-files={self.LIVE_VIDEOS_LIMIT} "
                f"playlist-length={self.LIVE_PLAYLIST_LIMIT} "
                f'playlist-location={os.path.join(out_place, "playlist.m3u8")} '
                f"target-duration={self.SEC_INTERVAL}"
            )
        else:
            out_place = os.path.join(out_place,"out.mp4")
            self.pipeline = Gst.parse_launch(
                f"appsrc emit-signals=True is-live=True name=m_appsrc ! "
                "video/x-h264,stream-format=(string)byte-stream,alignment=au ! "
                "mpegtsmux ! queue ! "
                f"filesink location={out_place}"
            )
        self.src = self.pipeline.get_by_name("m_appsrc")
        caps = Gst.caps_from_string(
            "video/x-h264, stream-format=(string)byte-stream, alignment=au"
        )
        # caps = Gst.caps_from_string("video/x-raw,format=RGB,width=640,height=480,framerate=0/1")
        self.src.set_property("format", Gst.Format.TIME)
        self.src.set_property("block", True)
        self.src.set_property("caps", caps)
        self.src.set_property("stream-type", 0)


        self.pts = 0
        # self.duration = 10 ** 9 / (12 / 1)
        self.duration = 10 ** 9 / (self.fps / 1)
        Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, "pipeline")
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.enable_sync_message_emission()
        self.bus.connect("message", self.bus_messages)
        self.bus.connect("sync-message::element", self.bus_messages)
        print(f"VideoWriter fps= {self.fps} out_place = {out_place}")
        # code.interact(local=locals())

    def _runGstPipeline(self):
        # Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL,
        #                               "pipeline.dot")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        Gst.debug_bin_to_dot_file(
            self.pipeline, Gst.DebugGraphDetails.ALL, "pipeline_start"
        )
        if ret == Gst.StateChangeReturn.FAILURE:
            print("run gst don't run")
            return False
        return True

    def _stopGstPipeline(self):
        ret = self.pipeline.set_state(Gst.State.PAUSED)
        if ret == Gst.StateChangeReturn.FAILURE:
            # logger.warning('Can`t stop player after probe')
            return False
        Gst.debug_bin_to_dot_file(
            self.pipeline, Gst.DebugGraphDetails.ALL, "pipeline_stop"
        )
        return True

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
                # if message.type == Gst.MessageType.STREAM_STATUS:
                # logger.debug(
                #     f"Stream Status msg ={message.get_stream_status_object().get_state()}"
                # )

    def frame_work(self, img):
        # print("frame work")
        gst_buffer = ndarray_to_gst_buffer(img)
        self.pts += self.duration
        gst_buffer.pts = self.pts
        gst_buffer.duration = self.duration
        # print("create buffer")
        self.src.emit("push-buffer", gst_buffer)
        # print("emit buffer")

    def write_arrays(self, img_array):
        l = self._runGstPipeline()
        print(f"set run suc {l}")
        for img in img_array:
            print("write arra")
            msg = self.bus.pop()
            self.bus_messages(self.bus, msg)
            self.frame_work(img)
        self.src.emit("end-of-stream")
        print("emit")
        self._stopGstPipeline()

    def end_work(self):
        self.src.emit("end-of-stream")
        self._stopGstPipeline()


if __name__ == "__main__":
    Gst.init(None)
    Gst.debug_set_active(True)
    Gst.debug_set_default_threshold(3)
    pass

# NUM_BUFFERS = 40
# WIDTH, HEIGHT = 768, 432
# WIDTH, HEIGHT = 640, 480
# CHANNELS = 3
# DTYPE = np.uint8
# pts = 0
# duration = 10 ** 9 / (12 / 1)
# b = GstConvertVido()
# b._runGstPipeline()
# array = np.random.randint(
#         low=0, high=255, size=(HEIGHT, WIDTH, CHANNELS), dtype=DTYPE
#     )
# for _ in range(NUM_BUFFERS):
#     array = np.random.randint(
#         low=0, high=255, size=(HEIGHT, WIDTH, CHANNELS), dtype=DTYPE
#     )
#     gst_buffer = ndarray_to_gst_buffer(array)
#     pts += duration
#     gst_buffer.pts = pts
#     gst_buffer.duration = duration
#     b.src.emit("push-buffer", gst_buffer)
#     # b.src.emit("push-buffer", ndarray_to_gst_buffer(array))
# b.src.emit("end-of-stream")
# b._stopGstPipeline()
# print("b")
# sys.exit(0)
