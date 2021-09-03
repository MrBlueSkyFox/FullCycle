import collections
import os
import numpy as np
import acl # type: ignore

from constant import check_ret, EncdingType, ImageType, MallocType, MemcpyType

H265_MAIN_LEVEL = EncdingType.H265_MAIN_LEVEL.value
H264_BASELINE_LEVEL = EncdingType.H264_BASELINE_LEVEL.value
H264_MAIN_LEVEL = EncdingType.H264_MAIN_LEVEL.value
H264_HIGH_LEVEL = EncdingType.H264_HIGH_LEVEL.value

PIXEL_FORMAT_YUV_SEMIPLANAR_420 = (
    ImageType.PIXEL_FORMAT_YUV_SEMIPLANAR_420.value
)  # NV12
PIXEL_FORMAT_YVU_SEMIPLANAR_420 = (
    ImageType.PIXEL_FORMAT_YVU_SEMIPLANAR_420.value
)  # NV21

ACL_MEMCPY_HOST_TO_DEVICE = MemcpyType.ACL_MEMCPY_HOST_TO_DEVICE.value
ACL_MEMCPY_DEVICE_TO_HOST = MemcpyType.ACL_MEMCPY_DEVICE_TO_HOST.value


def check_none(message, ret_none):
    if ret_none is None:
        raise Exception("{} failed".format(message))


class Venc:
    def __init__(
        self,
        context,
        width,
        height,
        enc_type=H264_HIGH_LEVEL,
        key_frame = 12,
        end_point="./venc_out/output.h264",
    ) -> None:
        self.input_stream_mem = None
        self.venc_channel_desc = 0
        self.dvpp_pic_desc = 0
        self.frame_config = None
        self.cb_thread_id = None
        self.pic_host = None
        self.callback_run_flag = True
        self.stream_host = None
        self.context = context
        self._en_type = enc_type
        self._format = PIXEL_FORMAT_YUV_SEMIPLANAR_420
        self.img_buff = []
        self.width = width
        self.height = height
        self.cb_thread_id = None
        if os.path.exists(end_point):
            os.remove(end_point)
        self.end_file = end_point
        self.output_buffer = collections.deque(maxlen=500)
        self.input_buffer = collections.deque(maxlen=500)
        self.counter_out = 0
        self.counter = 0
        self.key_frame_interval = key_frame
        print(f"self.key_frame_interval {self.key_frame_interval}")

    def __del__(self):
        print("[INFO] free resource")
        # if self.stream_host:
        #     acl.rt.free_host(self.stream_host)
        if self.pic_host:
            acl.rt.free_host(self.pic_host)
        if self.input_stream_mem:
            acl.media.dvpp_free(self.input_stream_mem)
        if self.venc_channel_desc != 0:
            acl.media.venc_destroy_channel_desc(self.venc_channel_desc)
        if self.dvpp_pic_desc != 0:
            acl.media.dvpp_destroy_pic_desc(self.dvpp_pic_desc)
        if self.frame_config:
            acl.media.venc_destroy_frame_config(self.frame_config)

    def get_try(self):
        try:
            p = self.output_buffer.popleft()
            return p
        except:
            return None

    def get_input_desc(self):
        try:
            if len(self.input_buffer > 10):
                p = self.input_buffer.popleft()
            else:
                p = None
            return p
        except:
            return None

    def get_try_input_desc(self):
        try:
            if len(self.input_buffer > 2):
                p = self.input_buffer.popleft()
            else:
                p = None
            return p
        except:
            return None

    def venc_init_single(self):
        self.venc_channel_desc = acl.media.venc_create_channel_desc()
        check_none("acl.media.venc_create_channel_desc", self.venc_channel_desc)
        timeout = 1000
        self.cb_thread_id, ret = acl.util.start_thread(
            self.cb_thread_func, [self.context, timeout]
        )
        check_ret("acl.util.start_thread", ret)
        print("[VENC][INFO] start_thread", self.cb_thread_id, ret)
        self.venc_set_desc(self.width, self.height)
        print("[VENC][INFO] set venc channel desc")
        ret = acl.media.venc_create_channel(self.venc_channel_desc)
        check_ret("acl.media.venc_create_channel", ret)

    def venc_init(self):
        self.venc_channel_desc = acl.media.venc_create_channel_desc()
        check_none("acl.media.venc_create_channel_desc", self.venc_channel_desc)

    def cb_thread_func(self, args_list):
        context = args_list[0]
        timeout = args_list[1]
        print(
            "[VENC] [INFO] cb_thread_func args_list = ",
            context,
            timeout,
            self.callback_run_flag,
        )

        ret = acl.rt.set_context(self.context)
        if ret != 0:
            print("[INFO] cb_thread_func acl.rt.set_context ret=", ret)
            return
        while self.callback_run_flag is True:
            ret = acl.rt.process_report(timeout)

        # print("LWE\n")
        # ret = acl.rt.destroy_context(context)
        # print("[VENC][INFO] cb_thread_func acl.rt.destroy_context ret=", ret)

    def callback_func(self, input_pic_desc, output_stream_desc, user_data):
        if output_stream_desc == 0:
            print("[INFO] [venc] output_stream_desc is null")
            return
        stream_data = acl.media.dvpp_get_stream_desc_data(output_stream_desc)
        if stream_data is None:
            print("[INFO] [venc] acl.media.dvpp_get_stream_desc_data is none")
            return
        ret = acl.media.dvpp_get_stream_desc_ret_code(output_stream_desc)
        if ret == 0:
            stream_data_size = acl.media.dvpp_get_stream_desc_size(output_stream_desc)
            # print("[INFO] [venc] stream_data size", stream_data_size)
            # stream memcpy d2h
            np_data = np.zeros(stream_data_size, dtype=np.byte)
            np_data_ptr = acl.util.numpy_to_ptr(np_data)
            ret = acl.rt.memcpy(
                np_data_ptr,
                stream_data_size,
                stream_data,
                stream_data_size,
                ACL_MEMCPY_DEVICE_TO_HOST,
            )
            if ret != 0:
                print("[INFO] [venc] acl.rt.memcpy ret=", ret)
                return
            # self.img_buff.append(np_data)
            # with open(self.end_file, "ab") as f:
            #     f.write(np_data)
            self.output_buffer.append(np_data)
            self.counter_out += 1

            # data_dev = acl.media.dvpp_get_stream_desc_data(input_pic_desc)
            self.input_buffer.append(input_pic_desc)

    def venc_set_desc(self, width, height):
        # venc_channel_desc set function
        acl.media.venc_set_channel_desc_thread_id(
            self.venc_channel_desc, self.cb_thread_id
        )

        acl.media.venc_set_channel_desc_callback(
            self.venc_channel_desc, self.callback_func
        )
        acl.media.venc_set_channel_desc_entype(self.venc_channel_desc, self._en_type)
        acl.media.venc_set_channel_desc_pic_format(self.venc_channel_desc, self._format)
        acl.media.venc_set_channel_desc_key_frame_interval(
            self.venc_channel_desc, self.key_frame_interval
        )
        acl.media.venc_set_channel_desc_pic_height(self.venc_channel_desc, height)
        acl.media.venc_set_channel_desc_pic_width(self.venc_channel_desc, width)

    def venc_set_frame_config(self, frame_confg, eos, iframe):
        acl.media.venc_set_frame_config_eos(frame_confg, eos)
        acl.media.venc_set_frame_config_force_i_frame(frame_confg, iframe)

    def venc_get_frame_config(self, frame_confg):
        get_eos = acl.media.venc_get_frame_config_eos(frame_confg)
        check_ret("acl.media.venc_get_frame_config_eos", get_eos)
        get_force_frame = acl.media.venc_get_frame_config_force_i_frame(frame_confg)
        check_ret("acl.media.venc_get_frame_config_force_i_frame", get_force_frame)

    def venc_process_one_image(self, input_mem, input_size):
        # set picture description
        self.dvpp_pic_desc = acl.media.dvpp_create_pic_desc()
        check_none("acl.media.dvpp_create_pic_desc", self.dvpp_pic_desc)
        ret = acl.media.dvpp_set_pic_desc_data(self.dvpp_pic_desc, input_mem)
        ret = acl.media.dvpp_set_pic_desc_size(self.dvpp_pic_desc, input_size)
        # print("[INFO] set pic desc size")

        self.venc_set_frame_config(self.frame_config, 0, 0)
        # print("[INFO] set frame config")
        self.venc_get_frame_config(self.frame_config)

        # send frame
        # venc_cnt = 16
        # while venc_cnt:
        #     ret = acl.media.venc_send_frame(
        #         self.venc_channel_desc, self.dvpp_pic_desc, 0, self.frame_config, None
        #     )
        #     check_ret("acl.media.venc_send_frame", ret)
        #     venc_cnt -= 1
        ret = acl.media.venc_send_frame(
            self.venc_channel_desc, self.dvpp_pic_desc, 0, self.frame_config, None
        )
        check_ret("acl.media.venc_send_frame", ret)
        # self.venc_set_frame_config(self.frame_config, 1, 0)

        # send eos frame
        # print("[INFO] venc send frame eos")
        # ret = acl.media.venc_send_frame(
        #     self.venc_channel_desc, 0, 0, self.frame_config, None
        # )
        # print("[INFO] acl.media.venc_send_frame ret=", ret)

        # print("[INFO] venc process success")

    def init_for_one_runs(self):
        timeout = 1000
        self.cb_thread_id, ret = acl.util.start_thread(
            self.cb_thread_func, [self.context, timeout]
        )
        check_ret("acl.util.start_thread", ret)
        print("[VENC][INFO] start_thread", self.cb_thread_id, ret)
        # self.venc_set_desc(width, height)
        self.venc_set_desc(self.width, self.height)
        print("[VENC][INFO] set venc channel desc")
        ret = acl.media.venc_create_channel(self.venc_channel_desc)
        check_ret("acl.media.venc_create_channel", ret)

    def run_one_frame(self, buffer):
        self.frame_config = acl.media.venc_create_frame_config()
        check_none("acl.media.venc_create_frame_config", self.frame_config)
        # print("[VENC][INFO] create_frame_config")
        input_mem, input_size = buffer
        self.venc_process_one_image(input_mem, input_size)

        print("[VENC] [DEBUG] Encoding img end")
        ret = acl.media.dvpp_free(input_mem)
        check_ret("acl.media.dvpp_free", ret)

    def run_one_image(self, buffer: dict):
        self.frame_config = acl.media.venc_create_frame_config()
        check_none("acl.media.venc_create_frame_config", self.frame_config)
        # print("[VENC][INFO] create_frame_config")
        input_mem, input_size = buffer.values()
        self.venc_process_one_image(input_mem, input_size)

        # print("[VENC] [DEBUG] Encoding img end")
        ret = acl.media.dvpp_free(input_mem)
        check_ret("acl.media.dvpp_free", ret)

    def end_one_frame(self):
        ret = acl.media.venc_destroy_channel(self.venc_channel_desc)
        check_ret("acl.media.venc_destroy_channel", ret)
        self.callback_run_flag = False
        ret = acl.util.stop_thread(self.cb_thread_id)
        print("[INFO] stop_thread", ret)
        stream_format = 0
        timestamp = 123456
        ret_code = 1
        eos = 1

        # stream desc
        dvpp_stream_desc = acl.media.dvpp_create_stream_desc()
        check_none("acl.media.dvpp_create_stream_desc", dvpp_stream_desc)

        # stream_desc set function
        acl.media.dvpp_set_stream_desc_format(dvpp_stream_desc, stream_format)
        acl.media.dvpp_set_stream_desc_timestamp(dvpp_stream_desc, timestamp)
        acl.media.dvpp_set_stream_desc_ret_code(dvpp_stream_desc, ret_code)
        acl.media.dvpp_set_stream_desc_eos(dvpp_stream_desc, eos)

        ret = acl.media.dvpp_destroy_stream_desc(dvpp_stream_desc)
        check_ret("acl.media.dvpp_destroy_stream_desc", ret)


if __name__ == "__main__":
    import os
    import time
    import code
    import readline
    import rlcompleter

    vars = globals()
    vars.update(locals())
    readline.set_completer(rlcompleter.Completer(vars).complete)
    readline.parse_and_bind("tab: complete")

    device_id = 0
    ret = acl.init("")
    check_ret("acl.init", ret)
    ret = acl.rt.set_device(device_id)
    check_ret("acl.rt.set_device", ret)
    context, ret = acl.rt.create_context(device_id)
    check_ret("acl.rt.create_context", ret)
    stream, ret = acl.rt.create_stream()
    check_ret("acl.rt.create_stream", ret)
    run_mode, ret = acl.rt.get_run_mode()
    # check_ret("acl.rt.get_run_mode", ret)
    venc_procc = Venc(context)
    w, h = 1920, 1080
    venc_procc.venc_init()
    venc_procc.init_for_one_runs(w, h)
    dir_to = "vdec_out"
    for i in range(0, 20):
        name = f"vdec_test_{i}.yuv"
        file = os.path.join(dir_to, name)
        a = os.path.isfile(file)
        if a == True:
            file_context = np.fromfile(file, dtype=np.byte)
            file_size = file_context.size
            file_mem = acl.util.numpy_to_ptr(file_context)
            input_size = file_size
            input_mem, ret = acl.media.dvpp_malloc(input_size)
            check_ret("acl.media.dvpp_malloc", ret)
            ret = acl.rt.memcpy(
                input_mem, input_size, file_mem, file_size, ACL_MEMCPY_HOST_TO_DEVICE
            )
            check_ret("acl.rt.memcpy", ret)
            venc_procc.run_one_frame([input_mem, input_size])
            # time.sleep(1)
        else:
            continue

    venc_procc.end_one_frame()
    # code.InteractiveConsole(vars).interact()
