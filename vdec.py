import collections
import numpy as np
import acl

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
# from acl_util import check_ret

ACL_MEMCPY_HOST_TO_DEVICE = MemcpyType.ACL_MEMCPY_HOST_TO_DEVICE.value
ACL_MEMCPY_DEVICE_TO_HOST = MemcpyType.ACL_MEMCPY_DEVICE_TO_HOST.value


class Vdec:
    def __init__(
        self, context, width, height, enc_type=H264_HIGH_LEVEL, stream=None
    ) -> None:
        self.context = context
        # self.stream = stream
        self.vdec_channel_desc = None
        self.input_width = width
        self.input_height = height
        # self.dtype = dtype # only need if your read from file
        self._vdec_exit = True
        self._en_type = enc_type
        self._format = PIXEL_FORMAT_YUV_SEMIPLANAR_420
        self._channel_id = 10  # random number
        self.output_count = 1
        self.images_buffer = collections.deque(maxlen=30)
        self.rest_len = 5

    def get_try(self):
        try:
            p = self.images_buffer.popleft()
            return p
        except:
            return None

    def get_image_buffer(self):
        return self.images_buffer

    def get_last_buffer(self):
        try:
            return self.images_buffer.popleft()
        except:
            return None

    def __del__(self):
        # ret = acl.media.dvpp_free(self.input_stream_mem)
        # check_ret("acl.media.dvpp_free", ret)
        ret = acl.media.vdec_destroy_channel_desc(self.vdec_channel_desc)
        check_ret("acl.media.vdec_destroy_channel_desc", ret)
        ret = acl.media.vdec_destroy_frame_config(self.frame_config)
        check_ret("acl.media.vdec_destroy_frame_config", ret)

    def _thread_func(self, args_list):
        timeout = args_list[0]
        # context, ret = acl.rt.get_context()
        # check_ret("acl.rt.get_context()", ret)
        # print(
        #     "[THREAD_FUNC] Context     :{}\n"
        #     "[THREAD_FUNC] Self Context:{}\n".format(context, self.context)
        # )
        # print(f"THREAD_FUNC {self.context}")
        acl.rt.set_context(self.context)

        while self._vdec_exit:
            acl.rt.process_report(timeout)
        print("[Vdec] [_thread_func] _thread_func out")

    def _callback(self, input_stream_desc, output_pic_desc, user_data):
        # input_stream_desc
        if input_stream_desc:
            inp_data_buff = acl.media.dvpp_get_pic_desc_data(input_stream_desc)
            inp_data_size = acl.media.dvpp_get_pic_desc_size(input_stream_desc)
            ret = acl.media.dvpp_free(inp_data_buff)
            if ret !=0:
                print("acl.media.dvpp_free(inp_data_buff) failed")
            ret = acl.media.dvpp_destroy_stream_desc(input_stream_desc)
            if ret != 0:
                print("acl.media.dvpp_destroy_stream_desc failed")
        # output_pic_desc
        if output_pic_desc:
            vdec_out_buffer = acl.media.dvpp_get_pic_desc_data(output_pic_desc)
            acl.media.dvpp_get_pic_desc_ret_code(output_pic_desc)
            data_size = acl.media.dvpp_get_pic_desc_size(output_pic_desc)
            self.images_buffer.append(
                dict({"buffer": vdec_out_buffer, "size": data_size})
            )
            ret = acl.media.dvpp_destroy_pic_desc(output_pic_desc)
            if ret != 0:
                print("acl.media.dvpp_destroy_pic_desc failed")
        self.output_count += 1
        # print("[Vdec] [_callback] _callback exit success")

    def init_resource(self, cb_thread_id):
        self.vdec_channel_desc = acl.media.vdec_create_channel_desc()
        acl.media.vdec_set_channel_desc_channel_id(
            self.vdec_channel_desc, self._channel_id
        )
        acl.media.vdec_set_channel_desc_thread_id(self.vdec_channel_desc, cb_thread_id)
        acl.media.vdec_set_channel_desc_callback(self.vdec_channel_desc, self._callback)
        acl.media.vdec_set_channel_desc_entype(self.vdec_channel_desc, self._en_type)
        acl.media.vdec_set_channel_desc_out_pic_format(
            self.vdec_channel_desc, self._format
        )
        acl.media.vdec_create_channel(self.vdec_channel_desc)

    def _gen_input_dataset(self, img_path):
        img = np.fromfile(img_path, dtype=self.dtype)
        img_buffer_size = img.size
        img_ptr = acl.util.numpy_to_ptr(img)
        img_device, ret = acl.media.dvpp_malloc(img_buffer_size)
        ret = acl.rt.memcpy(
            img_device,
            img_buffer_size,
            img_ptr,
            img_buffer_size,
            ACL_MEMCPY_HOST_TO_DEVICE,
        )
        check_ret("acl.rt.memcpy", ret)
        return img_device, img_buffer_size

    def _set_input(self, input_stream_size):
        self.dvpp_stream_desc = acl.media.dvpp_create_stream_desc()
        ret = acl.media.dvpp_set_stream_desc_data(
            self.dvpp_stream_desc, self.input_stream_mem
        )
        check_ret("acl.media.dvpp_set_stream_desc_data", ret)
        ret = acl.media.dvpp_set_stream_desc_size(
            self.dvpp_stream_desc, input_stream_size
        )
        check_ret("acl.media.dvpp_set_stream_desc_size", ret)

    def _set_pic_output(self, output_pic_size):
        # pic_desc
        output_pic_mem, ret = acl.media.dvpp_malloc(output_pic_size)
        check_ret("acl.media.dvpp_malloc", ret)

        self.dvpp_pic_desc = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(self.dvpp_pic_desc, output_pic_mem)

        acl.media.dvpp_set_pic_desc_size(self.dvpp_pic_desc, output_pic_size)

        acl.media.dvpp_set_pic_desc_format(self.dvpp_pic_desc, self._format)

    def forward_stock(self, output_pic_size, input_stream_size):
        self.frame_config = acl.media.vdec_create_frame_config()

        for i in range(self.rest_len):
            print("[Vdec] forward index:{}".format(i))
            self._set_input(input_stream_size)
            self._set_pic_output(output_pic_size)

            # vdec_send_frame
            ret = acl.media.vdec_send_frame(
                self.vdec_channel_desc,
                self.dvpp_stream_desc,
                self.dvpp_pic_desc,
                self.frame_config,
                None,
            )
            check_ret("acl.media.vdec_send_frame", ret)

    def run_stock(self, video_dict):
        (
            self.video_path,
            self.input_width,
            self.input_height,
            self.dtype,
        ) = video_dict.values()
        # 此处设置触发回调处理之前的等待时间，
        timeout = 100
        cb_thread_id, ret = acl.util.start_thread(self._thread_func, [timeout])

        self.init_resource(cb_thread_id)

        output_pic_size = (self.input_width * self.input_height * 3) // 2
        self.input_stream_mem, input_stream_size = self._gen_input_dataset(
            self.video_path
        )
        self.forward_stock(output_pic_size, input_stream_size)

        ret = acl.media.vdec_destroy_channel(self.vdec_channel_desc)
        check_ret("acl.media.vdec_destroy_channel", ret)

        self._vdec_exit = False
        ret = acl.util.stop_thread(cb_thread_id)
        check_ret("acl.util.stop_thread", ret)
        print("[Vdec] vdec finish!!!\n")

    def _gen_input_dataset_gst_test(self, img_chunks):
        size = img_chunks.get_size()
        # print("[VDEC] [DEBUG] Chunk size :{}".format(size))
        img = np.ndarray(
            (size,), buffer=img_chunks.extract_dup(0, size), dtype=np.uint8
        )
        img_ptr = acl.util.numpy_to_ptr(img)
        img_device, ret = acl.media.dvpp_malloc(size)
        ret = acl.rt.memcpy(
            img_device,
            size,
            img_ptr,
            size,
            ACL_MEMCPY_HOST_TO_DEVICE,
        )
        check_ret("acl.rt.memcpy", ret)
        return img_device, size

    def forward_gst_test(self, output_pic_size, input_stream_size):
        self.frame_config = acl.media.vdec_create_frame_config()
        self._set_input(input_stream_size)
        self._set_pic_output(output_pic_size)
        ret = acl.media.vdec_send_frame(
            self.vdec_channel_desc,
            self.dvpp_stream_desc,
            self.dvpp_pic_desc,
            self.frame_config,
            None,
        )
        check_ret("acl.media.vdec_send_frame", ret)
        # pass

    def run_gst_test(self, video_dict):
        (
            self.video_array,
            self.input_width,
            self.input_height,
            self.dtype,
        ) = video_dict.values()
        timeout = 100
        cb_thread_id, ret = acl.util.start_thread(self._thread_func, [timeout])

        self.init_resource(cb_thread_id)
        output_pic_size = (self.input_width * self.input_height * 3) // 2
        # self.input_stream_mem, input_stream_size = self._gen_input_dataset_gst_test(
        #     self.video_path
        # )
        for idx, image_chunk in enumerate(self.video_array):
            print("[DEBUG] id of chunk :{}".format(idx))
            self.input_stream_mem, input_stream_size = self._gen_input_dataset_gst_test(
                image_chunk
            )
            self.forward_gst_test(output_pic_size, input_stream_size)

        ret = acl.media.vdec_destroy_channel(self.vdec_channel_desc)
        check_ret("acl.media.vdec_destroy_channel", ret)

        self._vdec_exit = False
        ret = acl.util.stop_thread(cb_thread_id)
        check_ret("acl.util.stop_thread", ret)
        print("[Vdec] vdec finish!!!\n")

    def run_one_image(self, img):
        output_pic_size = (self.input_width * self.input_height * 3) // 2
        self.input_stream_mem, input_stream_size = self._gen_input_dataset_gst_test(img)
        self._set_input(input_stream_size)
        self._set_pic_output(output_pic_size)
        ret = acl.media.vdec_send_frame(
            self.vdec_channel_desc,
            self.dvpp_stream_desc,
            self.dvpp_pic_desc,
            self.frame_config,
            None,
        )
        check_ret("acl.media.vdec_send_frame", ret)

    def init_for_one_runs(self):
        self.frame_config = acl.media.vdec_create_frame_config()
        timeout = 100
        self.cb_thread_id, ret = acl.util.start_thread(self._thread_func, [timeout])
        self.init_resource(self.cb_thread_id)

    def stop_for_one_runs(self):
        ret = acl.media.vdec_destroy_channel(self.vdec_channel_desc)
        check_ret("acl.media.vdec_destroy_channel", ret)

        self._vdec_exit = False
        ret = acl.util.stop_thread(self.cb_thread_id)
        check_ret("acl.util.stop_thread", ret)
        print("[Vdec] vdec finish!!!\n")


if __name__ == "__main__":
    import argparse
    import code
    import readline
    import rlcompleter
    import sys
    import cv2
    import os

    vars = globals()
    vars.update(locals())

    readline.set_completer(rlcompleter.Completer(vars).complete)
    readline.parse_and_bind("tab: complete")
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model_path", type=str, default="face-boxes.om")
    parser.add_argument("-v", "--video", type=str, default="out2.yuv")
    parser.add_argument("-enc", "--enc_type", type=int, default=3)  # 264 High

    # parser.add_argument("-inf", "--infer_on", type=bool, default=False)
    args = parser.parse_args()
    base_path = "/app/test_dir/test/video"
    path = os.path.join(base_path, args.video)
    device_id = 0
    ret = acl.init()
    check_ret("acl.init", ret)
    ret = acl.rt.set_device(device_id)
    check_ret("acl.rt.set_device", ret)
    context, ret = acl.rt.create_context(device_id)
    check_ret("acl.rt.create_context", ret)
    stream, ret = acl.rt.create_stream()
    check_ret("acl.rt.create_stream", ret)

    # Normal conf
    # vdec = Vdec(context, stream)
    vdec = Vdec(context, 1920, 1080)
    # video_dict = {"video_path": path, "width": 1280, "height": 720, "dtype": np.uint8}
    # vdec.run_stock(video_dict)

    # Gst conf
    from gst_raw_test import GstEater
    import time

    path = "rtsp://171.25.232.232:554/57a748fba81a45e0a9fba696c3264d07"
    gst_proc = GstEater(path, mode=2)
    gst_proc._runGstPipeline()
    wait_time = time.time() + 0.3 * 60
    while wait_time > time.time():
        pass
    img_arr_buff = gst_proc.get_buffer()
    gst_proc._stopGstPipeline()
    img_arr_buff = img_arr_buff[:250]
    video_dict = {
        "video_path": img_arr_buff,
        "width": 1920,
        "height": 1080,
        "dtype": np.uint8,
    }
    vdec.init_for_one_runs()
    out_cnt = 0
    for idx, image in enumerate(img_arr_buff):
        img_dict = {
            "video_path": image,
            "width": 1920,
            "height": 1080,
            "dtype": np.uint8,
        }
        _, w, h, dtype = img_dict.values()
        # vdec.run_one_image(img_dict)
        vdec.run_one_image(image)

        img_desc = vdec.get_image_buffer()
        if len(img_desc) > 1:
            img_desc = img_desc[-1]
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
            dir_to = "vdec_out"
            # file_name = f"vdec_test_{idx}.yuv"
            # np_output.tofile(os.path.join(dir_to, file_name))
            yuv = np_output.reshape((int(h * 1.5), w))
            yuv = yuv.astype("uint8")
            bgr_to_save = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
            bgr_name = f"vdec_yuv2bgr_{idx}.jpg"
            print("Write img to {}".format(bgr_name))
            cv2.imwrite(os.path.join(dir_to, bgr_name), bgr_to_save)
        else:
            # print("img array empty: {}".format(len(img_desc)))
            out_cnt += 1
    print("Img array was empty :{}".format(str(out_cnt)))
    vdec.stop_for_one_runs()
    ret = acl.finalize()
    sys.exit(0)
    check_ret("acl.finalize()", ret)
    # vdec.run_gst_test(video_dict)
    img_arr = vdec.get_image_buffer()
    # code.InteractiveConsole(vars).interact()
    _, w, h, dtype = img_dict.values()
    for idx, item in enumerate(img_arr):

        np_output = np.zeros(item["size"], dtype=np.byte)
        np_output_ptr = acl.util.numpy_to_ptr(np_output)
        ret = acl.rt.memcpy(
            np_output_ptr,
            item["size"],
            item["buffer"],
            item["size"],
            ACL_MEMCPY_DEVICE_TO_HOST,
        )
        check_ret("acl.rt.memcpy", ret)
        dir_to = "vdec_out"
        file_name = f"vdec_test_{idx}.yuv"
        # np_output.tofile(os.path.join(dir_to, file_name))
        # yuv = np_output.reshape((int(w * 1.5), h))
        yuv = np_output.reshape((int(h * 1.5), w))
        yuv = yuv.astype("uint8")
        # code.InteractiveConsole(vars).interact()
        bgr_to_save = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        bgr_name = f"vdec_yuv2bgr_{idx}.jpg"
        print("Write img to {}".format(bgr_name))
        cv2.imwrite(os.path.join(dir_to, bgr_name), bgr_to_save)

    ret = acl.finalize()
    check_ret("acl.finalize()", ret)
