import traceback
import os
import numpy as np
import acl  # type: ignore
import collections
import shutil
from constant import (
    ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEMCPY_HOST_TO_DEVICE,
    check_ret,
    EncdingType,
    ImageType,
    MallocType,
    MemcpyType,
)
from vdec import Vdec
from venc import Venc
from gst_raw_test import GstRtspReciver
from gst_h264_test import GstConvertVideo
from model_acl import Model
import argparse
import cv2

"""
Get encoded frame from gst
Decoded wtih Vdec
Encoded wth Venc
!! Go trough gst conversation
Подсуить в gst pipepline для конвератции чистого 
закодированного потока в видео с контейнером(mp4 или типо того)
"""
codec_description = {
    0: "H265_MAIN_LEVEL",
    1: "H264_BASELINE_LEVEL",
    2: "H264_MAIN_LEVEL",
    3: "H264_HIGH_LEVEL",
}
file_description = {"ts": 1, "mp4": 2}
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "-i",
    "--rtsp",
    default="rtsp://admin:Q!w2e3r4t5@192.168.0.100:554/cam/realmonitor?channel=1&subtype=0",
    help="",
)
parser.add_argument(
    "-f",
    "--force_fps",
    default="",
    help="Указать fps пока/файла,если ffmpeg неправильно распазнает fps",
    type=str,
)
# 1 -множество маленьких файлов по 2 секунды( ts.format) лучше подходит ,если на входе rtsp поток
# 2- один mp4 файл, лучше подходит ,если на входе видео
parser.add_argument(
    "--out_format", default="ts", help="Указать формат выхода видео(ts,mp4)", type=str
)
# Урезать ли выходные кадры в Половину
parser.add_argument("-cut", "--cut_out_fps", default=False, help="", type=bool)
# Включить передачу device->host->device
parser.add_argument(
    "-mem_tr", "--enable_memory_transfer", default=False, help="", type=bool
)


args = parser.parse_args()


def device2host_mem(buff_inp, stream):
    buffer, size = buff_inp["buffer"], buff_inp["size"]
    np_output = np.zeros(size, dtype=np.byte)
    ptr_frame = acl.util.numpy_to_ptr(np_output)
    ret = acl.rt.memcpy_async(
        ptr_frame,
        size,
        buffer,
        size,
        ACL_MEMCPY_DEVICE_TO_HOST,
        stream,
    )
    check_ret("acl.rt.memcpy_async", ret)
    ret = acl.rt.synchronize_stream(stream)
    check_ret("acl.rt.synchronize_stream", ret)
    if int(np_output[0]) == 0:
        print(np_output[0:10])
    return np_output


def host2device_mem(np_input, buff_inp, stream, inplace=False):
    buffer, size = buff_inp["buffer"], buff_inp["size"]
    if inplace:
        img_buff = buffer
    else:
        img_buff, ret = acl.media.dvpp_malloc(size)
        check_ret("acl.media.dvpp_malloc", ret)
    ptr_frame = acl.util.numpy_to_ptr(np_input)
    ret = acl.rt.memcpy_async(
        img_buff,
        size,
        ptr_frame,
        size,
        ACL_MEMCPY_HOST_TO_DEVICE,
        stream,
    )
    check_ret("acl.rt.memcpy_async", ret)
    ret = acl.rt.synchronize_stream(stream)
    check_ret("acl.rt.synchronize_stream", ret)
    if not inplace:
        ret = acl.media.dvpp_free(img_buff)
        check_ret("acl.media.dvpp_free", ret)
    # return {"buffer": img_buff, "size": size}


def normal_run(context, args):

    # rtsp2 = "rtsp://admin:admin123@192.168.0.108:554"
    # НЕ ПОДАВАТЬ ВИДЕО С РАЗРЕШЕНИЕМ ВЫШЕ FULL HD, не подкручен resize для encodera
    rtsp = args.rtsp

    out_video_format = args.out_format
    try:
        out_video_format = file_description[args.out_format]
    except:
        out_video_format = file_description["ts"]
    cut_fps = args.cut_out_fps
    enable_mem_transf = args.enable_memory_transfer

    vid1 = "face-demographics-walking.mp4"
    vid2 = "face-demographics-walking-and-pause.mp4"
    # feed rtsp file or file path()
    # init feeder and val to decoder
    gst_thread = GstRtspReciver(rtsp, args.force_fps)
    gst_thread._runGstPipeline()
    dict_data = {}
    dict_data["fps"] = gst_thread.fps
    dict_data["fps_out_test"] = gst_thread.fps
    if cut_fps:
        dict_data["fps_out_test"] = int(gst_thread.fps / 2)
    dict_data["w"] = gst_thread.FRAME_WIDTH
    dict_data["h"] = gst_thread.FRAME_HEIGHT
    dict_data["CODEC"] = gst_thread.CODEC_PROFILE
    # w, h = 768, 432
    w = dict_data["w"]
    h = dict_data["h"]
    codec = dict_data["CODEC"]
    print(
        f"Info about strem/file {gst_thread.video_path}\n"
        f"Width: {dict_data['w']} Height: {dict_data['h']}\n"
        f"Fps: {dict_data['fps']} Codec {codec_description[dict_data['CODEC']]}\n"
    )

    print(
        f"Args Info:\n"
        f"Input file {rtsp}\n"
        f"out_format {out_video_format} cut_fps_ON= {cut_fps} mem_tranf_ON= {enable_mem_transf}"
    )
    # Init ACL DECODER
    vdec_thread = Vdec(context, w, h, codec)
    vdec_thread.init_for_one_runs()

    # some dev stuff
    end_point = "/app/test_dir/test/video/in_out_264/test.h264"
    stream_dvpp_encode, ret = acl.rt.create_stream()
    check_ret("acl.rt.create_stream", ret)
    memory_stream, ret = acl.rt.create_stream()
    check_ret("acl.rt.create_stream", ret)
    memory_stream2, ret = acl.rt.create_stream()
    check_ret("acl.rt.create_stream", ret)

    # Init ACL ENCODER
    # context2 ,ret = acl.rt.create_context(0)
    # check_ret("acl.rt.create_stream", ret)
    # venc_thread = Venc(
    #     context2, w, h, EncdingType.H264_HIGH_LEVEL.value, dict_data["fps_out_test"]
    # )
    venc_thread = Venc(
        context, w, h, EncdingType.H264_HIGH_LEVEL.value, dict_data["fps_out_test"]
    )
    venc_thread.venc_init_single()

    dir_to_save = "/app/test_dir/test/video/sink_gst_acl"
    dir_to_save_local = os.path.join(os.getcwd(), "HSLINK_PLAY")
    if os.path.exists(dir_to_save_local):
        shutil.rmtree(dir_to_save_local)
        print(f"Clean the dir of hsl {dir_to_save_local}")

    os.makedirs(dir_to_save_local)
    conv = GstConvertVideo(
        out_place=dir_to_save_local, format_save=out_video_format, dict_data=dict_data
    )
    conv._runGstPipeline()
    print("init_over")
    i = 0
    count_img = 0
    try:
        while True:
            gst_buffer = gst_thread.get()
            if gst_buffer[0] != None:
                vdec_thread.run_one_image(gst_buffer[0])
            vdec_buffer = vdec_thread.get_try()

            if vdec_buffer != None:
                to_enc_buff = vdec_buffer
                if cut_fps and count_img % 2 == 0:
                    ret = acl.media.dvpp_free(to_enc_buff["buffer"])
                    check_ret("acl.media.dvpp_free", ret)
                    pass  # free
                else:
                    if enable_mem_transf:
                        data_in_host = device2host_mem(to_enc_buff, memory_stream)
                        host2device_mem(data_in_host, to_enc_buff, memory_stream2)
                    venc_thread.run_one_image(to_enc_buff)
                count_img += 1
            venc_img = venc_thread.get_try()
            if type(venc_img) == np.ndarray:
                conv.frame_work(venc_img)
    except:
        print(traceback.format_exc())
        print("EXcp cal , or smth")
        conv.end_work()
    # code.interact(local=locals())
    return 0


if __name__ == "__main__":
    device_id = 0
    l = "/app/test_dir/test/acl_test/pipe_debug"
    log_place = "/app/test_dir/test/video/in_out_264"
    # os.environ["GST_DEBUG_FILE"] = log_place

    os.environ["GST_DEBUG_DUMP_DOT_DIR"] = l
    ret = acl.init("")
    check_ret("acl.init", ret)
    ret = acl.rt.set_device(device_id)
    check_ret("acl.rt.set_device", ret)
    context, ret = acl.rt.get_context()
    check_ret("acl.rt.create_stream", ret)

    out_h264 = normal_run(context, args)
