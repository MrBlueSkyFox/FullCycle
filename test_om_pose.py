from ast import arg
import traceback
import os
import numpy as np
import acl  # type: ignore
import collections
from constant import (
    ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEMCPY_HOST_TO_DEVICE,
    check_ret,
    EncdingType,
    ImageType,
    MallocType,
    MemcpyType,
)
import cv2
from vdec import Vdec
from venc import Venc
from gst_raw_test import GstRtspReciver
from gst_h264_test import GstConvertVideo
from model_acl import Model
import argparse
import code
import readline
import rlcompleter

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "-i",
    "--inp",
    default="/app/test_dir/test/img/GirlSkelet.jpg",
    help="",
)
parser.add_argument(
    "-m",
    "--model",
    default="",
    help="",
)
args = parser.parse_args()
import numpy as np

np.random.seed(0)

input_dict = {"0": np.zeros, "1": np.ones, "rand": np.random.random}


def get_mock(func_method, dsize=(600, 600, 3)):
    np.random.seed(0)
    val = func_method(dsize)
    return val


def prep_img(img, h_w):
    img = img.copy()
    img_height, img_width, _ = img.shape
    h_model, w_model = h_w
    scale_h, scale_w = h_model / img_height, w_model / img_width
    img = img.astype(np.float32)
    img = img * (2.0 / 255.0) - 1.0

    resized = cv2.resize(img, (0, 0),
                         fx=scale_w,
                         fy=scale_h,
                         interpolation=cv2.INTER_LINEAR)
    resized = resized.astype(np.float32)
    print(
        f"resized np.float32 shape {(np.frombuffer(resized.tobytes(),np.float32)).shape}"
    )
    print(
        f"resized np.float64 shape { (np.frombuffer(resized.tobytes(),np.float64)).shape }"
    )
    print(
        f"resized np.uint8 shape { (np.frombuffer(resized.tobytes(),np.uint8)).shape }"
    )
    print(
        f"resized np.int8 shape { (np.frombuffer(resized.tobytes(),np.int8)).shape }"
    )
    input_data = np.expand_dims(resized, axis=0)
    input_data = input_data.transpose((0, 3, 1, 2))
    print(
        f"img np.float32 shape {(np.frombuffer(input_data.tobytes(),np.float32)).shape}"
    )
    print(
        f"img np.float64 shape { (np.frombuffer(input_data.tobytes(),np.float64)).shape }"
    )
    print(
        f"img np.uint8 shape { (np.frombuffer(input_data.tobytes(),np.uint8)).shape }"
    )
    print(f"img np.int8 shape {(np.frombuffer(input_data.tobytes(),np.int8)).shape}")
    img_numpy = np.frombuffer(input_data.tobytes(), np.float32)
    # img_numpy = np.frombuffer(input_data.tobytes(), np.int8)
    return img_numpy, scale_h, scale_w


def test_om_pose(context, args):
    mock_zero = get_mock(input_dict["0"])
    mock_ones = get_mock(input_dict["1"])
    mock_rand = get_mock(input_dict["rand"])

    # mock = mock_zero
    # mock = mock_ones
    mock = mock_rand
    print(f"Mock [0][0] {mock[0][0]}")
    model = Model(args.model, "float32")
    model.init_RGB_input()
    model_dims = model.get_inp_outp()
    h_model, w_model = model_dims["inp_dims"][0]["dims"][2:]
    img_to_buffer = prep_img(mock, (h_model, w_model))
    out = model.run_RGB(img_to_buffer)
    # out_ar = out[0]
    res = out[0].reshape(1, 57, 32, 57)
    heatmaps = res[:, :19]
    pafs = res[:, 19:]

    # heatmaps2 = np.squeeze(heatmaps)
    # print(f"heatmaps[0][0]")
    # print(heatmaps2[0][0])
    # print(f"heatmaps[10][10]")
    # print(heatmaps2[10][10])


def inference_rgb_img(context, args):
    names = [
        "/app/test_dir/test/img/PoseInferMean.jpg",
        "/app/test_dir/test/img/PoseInferNoMean.jpg",
    ]
    model = Model(args.model, "float32")
    print("BOOOOOOOOOOOOOOOOOo")
    model.init_RGB_input()
    print("bbb")
    model_dims = model.get_inp_outp()
    h_model, w_model = model_dims["inp_dims"][0]["dims"][2:]
    # print("Mode dimiseon")
    # print(model_dims)
    # print("\n")
    print("b")
    image = cv2.imread(args.inp)
    img_to_buffer, scale_h, scale_w = prep_img(image, (h_model, w_model))
    vars = globals()
    vars.update(locals())
    readline.set_completer(rlcompleter.Completer(vars).complete)
    readline.parse_and_bind("tab: complete")
    code.InteractiveConsole(vars).interact()

    out = model.run_RGB(img_to_buffer)
    # print("2")
    res = out[0].reshape(1, 57, 32, 57)
    heatmaps = res[:, :19]
    pafs = res[:, 19:]
    poses = model.extract_poses(heatmaps, pafs, scale_w, scale_h)
    img_poses = model.draw(image, poses)
    # print(f"Img [100][100] {b[100][100]}")
    cv2.imwrite(names[0],img_poses)
    return heatmaps[0], pafs[0], image
    # cv2.imwrite(names[0],img_poses)
    # return out


if __name__ == "__main__":
    device_id = 0
    # l = "/app/test_dir/test/acl_test/pipe_debug"
    # log_place = "/app/test_dir/test/video/in_out_264"
    # os.environ["GST_DEBUG_FILE"] = log_place

    # os.environ["GST_DEBUG_DUMP_DOT_DIR"] = l
    # b = "/app/test_dir/test/acl_test/FullCycle/acl.json"
    b = ""
    ret = acl.init(b)
    check_ret("acl.init", ret)
    ret = acl.rt.set_device(device_id)
    check_ret("acl.rt.set_device", ret)
    context, ret = acl.rt.get_context()
    check_ret("acl.rt.create_stream", ret)
    # test_om_pose(context, args)
    heatmaps_acl, pafs_acl, resized = inference_rgb_img(context, args)
    # print("heatmaps_acl[0][0]")
    # print(heatmaps_acl[0][0])
    # print("\n")
    # print("heatmaps_acl[-1][-1]")
    # print(heatmaps_acl[-1][-1])
    # print("\n")
    # print("pafs_acl[0][0]")
    # print(pafs_acl[0][0])
    # print("\n")
    # print("pafs_acl[-1][-1]")
    # print(pafs_acl[-1][-1])
    # print(len(out))
