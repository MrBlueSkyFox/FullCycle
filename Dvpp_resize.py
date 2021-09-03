import acl
import numpy as np
from constant import check_ret, PIXEL_FORMAT_YUV_SEMIPLANAR_420


class Dvpp:
    def __init__(self, stream, model_w, model_h) -> None:
        self.model_w = model_w
        self.model_h = model_h
        self.stream = stream
        self.pixel_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420
        self._dvpp_channel_desc = None
        self._pic_desc_dic = {}  # Dict for picture description pointers
        self._dev_size_dic = {}  # Dict with size for picture
        self._dev_dic = {}  # Dict with buffers pointers
        self._alig_hw = {}
        self._roi_dic = {}
        self.init_resource()

    def __del__(self):
        if self._dvpp_channel_desc:
            ret = acl.media.dvpp_destroy_channel(self._dvpp_channel_desc)
            check_ret("acl.media.dvpp_destroy_channel", ret)
            ret = acl.media.dvpp_destroy_channel_desc(self._dvpp_channel_desc)
            check_ret("acl.media.dvpp_destroy_channel_desc", ret)

    def init_resource(self):
        self._dvpp_channel_desc = acl.media.dvpp_create_channel_desc()
        ret = acl.media.dvpp_create_channel(self._dvpp_channel_desc)
        check_ret("acl.media.dvpp_create_channel", ret)

    def gen_vpc_pic_desc(
        self, w_src, h_src, w_align, h_align, pic_desc=None, input_buffer=None
    ):
        align_width = ((w_src + w_align - 1) // w_align) * w_align
        align_height = ((h_src + h_align - 1) // h_align) * h_align
        self._alig_hw = {
            "align_width": align_width,
            "align_height": align_height
        }
        buf_size = (align_width * align_height * 3) // 2
        if pic_desc is None:
            pic_desc = acl.media.dvpp_create_pic_desc()
            # self._pic_desc_dic[opt] = pic_desc
        if input_buffer is None:
            # for vpc
            dev, ret = acl.media.dvpp_malloc(buf_size)
            check_ret("acl.media.dvpp_malloc", ret)
        else:
            dev = input_buffer
        acl.media.dvpp_set_pic_desc_data(pic_desc, dev)
        acl.media.dvpp_set_pic_desc_size(pic_desc, buf_size)

        acl.media.dvpp_set_pic_desc_format(pic_desc, self.pixel_format)
        acl.media.dvpp_set_pic_desc_width(pic_desc, w_src)
        acl.media.dvpp_set_pic_desc_height(pic_desc, h_src)
        acl.media.dvpp_set_pic_desc_width_stride(pic_desc, align_width)
        acl.media.dvpp_set_pic_desc_height_stride(pic_desc, align_height)
        return {"buffer": dev, "size": buf_size, "pic_desc": pic_desc}

    def run_one_frame_resize(self, width_src, height_src, buffer):
        w_align, h_align = 16, 2
        input_desc = self.gen_vpc_pic_desc(
            width_src, height_src, w_align, h_align, input_buffer=buffer
        )
        output_desc = self.gen_vpc_pic_desc(
            self.model_w, self.model_h, w_align, h_align
        )
        self._resize_config = acl.media.dvpp_create_resize_config()
        ret = acl.media.dvpp_vpc_resize_async(
            self._dvpp_channel_desc,
            input_desc["pic_desc"],
            output_desc["pic_desc"],
            self._resize_config,
            self.stream,
        )
        check_ret("acl.media.dvpp_vpc_resize_async", ret)
        ret = acl.rt.synchronize_stream(self.stream)
        check_ret("acl.rt.synchronize_stream", ret)
        # print(f"[DVPP] input_desc: {input_desc}\n")
        # print(f"[DVPP] output_desc:{output_desc}\n")
        # print(f"[DVPP] self.model_w:{self.model_w}\n")
        # print(f"[DVPP] self.model_h:{self.model_h}\n")

        self.clean(input_desc, output_desc)
        output_desc.pop("pic_desc")
        return output_desc

    def process_vpc_crop_one_frame(self, buffer, width_src, height_src,
                                   left_offs, right_offs, top_offs, bot_offs):
        # print("[Dvpp] vpc crop process start:")
        w_align, h_align = 16, 2
        # create input picture description
        input_desc = self.gen_vpc_pic_desc(width_src,
                              height_src,
                              w_align,
                              h_align,
                              input_buffer=buffer)
        if left_offs % 2 != 0:
            left_offs += 1
        if right_offs % 2 == 0:
            right_offs += 1
        if top_offs % 2 != 0:
            top_offs += 1
        if bot_offs % 2 == 0:
            bot_offs += 1
        width_dst = right_offs - left_offs + 1
        height_dst = bot_offs - top_offs + 1
        if height_dst % h_align != 0:
            h_desire = ((height_dst // h_align) + 1) * h_align
            h_dif = (h_desire - height_dst)
            if h_dif % 2 == 0 and (top_offs - h_dif) >= 0:
                tmp = top_offs
                top_offs -= h_dif
            elif h_dif % 2 != 0 and (bot_offs + h_dif) <= height_src:
                tmp = bot_offs
                bot_offs += h_dif
            # print("h_dif:{} h_desire:{} height_dst:{}\n"
            #       "tmp:{} top_offs:{} bot_offs:{} cur_width:{}".format(
            #           h_dif, h_desire, height_dst, tmp, top_offs, bot_offs,
            #           (bot_offs - top_offs + 1)))
            height_dst = bot_offs - top_offs + 1
        if width_dst % w_align != 0:
            w_desire = ((width_dst // w_align) + 1) * w_align
            w_dif = (w_desire - width_dst)
            if w_dif % 2 == 0 and (left_offs - w_dif) >= 0:
                tmp = left_offs
                left_offs -= w_dif
            elif w_dif % 2 != 0 and (right_offs + w_dif) <= width_src:
                tmp = right_offs
                right_offs += w_dif
            # print("w_dif:{} w_desire:{} width_dst:{}\n"
            #       "tmp:{} left_offs:{} right_offs:{} cur_width:{}".format(
            #           w_dif, w_desire, width_dst, tmp, left_offs, right_offs,
            #           (right_offs - left_offs + 1)))
            width_dst = right_offs - left_offs + 1
        
        # create output picture description
        output_desc = self.gen_vpc_pic_desc(width_dst, height_dst, w_align,
                              h_align)
        self._roi_dic["crop"] = acl.media.dvpp_create_roi_config(
            left_offs, right_offs, top_offs, bot_offs)
        
        ret = acl.media.dvpp_vpc_crop_async(
            self._dvpp_channel_desc,
            input_desc["pic_desc"],
            output_desc["pic_desc"],
            self._roi_dic["crop"],
            self.stream,
        )
        check_ret("acl.media.dvpp_vpc_crop_async", ret)
        ret = acl.rt.synchronize_stream(self.stream)
        check_ret("acl.rt.synchronize_stream", ret)
        return output_desc, self._alig_hw
        # np_output = np.zeros(self._dev_size_dic["out_crop"], dtype=np.byte)
        # np_output_ptr = acl.util.numpy_to_ptr(np_output)
        # ret = acl.rt.memcpy(
        #     np_output_ptr,
        #     self._dev_size_dic["out_crop"],
        #     self._dev_dic["out_crop"],
        #     self._dev_size_dic["out_crop"],
        #     ACL_MEMCPY_DEVICE_TO_HOST,
        # )

        # check_ret("acl.rt.memcpy", ret)

        # return np_output, self._dev_dic["out_crop"], self._dev_size_dic[
        #     "out_crop"], self._alig_hw["out_crop"]

    def clean(self, input_desc, output_desc):
        output_desc
        ret = acl.media.dvpp_destroy_pic_desc(output_desc["pic_desc"])
        check_ret("acl.media.dvpp_destroy_pic_desc", ret)
        ret = acl.media.dvpp_destroy_pic_desc(input_desc["pic_desc"])
        check_ret("acl.media.dvpp_destroy_pic_desc", ret)
        ret = acl.media.dvpp_destroy_resize_config(self._resize_config)
        check_ret("acl.media.dvpp_destroy_resize_config", ret)
