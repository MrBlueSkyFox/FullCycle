import acl  # type: ignore
import numpy as np
from constant import ACL_ERROR_NONE, check_ret, MemcpyType, MallocType
import cv2
import pose_estimator2d

buffer_method = {
    "in": acl.mdl.get_input_size_by_index,
    "out": acl.mdl.get_output_size_by_index,
}
ACL_MEM_MALLOC_HUGE_FIRST = MallocType.ACL_MEM_MALLOC_HUGE_FIRST.value
ACL_MEMCPY_DEVICE_TO_HOST = MemcpyType.ACL_MEMCPY_DEVICE_TO_HOST.value
ACL_MEMCPY_HOST_TO_DEVICE = MemcpyType.ACL_MEMCPY_HOST_TO_DEVICE.value


def to_host(output_data):
    # output_data list with dict buffer(ptr to mem) and size(weight of buff)
    # return list with dict  buffer(ptr to mem) and size(weight of buff) on host mach
    dataset = []
    for i, item in enumerate(output_data):
        temp, ret = acl.rt.malloc_host(item["size"])
        if ret != 0:
            raise Exception("can't malloc_host ret={}".format(ret))
        dataset.append({"size": item["size"], "buffer": temp})
        ptr = temp
        ret = acl.rt.memcpy(
            ptr, item["size"], item["buffer"], item["size"], ACL_MEMCPY_DEVICE_TO_HOST
        )
        check_ret("acl.rt.memcpy", ret)

    return dataset


def to_device(img_data, input_data):
    # input_data numpy collapse image (n,)
    dataset = []
    ptr = acl.util.numpy_to_ptr(img_data)
    if len(input_data) == 1:
        input_data = input_data[0]
        ret = acl.rt.memcpy(
            input_data["buffer"],
            input_data["size"],
            ptr,
            input_data["size"],
            ACL_MEMCPY_HOST_TO_DEVICE,
        )
        check_ret("acl.rt.memcpy", ret)
    return [input_data]


def buffer_to_numpy(data):
    # mem,size = data["buffer"],data["size"]
    np_arr_list = []
    for temp in data:
        mem, size = temp["buffer"], temp["size"]
        np_arr = acl.util.ptr_to_numpy(mem, (size,), 1)
        np_arr_list.append(np_arr)
    return np_arr_list


class Model:
    def __init__(self, model_path, dtype="float16") -> None:
        self.model_path = model_path
        self.input_data = []
        self.output_data = []
        self.load_input_dataset = None
        self.load_output_dataset = None

        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)
        print("model_id:{}".format(self.model_id))
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        self.input_size = acl.mdl.get_num_inputs(self.model_desc)
        self.output_size = acl.mdl.get_num_outputs(self.model_desc)
        self.dtype_out = [dtype for i in range(0, self.output_size)]
        self.input_data_dims = []
        self.output_data_dims = []
        for i in range(self.input_size):
            buffer_size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            dims, ret = acl.mdl.get_input_dims(self.model_desc, i)
            check_ret("acl.mdl.get_input_dims", ret)
            self.input_data_dims.append(dims)
        for i in range(self.output_size):
            buffer_size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            dims, ret = acl.mdl.get_output_dims(self.model_desc, i)
            self.output_data_dims.append(dims)
        self.upsample_ratio = 4
        self.stride = 8
        self.confidence_min = 5
        self.num_keypoints = 18
        print("end_init")

    def get_inp_outp(self):
        data = {"inp_dims": self.input_data_dims, "out_dim": self.output_data_dims}
        return data

    def init_RGB_input(self):
        self._gen_data_buffer(self.input_size, des="in")
        self._gen_data_buffer(self.output_size, des="out")
        self._gen_dataset("out")

    def init_YUV_input(self):
        self._gen_data_buffer(self.output_size, des="out")
        self._gen_dataset("out")

    def _gen_data_buffer(self, size, des):
        func = buffer_method[des]
        for i in range(size):
            # check temp_buffer dtype
            temp_buffer_size = func(self.model_desc, i)
            temp_buffer, ret = acl.rt.malloc(
                temp_buffer_size, ACL_MEM_MALLOC_HUGE_FIRST
            )
            check_ret("acl.rt.malloc", ret)

            if des == "in":
                self.input_data.append(
                    {"buffer": temp_buffer, "size": temp_buffer_size}
                )
                print(f"Input data size {temp_buffer_size}")
            elif des == "out":
                self.output_data.append(
                    {"buffer": temp_buffer, "size": temp_buffer_size}
                )
                print(f"Output data size {temp_buffer_size}")

    def _gen_dataset(self, type_str="input"):
        dataset = acl.mdl.create_dataset()
        temp_dataset = None
        if type_str == "in":
            self.load_input_dataset = dataset
            temp_dataset = self.input_data
        else:
            self.load_output_dataset = dataset
            temp_dataset = self.output_data

        for item in temp_dataset:
            data = acl.create_data_buffer(item["buffer"], item["size"])
            _, ret = acl.mdl.add_dataset_buffer(dataset, data)

            if ret != ACL_ERROR_NONE:
                ret = acl.destroy_data_buffer(data)
                check_ret("acl.destroy_data_buffer", ret)

    def _destroy_databuffer(self):
        for dataset in [self.load_input_dataset]:
            if not dataset:
                continue
            number = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(number):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)
                if data_buf:
                    ret = acl.destroy_data_buffer(data_buf)
                    check_ret("acl.destroy_data_buffer", ret)
            ret = acl.mdl.destroy_dataset(dataset)
            check_ret("acl.mdl.destroy_dataset", ret)

    def _destyoy_input_data(self):
        for i, item in enumerate(self.input_data):
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

    def run_YUV(self, buffer):
        self.input_data = buffer
        self._gen_dataset("in")
        ret = acl.mdl.execute(
            self.model_id, self.load_input_dataset, self.load_output_dataset
        )
        check_ret("acl.mdl.execute", ret)
        self._destroy_databuffer()
        res = to_host(self.output_data)
        res = buffer_to_numpy(res)
        out = []
        for i, temp in enumerate(res):
            a = np.frombuffer(temp.tobytes(), self.dtype_out[i])
            out.append(a)
        return out

    def run_RGB(self, img_numpy):
        self.input_data = to_device(img_numpy, self.input_data)
        self._gen_dataset("in")
        ret = acl.mdl.execute(
            self.model_id, self.load_input_dataset, self.load_output_dataset
        )
        check_ret("acl.mdl.execute", ret)
        self._destroy_databuffer()
        self._destyoy_input_data()
        res = to_host(self.output_data)
        res = buffer_to_numpy(res)
        out = []
        for i, temp in enumerate(res):
            a = np.frombuffer(temp.tobytes(), self.dtype_out[i])
            out.append(a)
        return out

    def extract_poses(self, heatmap, pafs, scale_w, scale_h):
        heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0))
        pafs = np.transpose(np.squeeze(pafs), (1, 2, 0))
        heatmap = cv2.resize(
            heatmap,
            (0, 0),
            fx=self.upsample_ratio,
            fy=self.upsample_ratio,
            interpolation=cv2.INTER_CUBIC,
        )
        pafs = cv2.resize(
            pafs,
            (0, 0),
            fx=self.upsample_ratio,
            fy=self.upsample_ratio,
            interpolation=cv2.INTER_CUBIC,
        )
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(self.num_keypoints):  # 19th for bg
            total_keypoints_num += pose_estimator2d.extract_keypoints(
                heatmap[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num
            )

        pose_entries, all_keypoints = pose_estimator2d.group_keypoints(
            all_keypoints_by_type, pafs
        )

        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (
                all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio / scale_w
            )
            all_keypoints[kpt_id, 1] = (
                all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio / scale_h
            )
        print(f"pose_entries len = {len(pose_entries)}")
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                print("No entr")
                continue
            pose_keypoints = np.ones((self.num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(self.num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 0]
                    )
                    pose_keypoints[kpt_id, 1] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 1]
                    )
            if pose_entries[n][18] < self.confidence_min:
                print("No confidence")
                print(f"pose_entries conf {pose_entries[n][18]}")
                continue
            # vars = globals()
            # vars.update(locals())
            # readline.set_completer(rlcompleter.Completer(vars).complete)
            # readline.parse_and_bind("tab: complete")
            # code.InteractiveConsole(vars).interact()
            print(f"pose_keypoints  {pose_keypoints.shape}")
            print(f"pose_entries conf {pose_entries[n][18]}")
            pose = pose_estimator2d.Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)
        # vars = globals()
        # vars.update(locals())

        # readline.set_completer(rlcompleter.Completer(vars).complete)
        # readline.parse_and_bind("tab: complete")

        # code.InteractiveConsole(vars).interact()
        return current_poses

    def draw(self, img, poses):
        orig_img = img.copy()

        for pose in poses:
            pose.draw(img)

        return cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
