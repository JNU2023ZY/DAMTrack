import sys
import os
import random
import numpy as np
import vot
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
import vot_utils
from dam4sam_tracker import DAM4SAMTracker
import sys
sys.path.append('/data_F/jinglin/vots2025/DAM4SAM-master')

def make_full_size(x, output_sz):
    pad_x = output_sz[0] - x.shape[1]
    pad_y = output_sz[1] - x.shape[0]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        pad_x = 0
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)

def get_vot_mask(masks_list, image_width, image_height):
    masks_multi = np.zeros((image_height, image_width), dtype=np.float32)
    for id_, mask in enumerate(masks_list, 1):
        m = make_full_size(mask, (image_width, image_height))
        masks_multi[m > 0] = id_
    return masks_multi

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

@torch.inference_mode()
@torch.cuda.amp.autocast()
def main():
    with open("./dam4sam_config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    seed_everything(config.get("seed", 42))

    handle = vot_utils.VOT("mask", multiobject=True)
    objects = handle.objects()
    imagefile = handle.frame()

    if not imagefile:
        sys.exit(0)

    image = Image.open(imagefile).convert('RGB')
    W, H = image.size

    # === 创建每个目标对应的 tracker 实例 ===
    trackers = []
    for i, obj_mask in enumerate(objects):
        mask_full = make_full_size(obj_mask, (W, H))
        trk = DAM4SAMTracker(tracker_name="sam21pp-L")
        trk.initialize(image, [mask_full])  # 单目标初始化
        trackers.append(trk)

    # === 开始逐帧处理 ===
    while True:
        imagefile = handle.frame()
        if not imagefile:
            break

        image = Image.open(imagefile).convert('RGB')
        pred_masks = []

        # 每个 tracker 独立进行 track
        for trk in trackers:
            out = trk.track(image)
            m = out['pred_mask']  # 单目标版本返回 pred_mask
            full_m = make_full_size(m, (W, H))  # pad 到完整尺寸
            pred_masks.append(full_m.astype(np.uint8))

        handle.report(pred_masks)

if __name__ == "__main__":
    main()
