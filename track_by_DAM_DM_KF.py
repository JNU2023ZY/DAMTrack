from numpy.lib.type_check import imag
import gc
import torch
import torch.nn.functional as F
import os
import sys
import cv2
import importlib
import numpy as np
import math
import random
import collections
Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])

DIR_PATH = os.path.join(os.path.dirname(__file__), '.')
sys.path.append(DIR_PATH)
from DMAOT.tools.transfer_predicted_mask2vottype import transfer_mask
AOT_PATH = os.path.join(os.path.dirname(__file__), './DMAOT/dmaot')
sys.path.append(AOT_PATH)
import DMAOT.dmaot.dataloaders.video_transforms as tr
from torchvision import transforms
from DMAOT.dmaot.networks.engines import build_engine
from DMAOT.dmaot.utils.checkpoint import load_network
from DMAOT.dmaot.networks.models import build_vos_model
from dam4sam_tracker import DAM4SAMTracker
import os
import random
from PIL import Image
import torch
import torch.nn.functional as F
import vot_utils
sys.path.append('/data_F/zhouyong/VOTS2025/DAM4SAM-master/')
sys.path.append('/data_F/zhouyong/VOTS2025/DAM4SAM-master/sam2')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import cv2
import numpy as np
from collections import defaultdict

DMAOT_config = {
    'exp_name': 'default',
    'model': 'swinb_dm_deaotl',
    'pretrain_model_path': 'DMAOT/dmaot/pretrain_models/SwinB_DeAOTL_PRE_YTB_DAV_VIP_MOSE_OVIS_LASOT_GOT.pth',
    'config': 'pre_ytb_dav',
    'long_max': 10,
    'long_gap': 30,
    'short_gap': 2,
    'patch_wised_drop_memories': False,
    'patch_max': 999999,
    'gpu_id': 0,
}

# set DMAOTcfg
engine_config = importlib.import_module('configs.' + f'{DMAOT_config["config"]}')
cfg = engine_config.EngineConfig(DMAOT_config['exp_name'], DMAOT_config['model'])
cfg.TEST_CKPT_PATH = os.path.join(DIR_PATH, DMAOT_config['pretrain_model_path'])
cfg.TEST_LONG_TERM_MEM_MAX = DMAOT_config['long_max']
cfg.TEST_LONG_TERM_MEM_GAP = DMAOT_config['long_gap']
cfg.TEST_SHORT_TERM_MEM_GAP = DMAOT_config['short_gap']
cfg.PATCH_TEST_LONG_TERM_MEM_MAX = DMAOT_config['patch_max']
cfg.PATCH_WISED_DROP_MEMORIES = True if DMAOT_config['patch_wised_drop_memories'] else False

def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)

def read_img(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_torch(1000000007)
torch.set_num_threads(4)
torch.autograd.set_grad_enabled(False)


class AOTTracker(object):
    def __init__(self, cfg, gpu_id):
        self.with_crop = False
        self.EXPAND_SCALE = None
        self.small_ratio = 12
        self.mid_ratio = 100
        self.large_ratio = 0.5
        self.AOT_INPUT_SIZE = (465, 465)
        self.gpu_id = gpu_id
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(gpu_id)
        print('cfg.TEST_CKPT_PATH = ', cfg.TEST_CKPT_PATH)
        self.model, _ = load_network(self.model, cfg.TEST_CKPT_PATH, gpu_id)
        self.engine = build_engine(cfg.MODEL_ENGINE,
                                   phase='eval',
                                   aot_model=self.model,
                                   gpu_id=gpu_id,
                                   short_term_mem_skip=cfg.TEST_SHORT_TERM_MEM_GAP,
                                   long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP)

        self.transform = transforms.Compose([
            # tr.MultiRestrictSize_(cfg.TEST_MAX_SHORT_EDGE,
            #                         cfg.TEST_MAX_LONG_EDGE, cfg.TEST_FLIP, cfg.TEST_INPLACE_FLIP,
            #                         cfg.TEST_MULTISCALE, cfg.MODEL_ALIGN_CORNERS),
            tr.MultiRestrictSize(cfg.TEST_MAX_SHORT_EDGE,
                                 cfg.TEST_MAX_LONG_EDGE, cfg.TEST_FLIP,
                                 cfg.TEST_MULTISCALE, cfg.MODEL_ALIGN_CORNERS),
            tr.MultiToTensor()
        ])
        self.model.eval()

    # add the first frame and label
    def add_first_frame(self, frame, mask, object_num):
        sample = {
            'current_img': frame,
            'current_label': mask,
        }
        sample['meta'] = {
            'obj_num': object_num,
            'height': frame.shape[0],
            'width': frame.shape[1],
        }
        sample = self.transform(sample)

        frame = sample[0]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
        mask = sample[0]['current_label'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)

        mask = F.interpolate(mask, size=frame.size()[2:], mode="nearest")

        # add reference frame
        self.engine.add_reference_frame(frame, mask, frame_step=0, obj_nums=object_num)

    def track(self, image):
        height = image.shape[0]
        width = image.shape[1]

        sample = {'current_img': image}
        sample['meta'] = {
            'height': height,
            'width': width,
        }
        sample = self.transform(sample)
        output_height = sample[0]['meta']['height']
        output_width = sample[0]['meta']['width']
        image = sample[0]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)

        self.engine.match_propogate_one_frame(image)
        pred_logit = self.engine.decode_current_logits((output_height, output_width))
        pred_prob = torch.softmax(pred_logit, dim=1)
        pred_label = torch.argmax(pred_prob, dim=1,
                                  keepdim=True).float()

        _pred_label = F.interpolate(pred_label,
                                    size=self.engine.input_size_2d,
                                    mode="nearest")
        self.engine.update_memory(_pred_label)

        mask = pred_label.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
        return mask

class MultiDAM4SAMTracker(object):
    def __init__(self, tracker_name="sam21pp-L"):
        self.tracker_name = tracker_name
        self.trackers = {}
        self.obj_counter = 0
        self.original_size = None  # 存储原始图像尺寸

    def _create_single_tracker(self):
        """ 创建单目标跟踪器实例 """
        return DAM4SAMTracker(tracker_name=self.tracker_name)

    def initialize(self, image, objects):
        """
        初始化多目标跟踪器
        Args:
            image: numpy.ndarray (H,W,3) BGR格式
            objects: list[numpy.ndarray] 初始mask列表
        """
        self.original_size = image.shape[:2]  # 记录原始尺寸(H,W)
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # image_pil.save('/data_F/zhouyong/DAM4SAM-master/vis_ini_mask/vis_ini_image.png')
        self.trackers = {}
        for obj_id, obj_mask in enumerate(objects):
            # 确保初始mask尺寸正确
            obj_mask = make_full_size(obj_mask, (self.original_size[1], self.original_size[0]))

            tracker = self._create_single_tracker()
            tracker.initialize(image_pil, obj_mask)
            self.trackers[obj_id] = tracker

        self.obj_counter = len(objects)

    def track(self, image, img_id):
        """ 执行多目标跟踪 """
        current_size = image.shape[:2]

        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pred_masks = []

        for obj_id, tracker in self.trackers.items():

            try:
                result = tracker.track(image_pil)
                mask = make_full_size(result['pred_mask'], (current_size[1], current_size[0]))

                pred_masks.append(mask)


            except Exception as e:
                print(f"Object {obj_id} tracking failed: {str(e)}")
                pred_masks.append(np.zeros(current_size, dtype=np.uint8))
        return pred_masks


class KalmanBoxTracker:
    """卡尔曼滤波器实现"""

    def __init__(self, bbox):
        self.kf = cv2.KalmanFilter(8, 4)  # 8状态量，4观测量
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)

        self.kf.measurementMatrix = np.zeros((4, 8), np.float32)
        np.fill_diagonal(self.kf.measurementMatrix[:4, :4], 1)

        self.kf.statePost = self._convert_bbox_to_z(bbox)

    def predict(self):
        pred = self.kf.predict()
        return self._convert_z_to_bbox(pred[:4])

    def update(self, bbox):
        z = self._convert_bbox_to_measurement(bbox)
        self.kf.correct(z)

    def _convert_bbox_to_measurement(self, bbox):
        """只返回观测量部分 [cx, cy, w, h] 的列向量"""
        return np.array([
            [bbox[0] + bbox[2] / 2],
            [bbox[1] + bbox[3] / 2],
            [bbox[2]],
            [bbox[3]]
        ], dtype=np.float32)
    def _convert_bbox_to_z(self, bbox):
        """转换bbox到状态向量 [cx,cy,w,h]"""
        return np.array([
            bbox[0] + bbox[2] / 2,
            bbox[1] + bbox[3] / 2,
            bbox[2],
            bbox[3],
            0, 0, 0, 0  # 速度初始化为0
        ], dtype=np.float32)

    def _convert_z_to_bbox(self, z):
        """转换状态向量到bbox [x,y,w,h]"""
        return np.array([
            z[0] - z[2] / 2,
            z[1] - z[3] / 2,
            z[2],
            z[3]
        ])


class UnifiedTracker:
    def __init__(self):
        self.kalman_filters = {}  # {obj_id: KalmanBoxTracker}
        self.track_states = defaultdict(dict)

    def _mask_to_box(self, mask):
        """mask转bbox [x,y,w,h]"""
        pos = np.where(mask)
        if len(pos[0]) == 0:
            return np.array([0, 0, 0, 0])
        return np.array([
            np.min(pos[1]), np.min(pos[0]),
            np.max(pos[1]) - np.min(pos[1]),
            np.max(pos[0]) - np.min(pos[0])
        ])

    def _box_iou(self, box1, box2):
        """计算两个bbox的IOU"""
        # 计算交集
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[0] + box1[2], box2[0] + box2[2])
        y_bottom = min(box1[1] + box1[3], box2[1] + box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        return intersection / (area1 + area2 - intersection)


# ==================== 主流程 ====================

# 初始化
handle = vot_utils.VOT("mask", multiobject=True)
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

# 读取图像（保持原始方式）
DMAOT_image = read_img(imagefile)  # RGB
DAM_image = cv2.imread(imagefile)  # BGR
objects = handle.objects()



# 初始化跟踪器
DMtracker = AOTTracker(cfg, DMAOT_config["gpu_id"])
DAMtracker = MultiDAM4SAMTracker(tracker_name="sam21pp-L")

# 初始化DMAOT（需要合并mask）
merged_mask = np.zeros(DMAOT_image.shape[:2])
object_id = 1
for object in objects:
    mask = make_full_size(object, (DMAOT_image.shape[1], DMAOT_image.shape[0]))
    mask = np.where(mask > 0, object_id, 0)
    merged_mask += mask
    object_id += 1
DMtracker.add_first_frame(DMAOT_image, merged_mask, len(objects))

# 初始化DAM（单目标并行）
DAMtracker.initialize(DAM_image, objects)

# 初始化卡尔曼滤波器
unified_tracker = UnifiedTracker()
for obj_id in range(1, len(objects) + 1):
    init_mask = np.where(merged_mask == obj_id, 1, 0)
    init_box = unified_tracker._mask_to_box(init_mask)
    unified_tracker.kalman_filters[obj_id] = KalmanBoxTracker(init_box)

# frame = 0
# 跟踪循环
while True:
    imagefile = handle.frame()
    if not imagefile:
        break

    # 读取当前帧
    DAM_image = cv2.imread(imagefile)  # BGR
#    DMAOT_image = read_img(imagefile)  # RGB
    DMAOT_image = DAM_image[..., ::-1]

    # 获取各跟踪器结果
    dam_masks = DAMtracker.track(DAM_image, img_id=0)
    dmaot_result = DMtracker.track(DMAOT_image)
    dmaot_result = F.interpolate(
        torch.tensor(dmaot_result)[None, None, :, :],
        size=merged_mask.shape,
        mode="nearest"
    ).numpy().astype(np.uint8)[0][0]
    dmaot_masks = transfer_mask(dmaot_result, len(objects))
#    frame += 1
    # 结果融合
    final_masks = []
    for obj_id in range(1, len(objects) + 1):
        # 获取各方法结果
        dam_mask = dam_masks[obj_id - 1]
        dmaot_mask = np.where(dmaot_masks[obj_id - 1] > 0, 1, 0).astype(np.uint8)
        # 卡尔曼预测
        pred_box = unified_tracker.kalman_filters[obj_id].predict()

        # 计算IOU
        dam_box = unified_tracker._mask_to_box(dam_mask)
        dmaot_box = unified_tracker._mask_to_box(dmaot_mask)

        dam_iou = unified_tracker._box_iou(dam_box, pred_box)
        dmaot_iou = unified_tracker._box_iou(dmaot_box, pred_box)

        # 决策逻辑
#        if abs(dam_iou - dmaot_iou) > 0.5:
        if (dmaot_iou - dam_iou) > 0.4:  # 差异过大时信任DMAOT
            chosen_mask = dmaot_mask
            update_box = dmaot_box

#            print(f"Obj {obj_id} corrected: DAM iou={dam_iou:.2f}, DMAOT iou={dmaot_iou:.2f}")
        else:  # 默认信任DAM
            chosen_mask = dam_mask
            update_box = dam_box

        # 更新卡尔曼滤波器
        unified_tracker.kalman_filters[obj_id].update(update_box)
        final_masks.append(chosen_mask)


    handle.report(final_masks)
    
DMtracker.reset()
DAMtracker.reset()
unified_tracker.kalman_filters.clear()
del DMtracker, DAMtracker, unified_tracker, merged_mask, objects
gc.collect()
torch.cuda.empty_cache()
