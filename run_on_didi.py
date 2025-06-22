import os
import time
import yaml
import torch
import hydra
import random
import argparse
import numpy as np
from PIL import Image

from vot.dataset import load_dataset
from vot.region.io import read_trajectory, write_trajectory
from vot.region.shapes import Mask

from dam4sam_tracker import DAM4SAMTracker
from utils.utils import get_seq_names
from utils.dataset_utils import pil2array
from utils.visualization_utils import VisualizerTracking

from pathlib import Path
config_path = Path(__file__).parent / "dam4sam_config.yaml"
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

seed = config["seed"]
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

@torch.inference_mode()
@torch.cuda.amp.autocast()
def run_sequence(tracker_name, dataset_path, sequence_names, output_dir=None):
    
    dataset = load_dataset(dataset_path)

    tracker = DAM4SAMTracker(tracker_name)
    for sequence_name in sequence_names:
        # check if the results for this sequence already exist
        if output_dir is not None:
            if os.path.exists(os.path.join(output_dir, sequence_name, f"{sequence_name}.txt")):
                continue
            
        if output_dir is None:
            visualizer = VisualizerTracking()

        sequence = dataset[sequence_name]
        frame_idxs = list(range(len(sequence)))
        
        pred_masks = []

        init_img = Image.open(sequence.frame(0).filename())
        img_width, img_height = init_img.width, init_img.height
        init_mask_path = os.path.join(sequence.metadata('root'), 'first_frame_segm.txt')
        mask = read_trajectory(init_mask_path)[0].rasterize((0, 0, img_width-1, img_height-1))
        outputs = tracker.initialize(init_img, mask)
        pred_masks.append(Mask(mask))
        
        for i, frame_idx in enumerate(frame_idxs[1:]):
            image = Image.open(sequence.frame(frame_idx).filename())

            outputs = tracker.track(image)

            pred_mask = outputs['pred_mask']
            pred_masks.append(Mask(pred_mask))

            if output_dir is None:
                visualizer.show(pil2array(image), mask=pred_mask)   

        if output_dir is not None:
            os.makedirs(os.path.join(output_dir, sequence.name), exist_ok=True)
            write_trajectory(os.path.join(output_dir, sequence.name, f"{sequence.name}.txt"), pred_masks)

    hydra.core.global_hydra.GlobalHydra.instance().clear()

def main():
    parser = argparse.ArgumentParser(description='Visualize sequence.')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence name.')
    parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--sam', type=str, default=None, help='SAM2 version (2 or 21).')
    parser.add_argument('--size', type=str, default=None, help='Size of the model (T, S, B, L).')

    args = parser.parse_args()

    if not (None in [args.sam, args.size]):
        tracker_name = f'sam{args.sam}pp-{args.size}'
    else:
        tracker_name = 'sam21pp-L'

    if args.sequence is None:
        seq_names = get_seq_names(args.dataset_path)
    else:
        seq_names = [args.sequence]

    run_sequence(tracker_name, args.dataset_path, seq_names, output_dir=args.output_dir)
    
if __name__ == "__main__":
    main()