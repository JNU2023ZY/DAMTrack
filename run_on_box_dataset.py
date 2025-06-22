import os
import yaml
import hydra
import torch
import random
import argparse
import numpy as np

from utils.mask_utils import mask2box, save_boxes
from utils.dataset_utils import get_dataset, pil2array
from utils.visualization_utils import VisualizerTracking

from dam4sam_tracker import DAM4SAMTracker

with open("./dam4sam_config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

seed = config["seed"]
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

@torch.inference_mode()
@torch.cuda.amp.autocast()
def main(tracker_name, dataset_name, output_dir, selected_sequence=None):
    dataset = get_dataset(dataset_name, init_masks='sam2')

    for i in range(dataset.size()):
        tracker = DAM4SAMTracker(tracker_name)
        if output_dir is None:
            visualizer = VisualizerTracking()

        sequence_name = dataset.get_seq_name(i)
        if selected_sequence is not None and selected_sequence != sequence_name:
            continue

        print('Processing sequence:', sequence_name)

        if output_dir is not None:
            output_path = os.path.join(output_dir, '%s.txt' % sequence_name)
            # suppose that we want to save results to a file
            if os.path.exists(output_path):
                print('This sequense has already been processed. Skipping...')
                continue

        seq_len = dataset.get_seq_len(sequence_name)
        predictions = []
        for frame_idx in range(seq_len):
            img = dataset.get_pil_frame(sequence_name, frame_idx)

            if frame_idx == 0:
                init_mask = dataset.get_init_mask(sequence_name)
                pred_bbox = mask2box(init_mask)
                _ = tracker.initialize(img, init_mask)
            else:
                outputs = tracker.track(img)
                pred_mask = outputs['pred_mask']
                pred_bbox = mask2box(pred_mask)

            if pred_bbox is None:
                predictions.append([-1, -1, -1, -1])
            else:
                predictions.append(pred_bbox)

            if output_dir is None:
                visualizer.show(pil2array(img), box=pred_bbox)
        
        if output_dir is not None:
            # save bboxes to disk in a txt file
            save_boxes(output_path, predictions)
            print('Results saved to:', output_path)
        
    hydra.core.global_hydra.GlobalHydra.instance().clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=None, help='got | lasot | lasot_ext ')
    parser.add_argument('--sam', type=str, default=None, help='SAM2 version (2 or 21).')
    parser.add_argument('--size', type=str, default=None, help='Size of the model (T, S, B, L).')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory.')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence name.')

    args = parser.parse_args()

    dataset_name = args.dataset_name
    if not (None in [args.sam, args.size]):
        tracker_name = f'sam{args.sam}pp-{args.size}'
    else:
        tracker_name = 'sam21pp-L'

    if args.output_dir is not None:
        base_output_dir = os.path.join(args.output_dir, tracker_name)
        run_idx = 0
        output_dir = os.path.join(base_output_dir, dataset_name, '%03d' % run_idx)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = None

    main(tracker_name, dataset_name, output_dir=output_dir, selected_sequence=args.sequence)