#!/usr/bin/env python
"""CLI demo for the ai-posture-monitor package.

Runs MediaPipe pose estimation + the package's rule-based static-posture
classifier on images, prints the verdicts, and saves annotated copies with
the pose landmarks drawn on.

Usage:
    # On your own images:
    python examples/cli_demo.py --images path/to/img1.jpg path/to/img2.jpg

    # Or let it fetch 3 sample images (stand/sit/lie) from the published
    # Kaggle dataset (requires: pip install kagglehub):
    python examples/cli_demo.py

Outputs are written to ./output (override with --output-dir).

Supported environments: local CLI / Jupyter and Kaggle notebooks.
(Google Colab is not officially supported: its runtime preloads libraries
that conflict with mediapipe's pinned dependencies.)
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')  # headless-safe; we only save figures, never show them
import matplotlib.pyplot as plt

import ai_posture_monitor as pm

DATASET = 'patrickogbuitepu/posture-monitor-and-fall-detection'
SAMPLE_IMAGES = ['stand.jpg', 'sit.jpg', 'lie.jpg']


def fetch_sample_images():
    try:
        import kagglehub
    except ImportError:
        sys.exit('No --images given and kagglehub is not installed.\n'
                 'Either pass your own images or: pip install kagglehub')
    paths = []
    for name in SAMPLE_IMAGES:
        print(f'Fetching sample from Kaggle dataset: train/pose/{name}')
        paths.append(kagglehub.dataset_download(DATASET, path=f'train/pose/{name}'))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', nargs='+', default=None,
                        help='image files to classify (default: fetch 3 samples from the Kaggle dataset)')
    parser.add_argument('--output-dir', default='output',
                        help='where to write annotated images (default: ./output)')
    args = parser.parse_args()

    image_paths = args.images if args.images else fetch_sample_images()
    os.makedirs(args.output_dir, exist_ok=True)

    pose, mp_drawing, mp_pose = pm.initialize_mediapipe()
    feature_cols = pm.get_attr_of_features()

    print()
    print(f'{"image":<28} {"stand":<24} {"sit":<24} {"lie":<22}')
    print('-' * 98)

    for img_path in image_paths:
        name = os.path.basename(img_path)
        results, img_rgb, landmarks_df = pm.detect_pose_landmarks(img_path, pose=pose, show=True)

        if not results.pose_landmarks:
            print(f'{name:<28} no pose landmarks detected')
            continue

        landmarks_3d = landmarks_df[['X', 'Y', 'Z']].to_numpy()
        feats = dict(zip(feature_cols, pm.get_features(landmarks_3d, image_name=name)))
        print(f'{name:<28} '
              f'{feats["stand_left"] + "/" + feats["stand_right"]:<24} '
              f'{feats["sit_left"] + "/" + feats["sit_right"]:<24} '
              f'{feats["lie_left"] + "/" + feats["lie_right"]:<22}')

        annotated = img_rgb.copy()
        mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.imshow(annotated)
        ax.set_title(name)
        ax.axis('off')
        out_path = os.path.join(args.output_dir, f'annotated_{name}')
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

    print()
    print(f'Annotated images saved to {os.path.abspath(args.output_dir)}')


if __name__ == '__main__':
    main()
