#!/usr/bin/env python
"""CLI demo for the ai-posture-monitor package.

Runs MediaPipe pose estimation + the package's rule-based static-posture
classifier on images, prints the per-image stand/sit/lie verdicts as a table,
and saves annotated copies with the pose landmarks drawn on.

Requirements:
    - Python 3.9 - 3.12 (mediapipe, the pose-estimation dependency, does not
      ship wheels for 3.13+ in the version range this package supports)
    - pip install ai-posture-monitor
    - pip install kagglehub   (only needed for the no-arguments sample mode)

Setup (recommended - use a virtual environment):
    # Linux / macOS / WSL:
    python3 -m venv .venv            # use python3.12/python3.11/python3.10
    source .venv/bin/activate        # if the default python3 is 3.13+
    pip install ai-posture-monitor kagglehub

    # Windows (PowerShell or cmd):
    py -3.12 -m venv .venv
    .venv\\Scripts\\activate
    pip install ai-posture-monitor kagglehub

    # When you're done:
    deactivate

Usage (commands below assume your shell is inside the repository folder -
`cd AI-Posture-Monitor` first, or use the full path to this script):
    # 1. On your own images (JPEG/PNG, one person, reasonably well lit):
    python examples/cli_demo.py --images path/to/img1.jpg path/to/img2.jpg

    # 2. No arguments: fetches 3 sample images (stand/sit/lie) from the
    #    published Kaggle dataset. The dataset is public, so no Kaggle
    #    account or API token is needed; downloads are cached under
    #    ~/.cache/kagglehub, so repeat runs are instant.
    python examples/cli_demo.py

    # 3. Choose where annotated images are written (default: ./output):
    python examples/cli_demo.py --output-dir my_results

Output:
    - A table on stdout: one row per image with the classifier's left/right-leg
      standing, sitting, and lying verdicts.
    - annotated_<image>.jpg in the output directory: the input image with
      MediaPipe's 33 pose landmarks and skeleton connections drawn on.
    - Images where MediaPipe finds no pose landmarks are reported as such and
      skipped. (Expected for some lying-down poses - self-occlusion from a
      fixed camera angle is a known MediaPipe limitation, and is exactly why
      the full system also uses bounding-box-based features as a fallback.)

Supported environments: local CLI / Jupyter and Kaggle notebooks.
(Google Colab is not officially supported: its runtime preloads libraries
that conflict with mediapipe's pinned dependencies.)

Dataset: https://www.kaggle.com/datasets/patrickogbuitepu/posture-monitor-and-fall-detection
Source:  https://github.com/pat2echo/AI-Posture-Monitor
"""
import argparse
import os
import sys

try:
    import matplotlib
    matplotlib.use('Agg')  # headless-safe; we only save figures, never show them
    import matplotlib.pyplot as plt
    import ai_posture_monitor as pm
except ImportError as e:
    sys.exit(f'Missing dependency: {e.name}\n\n'
             'Install the package first (this pulls in all dependencies):\n'
             '    pip install ai-posture-monitor\n'
             'and, for the no-arguments sample mode:\n'
             '    pip install kagglehub\n\n'
             'Note: requires Python 3.9 - 3.12 (mediapipe has no 3.13+ wheels\n'
             'in the version range this package supports).')
except SyntaxError:
    # ai-posture-monitor <=0.0.16 had a PEP 701 f-string that only parses on
    # Python 3.12+; fixed in 0.0.17
    sys.exit('Your installed ai-posture-monitor is outdated and fails to parse\n'
             'on this Python version. Upgrade it:\n'
             '    pip install -U ai-posture-monitor')

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
