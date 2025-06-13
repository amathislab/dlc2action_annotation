# cluster\_annotation

A PyQt5-based tool for visualizing, clustering, and interactively annotating short video segments using DLC features or synthetic embeddings.

## Features

* **Clustering & Visualization**: Dimensionality reduction (PCA, t-SNE, LLE, ICA, FA) applied to feature embeddings, with an interactive scatterplot.
* **Interactive Annotation**: For each selected clip, a video window shows the clip with start/end timestamps and a scrollable list of behaviors. Add, toggle, and label behaviors on-the-fly.
## Installation

1. **Clone repository**:

   ```bash
   git clone https://github.com/Mikehau/dlc2action_annotation.git
   cd dlc2action_annotation
   git fetch origin hbmae:hbmae
   git checkout hbmae
   ```
2. **Create & activate environment** (conda or venv), then install dependencies:

   ```bash
   conda env create -f dlc2action_gui.yaml
   conda activate dlc2action_gui
   ```

## Usage

Run the clustering and annotation tool:

```bash
python cluster_annotation.py \
  --video_folder dlc2action_project/video \
  --feature_folder dlc2action_project/feature \
  --annotation_folder dlc2action_project/Annotations \
  --feature_suffix .npy \
  --hbmae
```

### Command-line Options

* `--video_folder`     : Path to raw video folder (`.mp4`, `.avi`, etc.).
* `--feature_folder`   : Precomputed feature embeddings folder (`.npy` files).
* `--feature_suffix`   : Suffix for feature files (e.g. `.npy`).
* `--annotation_folder`: Folder to save/load `cluster_labels.pkl` and annotation pickles.
* `--sampling`         : Frame sampling rate (default: 1).
* `--clip_length`      : Number of frames per video clip window (default: 300).
* `--hbmae`            : Use HBMAE embeddings branch.

## Data Structure

Below is a typical layout of your project directory. Be careful, the file\_name should be the same for video, feature and annotations.

```
cluster_annotation/                   # repository root
├── dlc2action_project/               # example data directory
│   ├── video/file_name.mp4           # raw video files
│   ├── feature/file_name.npy         # feature `.npy` files
│   └── Annotations/file_name.pickle  # saved labels & pickles (`cluster_labels.pkl`)
└── 
```

## Interactive Annotation Workflow

1. **Cluster Plot**: Click on points to open video windows.
2. **Video Window**: Displays clip, start/end times, and behaviors list.
3. **Behaviors**: Check/uncheck to label; add new behaviors interactively.
## Example

```bash
python cluster_annotation.py --video_folder dlc2action_project/video \
  --feature_folder dlc2action_project/feature \
  --annotation_folder dlc2action_project/Annotations \
  --feature_suffix .npy --skip_dlc2action --hbmae \
  --sampling 1 --clip_length 200
```