# DLC2Action: annotation

You can use this program to label a video with a single-frame precision, without any fancy algorithms. The interface supports multi-view setups, 2D and 3D pose data from DeepLabCut. It was written by [Liza Kozlova](https://github.com/elkoz) in the [Mathis group](https://www.mathislab.org/). The algorithm is still in development but fully operational on the main branch, licensing etc. to follow. 

## Installation

Install git and [Anaconda](https://docs.anaconda.com/anaconda/install/) and run the following commands in a terminal.
```bash
git clone https://github.com/amathislab/dlc2action_annotation
cd dlc2action_annotation
conda env create -f AnnotationGUI.yaml
``` 

Further detailed in [Installation and updating](readme_media/installation.md).

## Quick Start
You can start using the interface by running the following commands in a terminal
```bash
conda activate AnnotationGUI
python annotator.py
```
The standard workflow is rather straightforward and involves
1) Loading videos
2) Setting labels and shortcuts
3) Annotating videos
4) Saving your work

You can find detail documentation on how to use the annotation tool in the
[Main user guide](readme_media/userguide.md)

