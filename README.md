[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](README.md)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

# DLC2Action: Graphical user interface for annotating behaviors

You can use this graphical user interface to manually label a video frame-by-frame. You can also use [DLC2action](https://github.com/amathislab/DLC2action) to semi-automatically label frames! 

## Installation

Install git and [Anaconda](https://docs.anaconda.com/anaconda/install/) and run the following commands in a terminal.
```bash
git clone https://github.com/amathislab/dlc2action_annotation
cd dlc2action_annotation
conda env create -f dlc2action_gui.yaml
conda activate dlc2action_gui
``` 

Further detailed in [Installation and updating](readme_media/installation.md).

## Quick Start
You can start using the interface by running the following commands in a terminal
```bash
conda activate dlc2action_gui
python annotator.py
```
The standard workflow is rather straightforward and involves
1) Loading videos
2) Setting labels and shortcuts
3) Annotating videos
4) Saving your work

You can find detail documentation on how to use the annotation tool in the
[Main user guide](readme_media/userguide.md)

## Acknowledgments and Credits

The GUI was initially written by [Liza Kozlova](https://github.com/elkoz) in the [Mathis group](https://www.mathislab.org/). The GUI is still in development but fully operational on the main branch. Please reach out, or open an issue if you have questions! Collaborations are welcome. 
