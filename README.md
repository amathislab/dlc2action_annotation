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

## Quick Start : Annotation Workflow Tutorial

Welcome to the Annotation Workflow Tutorial section, your comprehensive guide to mastering the annotation process. Below is a breakdown of the standard workflow, designed to ensure a seamless experience:

**1) Setting up your environment :**

You can start using the interface by running the following commands in a terminal
```bash
conda activate AnnotationGUI
python annotator.py
```
Annotation Workflow Tutorial

**2) Creating a project :**
Once the application is launched, locate the option to create a new project. 
- Provide a title for your project
- Set the annotator's name
- Set your labels for the project and/or select exiting labels
- Set the keybord shortcuts for your annotations to improve your workflow speed

**3) Loading videos :**
Once the project is created a window will open to prompt you to select your videos.
- You can select one or multiple videos
- If you select multiple videos you'll have the option to display them sequentially or conjointly
- Select 'Yes' for multiview to display all videos conjointly (depending on the size of your videos this will take a few minutes)

**4) Handling videos :**
Actions you can perform: 
- Play/stop (shortcut: space bar)
- Set video speed
- Select frames
- Move video frames

**5) Annotating videos :**

Dive into detailed tutorials on the annotation process. Explore techniques for tagging and marking within videos.

**6) Add or modify labels :**


**7) Saving your work :**

Explore methods to save your annotated data securely. Learn about saving options, including formats such as CSV or JSON. Ensure your progress is preserved for future reference and analysis.

**8) Opening a project :**
Once the application is launched, locate the option to open a project. 

**9) Change the setting of your project :**

You can find detail documentation on how to use the annotation tool in the
[Main user guide](readme_media/userguide.md)



