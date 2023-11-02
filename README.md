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


To add a list of chapters in a README file, you can use markdown syntax. Markdown is a lightweight markup language that allows you to format text, create lists, and add links in a simple and readable way. Here's how you can create a list of chapters using markdown:

## Table of Contents

- [Setting up your environment](#Setting-up-your-environment)
- [Creating a project](#Creating-a-project)
- [Loading videos](#Loading-videos)
- [Handling videos](#Handling-videos)
- [Annotating videos](#Annotating-videos)
- [Add or modify labels](#Add-or-modify-labelss)
- [Saving your work](#Saving-your-work)
- [Opening a project](#Opening-a-project)
- [Change settings](#Change-settings)

## 1) Setting up your environment :

You can start using the interface by running the following commands in a terminal
```bash
conda activate AnnotationGUI
python annotator.py
```
Annotation Workflow Tutorial

## 2) Creating a project :
Once the application is launched, locate the option to create a new project. 
- Provide a title for your project
- Set the annotator's name
- Set your labels for the project and/or select exiting labels
- Set the keybord shortcuts for your annotations to improve your workflow speed

## 3) Loading videos :
Once the project is created a window will open to prompt you to select your videos.
- You can select one or multiple videos
- If you select multiple videos you'll have the option to display them sequentially or conjointly
- Select 'Yes' for multiview to display all videos conjointly (depending on the size of your videos this will take a few minutes)

## 4) Handling videos :
Actions you can perform: 
- Play/stop (shortcut: space bar)
- Set video speed
- Select frames
- Move video frames using the hand icon
- Drag and zoom
- Clicking anywhere on the bar will take you to the corresponding frame
- If multiple animals are displayed and their keypoints have been uploaded, the animal that is being annotated is the one with the colored keypoint markers


## 5) Annotating videos :

Dive into detailed tutorials on the annotation process. Explore techniques for tagging and marking within videos.
- To annotate or handle your annotations you have to first select the action you want to perform then click on the annotation
- To create a new annotation, hit the + icon then drag the label below the video 
- Modify any action a by clicking and dragging the edges of your annotation in the **Move** mode.  
- Select the trash bin icon / press `Cmd + R` to delete an annotation
- Select the scissors icon / press `Cmd + C` to split an annotation in two
- Select the transparency icon / press `Cmd + B` to mark actions as ambiguous. In that case the actions will be transparent on the action bar
- Select the label icon / press `Cmd + A` to change the annotation's label to another in the label's menu.


## 6) Add, edit or delet labels :
- Use the keyboard shortcut cmd+L or go to "labels" then "Change labels"
- For nested annotation, you can choose a category by double-clicking it and go back to the categories list by pressing `Esc` or the 'Go back to categories' button. 

## 7) Saving your work :

The program should run smoothly, but please don't forget to save your results regularly by selecting the 'Save' action in the 'File' menu or pressing `Cmd + S`, just in case.

Backups for your annotated data are automatically created every **30 minutes** in a folder located next to your first video: `path/to/video1_backups`. You can change the directory
where backups are saved for your project, and the interval at which they are saved. 
Running
```bash
python annotator.py --backup-dir /path/to/backups --backup-interval 120
```
will save backups every 2 hours (120 minutes) in the `/path/to/backups` folder.

The results of your work will be saved at `path/to/video1_annotation.pickle`. The `_annotation.pickle` 
suffix is the default, you can change it in the settings window (at Files / Annotation suffix). If you 
open a video that already has a corresponding annotation file in the same folder, that file will be loaded 
automatically. A human-readable version of the annotations will also be saved in CSV format (with the same suffix, and `.csv` extension).

## 8) Opening a project :
Once the application is launched, locate the option to open a project. 
- Select your project folder then click open

## 9) Change settings :

You can find detail documentation on how to use the annotation tool in the
[Main user guide](readme_media/userguide.md)



