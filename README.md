# DLC2Action: annotation

You can use this program to label a video frame-by-frame, without any fancy algorithms. It was written by [Liza Kozlova](https://github.com/elkoz) in the [Mathis group](https://www.mathislab.org/). Currently we are in developer mode, licensing etc. to follow. 

## Installation: 

[Installation and updating](readme_media/installation.md)

## User guide

[Main user guide](readme_media/userguide.md)

[Clustering interface](readme_media/cluster.md)


# User experience design project

You can experiment and improve in this branch the overall package design. 

### Datasets
You can use the data stored in the [drive](https://drive.google.com/drive/folders/1BVJ9W3VMJvw9DYi4lrJbMEH4veEzfMxa?usp=sharing), it should contains an example from the [Open Field Test of Sturman (2020)](https://www.nature.com/articles/s41386-020-0776-y) and some data from the MausHaus project (internal). The OFT data video, DLC pose and orignal behavior annotation converted to DLC2Action format of a freely moving mouse in a constrained environment. There is also a suggestion file representing predictions from a DLC2Action model. We also have access to more data examples if needed. The MausHaus project contains multi-view video and DLC poses of a single mouse freely moving in a cage along with the behavior annotations and the corresponding 3D poses.

### First steps
* Try the annotation tool
    - Maybe draw a map on how it works
    - Load data and play with the tool
    - Feel free to report every single bug
    - Feel free to update the documentation whenever you understand anything

* There is nothing better than fixing some bugs in order to get familiar with the code base. Here are 2 known design problems to improve:
    - Loading label choices everytime a setting is change
        - Get rid of the previous label choice feature
        - Labels may be automatically infered from the annotation file but the algorithm should be flexible for new unannotated labels.
    - Changing labels for nested behaviors
        - Having nested behavior is a feature that we want but it also comes with more complex ways to select the data
        - The user should be able to choose when selecting behavior whether it is nested or not
        - Behavior categories should work without subcategories
        - Collapse all pop-up windows into one "Change Behavior" window that should have the potential to add associate the shortcuts.

### GOAL
* After improving the user designs, record a short tutorial video on how to use the annotation tool !

### Future steps
* Test the active learning pipeline
* Test the clustering pipeline

* Link the annotation tool with DLC2Action 
