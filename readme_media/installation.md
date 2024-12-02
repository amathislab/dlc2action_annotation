## Installation

Recommended: install git and [Anaconda](https://docs.anaconda.com/anaconda/install/) and run the following.

This was tested on Ubuntu 20.04 (AM), Windows 10 and MacOS 10.15.7 (LK). 

```bash
git clone https://github.com/amathislab/dlc2action_annotation
cd dlc2action_annotation
conda env create -f dlc2action_annotation.yaml
``` 

### Updating

To download the latest release, you should go to the ```dlc2action_annotation``` folder and run these commands.
```bash
git pull
conda activate base
conda env update -f dlc2action_annotation.yaml
``` 

### Troubleshooting

- In case you get the following error you likely need to reinstall the `libxcb` package.
```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
```

Back to main [readme](../README.md).
