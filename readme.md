# College CV

This repository contains the implementation of a system to detect yield signs from images and videos.
It is a proof of concept realization based on a course of machine vision at RheinMain University of Applied
Sciences in 2017. The scope of the task is to implement some operations on images including but not
restricted to color segmentation, morphological operations, hough transform for lines and highlighting
found yield signs.

## Installation and Execution

The least stressful way to set this up is probably to create a virtualenv with python 3.5+ and then
run `pip install -r requirements`. You find two executables: `main.py` for
the graphical user interface where you can play around with parameters for the different algorithms
and `video.py`, a script to process video data.

```
 > ./main.py --help
usage: main.py [-h] [--fname FNAME]

optional arguments:
  -h, --help     show this help message and exit
  --fname FNAME  open a file directly
```

```
 > ./video.py --help
usage: video.py [-h] [--config CONFIG] [--binary] [--edges] [--save-all]
                f_in f_out

positional arguments:
  f_in             input file
  f_out            output file

optional arguments:
  -h, --help       show this help message and exit
  --config CONFIG  configuration file
  --binary         only apply segmentation and morphology
  --edges          only apply --binary and edge detection
  --save-all       save not only the result but all intermediate steps
```

![GUI screenshot](https://github.com/dreadworks/college-cv/raw/docs/docs/src/gui_morph.png)
![GUI screenshot](https://github.com/dreadworks/college-cv/raw/docs/docs/src/gui_hough.png)