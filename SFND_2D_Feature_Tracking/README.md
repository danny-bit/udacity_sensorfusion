# SFND 2D Feature Tracking

<p align="center">
  <img src="https://github.com/danny-bit/udacity_sensorfusion/assets/59084863/84241a5b-7e0a-498c-a1d9-7793edb71ecc" alt="Sublime's custom image"/>
</p>
<p align="center">
  <img src="https://github.com/danny-bit/udacity_sensorfusion/assets/59084863/38f60253-25ac-4743-8f6a-1b37a68d170d" alt="Sublime's custom image"/>
</p>

This repository contains the submission for the mid-term project related to "Camera Based 2D Feature Tracking", which is part of Udacity's Sensor Fusion Nanodegree program.

To obtain the starter code, read about dependencies, please refer to the following repository: https://github.com/udacity/SFND_2D_Feature_Tracking.git

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Create VS Code Solution: `cmake .. -G "Visual Studio 14 2015 Win64" -DVCPKG_TARGET_TRIPLET=x64-windows`
4. Compile `cmake -build .`

## Run

To run the program a desired descriptor and detector must be provided:

1. Run it: `./2D_feature_tracking SIFT SIFT`.

In the build folder there are python scripts to run different combinations and for evaluation.
In the build/results folder there are the logged outputs of a run.

## Report and Results

The writeup that explains how, the rubric points were adressed can be found in the Report.pdf file.



