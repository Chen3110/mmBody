# mmBody Benchmark: 3D Body Reconstruction Dataset and Analysis for Millimeter Wave Radar

## Project page
https://chen3110.github.io/mmbody/index.html

## Data Explanation

### Mesh Results

The mesh results include parameters of SMPL-X and 3D coordinates of joints and vertices. For the "pose" parameters of each frame, the first 3 dimensions denote the root translation, and 4-6 represent the root rotation. The rest dimensions represent rotations of 21 body joints. The SMPL-X model can be downloaded from https://smpl-x.is.tue.mpg.de/index.html.
### Calibration

We provide calibration matrix for three sub-system. We set the mmWave radar as the target coordinate system. You can also transform it to the camera system by using the calibration matrix.

### Intrinsic of Cameras

INTRINSIC = {
    'master': ([
        [969.48345947265625,    0,                  1024.9678955078125],
        [0,                     968.99578857421875, 781.4013671875],
        [0,                     0,                  1]]),
    'sub': ([
        [972.07073974609375,    0,                  1021.4869384765625  ],
        [0,                     971.651123046875,   780.25439453125     ],
        [0,                     0,                  1                   ]
    ])
}

## BibTeX
@inproceedings{chen2022mmbody,\
title={mmBody Benchmark: 3D Body Reconstruction Dataset and Analysis for Millimeter Wave Radar},\
author={Chen, Anjun and Wang, Xiangyu and Zhu, Shaohao and Li, Yanxu and Chen, Jiming and Ye, Qi},\
booktitle={Proceedings of the 30th ACM International Conference on Multimedia},\
pages={3501--3510},\
year={2022}\
}
