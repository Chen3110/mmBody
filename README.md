# mmBody Benchmark: 3D Body Reconstruction Dataset and Analysis for Millimeter Wave Radar

## News
Our new work of fusion method ImmFusion for mmWave point clouds and RGB images has been released [here](https://github.com/Chen3110/ImmFusion).

## Project page
https://chen3110.github.io/mmbody/index.html

## Dataset Download
To download the dataset, please send us an e-mail (anjun_chen@outlook.com) including contact details (title, full name, organization, and country) and the purpose for downloading the dataset. Important note for students and post-docs: we hope to know the contact details of your academic supervisor. By sending the e-mail you accept the following terms and conditions.

### Terms and Conditions
When you download and use the mmBody dataset, please carefully read the following terms and conditions. Downloading and using the dataset means you have read and agree to them. Any violation of the terms of this agreement will automatically terminate your rights under this license.
The materials are developed at the State Key Laboratory of Industrial Control Technology in Zhejiang University. Any copyright and patent right are owned by it.
- The dataset is only available for non-commercial academic research and education purposes.
- Any other use, including incorporation in a commercial product, use in a commercial service, and further development of commercial products is banned.
- The dataset can't be modified, re-sold, or redistributed without our permission.
- Please cite the paper when you make use of the dataset and idea.
For any questions about the dataset, please send an email to anjun_chen@outlook.com

## Specifications of Radar
https://arberobotics.com/wp-content/uploads/2021/05/4D-Imaging-radar-product-overview.pdf

## Data Explanation

### Mesh Results

The mesh results include parameters of SMPL-X and 3D coordinates of joints and vertices. For the "pose" parameters of each frame, the first 3 dimensions denote the root translation, and 4-6 represent the root rotation. The rest of the dimensions represent rotations of 21 body joints. For shape parameters beta, we use 16 dimensions to represent human body shape. So please use the corresponding interface for generating the mesh. The SMPL-X model can be downloaded from https://smpl-x.is.tue.mpg.de/index.html.

### Dimensions of Radar Point Cloud
Each point of the radar point cloud for a frame contains its 3D location, range velocity, amplitude, and energy power of a reflected wave of the corresponding point in the scene. For the 3D location, X represents horizontal, Y represents depth, Z represents height, and the unit is in meter. 

### Calibration

We provide the calibration matrix for three sub-systems. We set the mmWave radar as the world coordinate system and transform the labels obtained from the MoCap system to it. You can also transform the coordinates to the sub-system by using the calibration matrix. The transformation matrixes in the calib.txt are in the sensor2world format. 

### Intrinsic of Cameras

```python
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
```

## Baselines
Our mesh results are obtained by fitting the [SMPL-X](https://smpl-x.is.tue.mpg.de/) body model to markers attached to subjects using [MoSh++](https://github.com/nghorbani/moshpp). We use [P4Transformer](https://github.com/hehefan/P4Transformer) and [VIBE](https://arxiv.org/abs/1912.05656) for the point- and image-based reconstruction. Please consider citing them if they also help on your project.

## BibTeX
```
@inproceedings{chen2022mmbody,
    title={mmBody Benchmark: 3D Body Reconstruction Dataset and Analysis for Millimeter Wave Radar},
    author={Chen, Anjun and Wang, Xiangyu and Zhu, Shaohao and Li, Yanxu and Chen, Jiming and Ye, Qi},
    booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
    pages={3501--3510},
    year={2022}
}
```
