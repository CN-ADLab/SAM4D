# SAM4D

> Jianyun Xu†, Song Wang†, Ziqian Ni†, Chunyong Hu, Sheng Yang*, Jianke Zhu, Qiang Li

This is the official implementation of **SAM4D: Segment Anything in Camera and LiDAR Streams** (ICCV2025)  [[Paper](https://arxiv.org/abs/2506.21547)] [[Project Page](https://sam4d-project.github.io/)].

## Abstract

We present SAM4D, a multi-modal and temporal foundation model designed for promptable segmentation across camera and LiDAR streams. Unified
Multi-modal Positional Encoding (UMPE) is introduced to align camera and LiDAR features in a shared 3D space, enabling seamless cross-modal
prompting and interaction. Additionally, we propose Motion-aware Cross-modal Memory Attention (MCMA), which leverages ego-motion
compensation to enhance temporal consistency and long-horizon feature retrieval, ensuring robust segmentation across dynamically changing
autonomous driving scenes. To avoid annotation bottlenecks, we develop a multi-modal automated data engine that synergizes VFM-driven video
masklets, spatiotemporal 4D reconstruction, and cross-modal masklet fusion. This framework generates camera-LiDAR aligned pseudo-labels at a
speed orders of magnitude faster than human annotation while preserving VFM-derived semantic fidelity in point cloud representations. We
conduct extensive experiments on the constructed Waymo-4DSeg, which demonstrate the powerful cross-modal segmentation ability and great
potential in data annotation of proposed SAM4D.

<p align="center"> <a><img src="assets/teaser.png" width="90%"></a> </p>



## Installation

<details>
<summary>
Please ensure that your Linux system has cuda 12.1 or above installed, along with nvcc.
</summary>

```bash
#If you dont have, you may install cuda via:
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run
sudo sh cuda_12.3.2_545.23.08_linux.run \
     --silent \
     --toolkit \
     --toolkitpath=/usr/local/cuda-12.3 \
     --no-opengl-libs \
     --override
```

</details>

1. Create a conda environment and install torch

```bash
conda create -n sam4d python=3.8
conda activate sam4d
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

2. Clone the repo and install dependencies

```bash
git clone https://github.com/CN-ADLab/SAM4D.git && cd SAM4D
pip install -r requirements.txt
python setup.py develop
```

> Note: If you encounter google/dense_hash_map issue with installing torchsparse, please install sparsehash first
> via `sudo apt update && sudo apt install libsparsehash-dev`.


## Reproducible dependency installation

The reference environment in this repository is Python 3.8 with PyTorch 2.3.1, torchvision 0.18.1, torchaudio 2.3.1 and CUDA 12.1. For a different PyTorch/CUDA environment, please validate the full compatibility tuple before running the demo. In particular, the `mine` branch of the TorchSparse dependency is compiled from source; a build can fail against newer PyTorch C++ headers if the branch does not contain the corresponding `Tensor.scalar_type()` dispatch updates.

Install the framework first and install source-built dependencies only after PyTorch imports successfully. If a source package's isolated build environment cannot import the already-installed framework, use `--no-build-isolation` for that package rather than reinstalling or replacing PyTorch. Keep the exact TorchSparse commit and build log when reproducing an installation issue.

A successful package installation or `pip check` is not a runtime validation. At minimum, verify that `torchsparse.backend.__file__` resolves to a compiled extension outside the source checkout and run a small known-value sparse-convolution test before loading SAM4D.

The official installation instructions target Linux NVIDIA CUDA environments. Other accelerator SDKs may provide platform-specific PyTorch packages and require separate validation; this repository does not claim official support for non-NVIDIA accelerators.

## Quick Start
### Prepare Data
Download dataset Waymo-4DSeg from here: [modelscope](https://modelscope.cn/datasets/StarsMyDestination/Waymo-4DSeg). You can put the extracted example data from `Waymo-4DSeg/samples.zip` to `./data`.
The data structure are as follows:

```bash
${dataset}
├── meta_infos
│   └── ${sequence_name}.pkl
├── pcds
│   └── ${sequence_name}
│       ├── {timestamp1}.npz
│       ├── {timestamp2}.npz
│       └── ...
├── sam4d_labels (optional)
│   └── ${sequence_name}
│       ├── {timestamp1}.json
│       ├── {timestamp2}.json
│       └── ...
└── undistort_images
    └── ${sequence_name}
        ├── ${timestamp1}
        │   ├── ${cam_name}.jpg
        │   └── ...
        ├── ${timestamp2}
        │   ├── ${cam_name}.jpg
        │   └── ...
        └── ...
```

meta_infos: a pickle file containing meta information of the sequence
```python
# meta_infos/${sequence_name}.pkl structure:
from typing import Dict, List, Tuple, Union

MetaInfoType = Dict[str, Union[
    str,
    List[Dict[str, Union[
        Dict[str, Dict[str, Union[
            str,
            List[List[float]],
            None
        ]]],
        Dict[str, str],
        List[List[float]]
    ]]]
]]

example_meta_info: MetaInfoType = {
    'seq_name': 'your_sequence_name',
    'frames': [
        {
            'cams_info': {
                'your_cam_name': {
                    'data_path': 'undistort_images/your_sequence_name/your_timestamp/your_cam_name.jpg', # path to image
                    'camera_intrinsics': [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],  # 3x3 matrix
                    'camera2lidar': [[...], [...], [...], [...]]  # 4x4 matrix
                },
                'your_cam_name2': {...}
            },
            'path': {
                'pcd': 'pcds/your_sequence_name/your_timestamp.npz', # path to point cloud
            },
            'lidar2world': [[...], [...], [...], [...]]  # 4x4 matrix
        }
    ]
}
```


You can follow `data/visualize_sam4d_labels.ipynb` to explore the dataset. Make sure you have extracted `samples.zip` to `./data`.


### Run Demo
Download the model checkpoints from: [modelscope](https://modelscope.cn/models/StarsMyDestination/SAM4D/files).

Please follow `notebooks/sam4d_predictor_example.ipynb` step by step to proceed. Make sure you have extracted `samples.zip` to `./data`.



## Acknowledgement

We gratefully acknowledge the developers of the following open-source projects and datasets, whose foundational tools enabled our
research: [SAM2](https://github.com/facebookresearch/sam2), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2), [Waymo Open Dataset](https://waymo.com/open), [VDBFusion](https://github.com/PRBonn/vdbfusion),
among others.


## Citing SAM4D

```bibtex
@article{xu2025sam4d,
  title={SAM4D: Segment Anything in Camera and LiDAR Streams},
  author={Xu, Jianyun and Wang, Song and Ni, Ziqian and Hu, Chunyong and Yang, Sheng and Zhu, Jianke and Li, Qiang},
  journal={arXiv preprint arXiv:2506.21547},
  year={2025}
}
```
