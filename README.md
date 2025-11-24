# Learned Compression of Point Cloud Geometry and Attributes in a Single Model through Multimodal Rate-Control 

[![Paper](TODO)](TODO)


## Table of Contents

- [Overview](#overview)
- [Approach](#approach)
- [Results](#results)
- [Usage](#usage)
- [Citation](#citation)

## Overview

## Usage
We used Python 3.10.12 for our experiments.

### Setup
Set up the virtual environment
```
python -m venv .env
source .env/bin/activate
python -m pip install -r requirements.txt
```

### MinkowskiEngine
For CUDA > 12, we will need to patch the headers in some code and install MinkowskiEngine locally:
```
git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowkskiEngine
sed -i '1 i\#include <thrust/execution_policy.h>' src/3rdparty/concurrent_unordered_map.cuh \
    && sed -i '1 i\#include <thrust/execution_policy.h>' src/convolution_kernel.cuh \
    && sed -i '1 i\#include <thrust/unique.h>\n#include <thrust/remove.h>' src/coordinate_map_gpu.cu \
    && sed -i '1 i\#include <thrust/execution_policy.h>\n#include <thrust/reduce.h>\n#include <thrust/sort.h>' src/spmm.cu
python setup.py install --force_cuda --blas=openblas
```


# Open3D
```
git clone https://github.com/isl-org/Open3D
cd Open3D

sudo apt-get install libosmesa6-dev
util/install_deps_ubuntu.sh

mkdir build && cd build

cmake -DENABLE_HEADLESS_RENDERING=ON \
                -DBUILD_GUI=OFF \
                -DBUILD_WEBRTC=OFF \
                -DUSE_SYSTEM_GLEW=OFF \
                -DUSE_SYSTEM_GLFW=OFF \
                ..

make -j$(nproc)
make install-pip-package
```

# PCQM
```
git clone https://github.com/MEPP-team/PCQM.git
mkdir PCQM/build && cd PCQM/build
cmake ..
make
```

# G-PCC
```
git clone https://github.com/MPEGGroup/mpeg-pcc-tmc13.git
cd mpeg-pcc-tmc13
mkdir build && cd build
cmake ..
make
```

# V-PCC
You will have to build with VTM Lib video codec (Hacky Solution: Set USE_VTMLIB_VIDEO_CODEC to true in CMakeLists.txt before building)
```
git clone https://github.com/MPEGGroup/mpeg-pcc-tmc2.git --branch release-v24.0
cd mpeg-pcc-tmc2 && ./build.sh
```

# IT-DL-PCC
```
git clone https://github.com/aguarda/IT-DL-PCC.git
```
Download the weights from https://github.com/aguarda/IT-DL-PCC.git and place the unzipped repository in ./dependencies/IT-DL-PCC

### Metrics
If you have access to the mpeg-pcc-dmetric repositry, install and compile it into the dependencies folder.
The results in the paper where computed using aformentioned repository.
We supply a simplified python metric implementation as fallback solution to compute the metrics. 
The evaluation script checks for the mpeg implementation and resorts to fallback if it is not in the dependencies folder.

### Preparing the Dataset
We use the [8iVFBv2](http://plenodb.jpeg.org/pc/8ilabs) and the [Owlii](https://plenodb.jpeg.org/pc/microsoft) dataset for testing. 
The test sequences with normals are contained in the GitHub repository.

For training, we sample point clouds from [UVG-VPC](https://ultravideo.fi/UVG-VPC/)
To download the UVG-VPC dataset automatically, run
```
cd data
python download_raw_pointclouds.py 
```

(This downloads raw data for all 3 datasets, so it will fill up your disk)
Datasets are specified in a config file (pointcloud and frames), a dataset configuration can be found in ./data/datasets/full_128.


### Training
We provide a configuration in ./configs for training our model.

```
python train.py --config=./configs/Ours.yaml
```

Training takes roughly 1-2 days on an NVIDIA RTX 4090. 


### Results
Use the weights from here []() to rerun the evaluation with our dataset.

You can get the test data from [https://plenodb.jpeg.org/](https://plenodb.jpeg.org/).
Download the **JPEG Pleno PCC CTTC Point Clouds** dataset and the **8i Voxelized Full Bodies** dataset.
Unpack all point clouds, and put them into data/datasets/8iVFBv2 and data/datasets/jpeg_testset. 

Then re-run the evaluation using 
```python3
    python evaluate.py
```





## Citation

If you find our work helpful, please consider citing us in your work:
```
@article{rudolph2024learnedcompressionpointcloud,
      title={Learned Compression of Point Cloud Geometry and Attributes in a Single Model through Multimodal Rate-Control}, 
      author={Michael Rudolph and Aron Riemenschneider and Amr Rizk},
      journal={arXiv preprint arXiv:2408.00599},
      year={2024},
}
```
