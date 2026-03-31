# Title 

This repository contains code for the paper [Unified Compression of Point Cloud Geometry and Attributes through Variable-Rate Conditioning]() presented at the ACM MMSys'26 Conference.

Essentially, we provide a method for training a learning-based compression model with a single encoder and decoder which allows to adaptively control the geometry and attribute quality and thus coding rate during inference.

***News:*** 
- We showed a [Demo](https://dl.acm.org/doi/abs/10.1145/3712676.3719266) of a preliminary version of this model at the MMsys'25 Conference for Streaming with a Live Recording and 2 Jetson Devices.
- An early stage of this work is available in the [Pre-Print](https://arxiv.org/abs/2408.00599). The code of the pre-print results can be found in branch ***pre-print***


## Data 
### Testset
Download the [JPEG Pleno CPCC CTTC Point Clouds](https://plenodb.jpeg.org/pc/JPEG_Pleno_PCC_CTTC.zip) dataset and place the *.ply files under _./data/datasets/jpeg_testset/_

### Training Dataset
The training set is manually collected from various sources. For reconstructring the dataset, collect the point clouds listed in _./data/datasets/jpeg_128/config.yaml_ into _./data/datasets/jpeg_testset](./data/datasets/jpeg_trainset_

Alternatively, ready-made scripts for a easily available training routine on the UVG Point Cloud dataset are available. 
```
cd data/utils
python3 download_raw_pointclouds.py
```
Note that the results in the publication were generated using the jpeg trainset to allow for fair comparison to the JPEG Pleno PCC coding solution.



## Setup
We used Python 3.10.12 for our experiments.

Set up the virtual environment
```
python3 -m venv .env
source .env/bin/activate
python -m pip install -r requirements.txt
```

### MinkowskiEngine
For CUDA > 12, we will need to patch the headers in some code and install MinkowskiEngine locally:
```
cd dependencies
git clone --recursive https://github.com/NVIDIA/MinkowskiEngine
cd MinkowkskiEngine
sed -i '1 i\#include <thrust/execution_policy.h>' src/3rdparty/concurrent_unordered_map.cuh \
    && sed -i '1 i\#include <thrust/execution_policy.h>' src/convolution_kernel.cuh \
    && sed -i '1 i\#include <thrust/unique.h>\n#include <thrust/remove.h>' src/coordinate_map_gpu.cu \
    && sed -i '1 i\#include <thrust/execution_policy.h>\n#include <thrust/reduce.h>\n#include <thrust/sort.h>' src/spmm.cu
python setup.py install --force_cuda --blas=openblas
```

### Open3D
```
cd dependencies
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

### PCQM
```
git clone https://github.com/MEPP-team/PCQM.git
mkdir PCQM/build && cd PCQM/build
cmake ..
make
```

### G-PCC
```
git clone https://github.com/MPEGGroup/mpeg-pcc-tmc13.git
cd mpeg-pcc-tmc13
mkdir build && cd build
cmake ..
make
```

### V-PCC (optional, for related work comparison)
You will have to build with VTM Lib video codec (Hacky Solution: Set USE_VTMLIB_VIDEO_CODEC to true in CMakeLists.txt before building)
```
git clone https://github.com/MPEGGroup/mpeg-pcc-tmc2.git --branch release-v24.0
cd mpeg-pcc-tmc2 && ./build.sh
```

### IT-DL-PCC (optional, for related work comparison)
```
git clone https://github.com/aguarda/IT-DL-PCC.git
```
Download the weights from https://github.com/aguarda/IT-DL-PCC.git and place the unzipped repository in ./dependencies/IT-DL-PCC

### Metrics (optional, for related work comparison)
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
python train.py --config=./configs/Main.yaml
```

Training takes roughly 1-2 days on an NVIDIA RTX 4090. 


### Results
Use the weights from [here](https://github.com/ikt-luh/Unified-Point-Cloud-Compression/releases/tag/Main) to rerun the evaluation.
Download the **JPEG Pleno PCC CTTC Point Clouds** dataset and the **8i Voxelized Full Bodies** dataset.
Unpack all point clouds, and put them into data/datasets/8iVFBv2 and data/datasets/jpeg_testset. 

Then re-run the evaluation using 
```
python evaluate.py
```


## Citation

If you find our work helpful, please consider citing:
```
@inproceedings{rudolph2026unified,
    title={Unified Compression of Point Cloud Geometry and Attributes through Variable-Rate Conditioning}, 
    author={Michael Rudolph and Aron Riemenschneider and Amr Rizk},
    booktitle = {ACM Multimedia Systems Conference 2026 (MMSys '26)},
    year      = {2026},
    publisher = {ACM},
    doi       = {10.1145/3793853.3795742},
}
```
