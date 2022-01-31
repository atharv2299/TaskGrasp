## Same Object, Different Grasps: Data and Semantic Knowledge for Task-Oriented Grasping
#### In CoRL 2020 [[arXiv]](https://arxiv.org/abs/2011.06431) [[Robot Video]](https://youtu.be/ByHVc-sPmd8) [[project page]](https://sites.google.com/view/taskgrasp) [[pdf]](https://arxiv.org/pdf/2011.06431.pdf)

[Adithya Murali](http://adithyamurali.com), [Weiyu Liu](http://weiyuliu.com/), [Kenneth Marino](http://kennethmarino.com/), [Sonia Chernova](https://www.cc.gatech.edu/~chernova/), [Abhinav Gupta](http://www.cs.cmu.edu/~abhinavg)

Carnegie Mellon University Robotics Institute, Georgia Institute of Technology, Facebook AI Research

<!-- <img src='images/hcp.png' width="400"> <img src="https://thumbs.gfycat.com/SafeNeighboringHydatidtapeworm-size_restricted.gif" width="400"> -->

This is a pytorch-based implementation for our [CoRL 2020 paper on task-oriented grasping](https://arxiv.org/abs/2011.06431). If you find this work useful in your research, please cite:

	@inproceedings{murali2020taskgrasp,
	  title={Same Object, Different Grasps: Data and Semantic Knowledge for Task-Oriented Grasping},
	  author={Murali, Adithyavairavan and Liu, Weiyu and Marino, Kenneth and Chernova, Sonia and Gupta, Abhinav},
	  booktitle={Conference on Robot Learning},
	  year={2020}
	}

The code has been tested on **Ubuntu 16.04** and with **CUDA 10.0**.

## Installation

1. Clone this repo:
	```bash
	mkdir ${HOME}/taskgrasp_ws && cd ${HOME}/taskgrasp_ws
	git clone https://github.com/atharv2299/TaskGrasp.git
	```

2. Install PyTorch and related libraries for your version of CUDA. For example, for 11.3:
	```bash
	python3 -m pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 \
		-f https://download.pytorch.org/whl/cu113/torch_stable.html
	python3 -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==1.5.0 \
		-f https://pytorch-geometric.com/whl/torch-1.10.0%2Bcu113.html
	```

3. Install other dependencies:
	```bash
	cd TaskGrasp
	python3 -m pip install -r requirements.txt
	```

4. Compile and install [PointNet ops](https://github.com/erikwijmans/Pointnet2_PyTorch)
	```bash
	cd ../
	git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
	cd Pointnet2_PyTorch
	CUDA_HOME=/usr/local/cuda python3 -m pip install -r requirements.txt
	```

## Dataset
The dataset (5 GB) could be downloaded [here](https://drive.google.com/file/d/1aZ0k43fBIZZQPPPraV-z6itpqCHuDiUU/view?usp=sharing) and place it in the `data` folder as shown below:
```shell
cd ~/taskgrasp_ws/TaskGrasp
unzip ~/Downloads/data.zip -d ./
rm ~/Downloads/data.zip
```
To run any of the demo scripts explained below, do the same with the [pretrained models (589 MB)](https://drive.google.com/file/d/1fasm-8MV6zBjdnbAHLbU8_8TZOkeABkR/view?usp=sharing) and [config files (10 KB)](https://drive.google.com/file/d/1vJfkaCCLJmvT8i5OB-qx_pOojgx2ouPf/view?usp=sharing) and put them in the `checkpoints` and `cfg` folder respectively.

**Coming Soon:** We are trying to release mesh models for the objects as well.

## Usage

**NOTE:** The stable grasps were sampled from the point cloud using the [agile_grasp](https://github.com/GT-RAIL/rail_agile) repo from [GT-RAIL](http://www.rail.gatech.edu/).

Point Cloud         |  Stable grasps
:-------------------------:|:-------------------------:
<img src="assets/pc.gif" width="256" height="256" title="pc">  |  <img src="assets/grasps.jpg" width="256" height="256" title="grasps">

To visualize the point cloud data and stable grasps:
```shell
python visualize.py --data_and_grasps --obj_name 124_paint_roller
```
Add the `--obj_path` argument with the absolute path to the dataset if you have placed it somewhere other than the default location (`data/taskgrasp`). The object can be specified with the `--obj_name` argument and the full list of objects can be found [here](gcngrasp/data/taskgrasp_objects.txt).

<img src="assets/tog.gif" width="512" height="512" title="tog">

To visualize the labelled task-oriented grasps in the TaskGrasp dataset:
```shell
python visualize.py --visualize_labels  --visualize_labels_blacklist_object 124_paint_roller
```

To visualize a specific grasp:
```shell
python visualize.py --obj_name 124_paint_roller --visualize_grasp --grasp_id 10
```

To run any of the training or evaluation scripts, download the config files from [here](https://drive.google.com/file/d/1vJfkaCCLJmvT8i5OB-qx_pOojgx2ouPf/view?usp=sharing) and put them in the cfg folder. To train a model:
```shell
python gcngrasp/train.py --cfg_file cfg/train/gcngrasp/gcngrasp_split_mode_t_split_idx_3_.yml
```

To run pretrained models, download the models from [here](https://drive.google.com/file/d/1fasm-8MV6zBjdnbAHLbU8_8TZOkeABkR/view?usp=sharing) and unzip them into the checkpoints folder. To evaluate trained model on test set:
```shell
python gcngrasp/eval.py cfg/eval/gcngrasp/gcngrasp_split_mode_t_split_idx_3_.yml --save
```

Here's how to run evaluation on a single point cloud (e.g. sample data of an unknown object captured from a depth sensor) and assuming you already have sampled stable grasps. Make sure you have downloaded the pretrained models and data from the previous step, and run the following:
```shell
python gcngrasp/infer.py cfg/eval/gcngrasp/gcngrasp_split_mode_t_split_idx_3_.yml --obj_name pan --obj_class pan.n.01 --task pour
```

<img src="assets/pan.jpg" width="300" height="256" title="pan">
