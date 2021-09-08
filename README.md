# Semantic CycleGAN in Pytorch

## Getting Started

### Installation

- Clone this repo:

```shell
git clone -b semantic_cyclegan https://github.com/WMR-team/pytorch-CycleGAN-and-pix2pix.git
```

### Semantic CycleGAN train

- Download the MarsRock dataset:

```shell
cd datasets
git clone https://github.com/WMR-team/mars-rock-dataset.git
```

You need to download the [Rock](https://github.com/WMR-team/mars-rock-dataset) dataset into the **datasets** folder.

- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- Train a model:

```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/rocks --name rocks_semanticgan --model semantic_gan --batch_size 2 --lr 0.0001 --gpu_ids 0,1,2,3,4,5
```

Note: try to use 6 gpus first, if it is not feasible, try to use 4 gpus by using  `--gpus_ids 0,1,2,3`

The batch size can be modified according to the device capability by using `--batch_size [appropriate number]`

To see more intermediate results, check out `./checkpoints/semantic_cyclegan/web/index.html`.

