# PyTorch implementation of CDistNet: Perceiving Multi-Domain Character Distance for Robust Text Recognition

The unofficial code of [CDistNet](https://arxiv.org/abs/2111.11011).

Now, we have implemented all the modules according to the papaer except for TPS in the visual branch.You can refer [ASTER](https://github.com/ayumiymk/aster.pytorch) for the implementation of TPS.

## Requirements

```bash
Python3.6.8
lmdb==0.98
torch==1.5.1
torchvision==0.6.1
Pillow==6.1.0
opencv-python==4.2.0.32
numpy==1.17.1
```

## Data preparation

We offer you a tool to transform raw dataset to LMDB dataset. Details please refer to tools/create_lmdb_dataset.py

You can also download lmdb dataset from [OCR_Dataset](https://github.com/WenmuZhou/OCR_DataSet)

## Train

First you need to modify some arguments in configs/cdistnet.yml.

- `TrainReader` set the path of train lmdb dataset.
- `EvalReader` set the path of evaluation lmdb dataset.
- `Global` set the args like image_shape, dict_file, etc.
- `VisualModule` set the args of visual branch in the original paper.
- `PositionalEmbedding` set the args of positional branch.
- `SemanticEmbedding` set the args of semantic branch.
- `MDCDP` set the args of MDCDP.

```bash
python train.py -c configs/cdistnet.yml
```

## Demo

Modify these arguments below in configs/cdistnet.yml.

- `pretrain_weights` set the path of model file path.
- `infer_img` set the image path.
- `is_train set to False.

```bash
python predict.py -c configs/cdistnet.yml
```

## TODO

- [ ] Pretrained models
- [ ] Test code
- [ ] Comparison with original paper on benchmarks(CUTE, IC13, IC15, IIIT5K, SVT, SVTP)