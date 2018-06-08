# Visual Navigation - Semantic

### Deprecated - refer to https://github.com/Jiankai-Sun/Visual-Navigation-RGB/tree/baseline

## Requirements
* Python 2.7
* [House3D](https://github.com/facebookresearch/House3D)

## How to Run
Node `20200`
```
CUDA_VISIBLE_DEVICES=4 python train.py
```

Node `20300`
```
source /SSD2/jack/Applications/PythonEnv/python2
CUDA_VISIBLE_DEVICES=4 python train.py
```
Node `20500`
```
source /home/cxy/Applications/PythonEnv/python2
CUDA_VISIBLE_DEVICES=4 python train.py
```
## File Tree
```bash
.
├── checkpoints/
├── config.json
├── constants.py
├── evaluate.py
├── LICENSE
├── logs/
├── network.py
├── README.md
├── requirements.txt
├── roomnav.py
├── training_thread.py
├── train.py
└── utils
│     ├── accum_trainer.py
│     ├── __init__.py
│     ├── ops.py
│     ├── rmsprop_applier.py
│     └── tools.py
├── SUNCG
│   ├── house
│   │   └── 00065ecbdd7300d35ef4328ffe871505
│   │       ├── house.json
│   │       ├── house.mtl
│   │       └── house.obj
│   ├── room
│   │   └── 00065ecbdd7300d35ef4328ffe871505
│   │       ├── fr_0gd_0f.mtl
│   │       ├── fr_0gd_0f.obj
            ......
│   └── texture
│       ├── akugum.jpg
│       ├── Aurora.jpg
│       ├── beam_1.jpg
        .......

```

