# An Attribute-aware Attentive GCN Model

This is our implementation for the paper:

Fan Liu, Zhiyong Cheng*, Lei Zhu, Chenghao Liu and Liqiang Nie*. [An Attribute-aware Attentive GCN Model for Attribute Missing in Recommendation]


## Environment Settings
- Tensorflow-gpu version:  1.3.0

## Example to run the codes.

Run a2_gcn.py
```
python a2_gcn.py --dataset Toys_and_Games  --embed_size 64 --batch_size 1024 --layer_size [64,64,64] --lr 0.00005 --pretrain -1
```

### Dataset
We provide two processed datasets: Amazon-Toys&Games, Amazon-Kindle Store (Kin).
Each dataset file contains:

train.txt:
- Train file.

test.txt:
- Test file.

tag.txt:
- Tag file.

We also provide the pretrained features, which can improve the speed of convergence. The pre-trained feature (Kindle Store) could be downloaded from :
- Link:  https://pan.baidu.com/s/1-f0BsPs-kPZSupzb--EdLA
- Extract code:  ditf 
