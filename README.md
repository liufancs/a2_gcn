# An Attribute-aware Attentive GCN Model

This is our implementation for the paper:

Fan Liu, Zhiyong Cheng*, Lei Zhu, Chenghao Liu and Liqiang Nie*. [An Attribute-aware Attentive GCN Model for Attribute Missing in Recommendation]


## Environment Settings
- Tensorflow-gpu version:  1.3.0

## Example to run the codes.

Run a2_gcn.py
```
python a2_gcn.py --dataset Toys_and_Games --num_neg 4 --embed_size 64 --batch_size 1024 --layer_size [64,64,64] --pretrain -1
```

### Dataset

train.txt:
- Train file.

test.txt:
- Test file.

tag.txt:
- Tag file.
