# Event-based Few-shot Fine-grained Human Action Recognition

## Environment Setup

### Installation

1. Create a new conda env. and install `pytorch` with conda. Our `pytorch` version is 1.12.0 and `cuda` version is 11.3. 

```bash
conda create -n afford python=3.8
conda activate afford
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

2. Install other dependencies with `pip`.

```bash
pip install -r requirements.txt
```

## Dataset
For raw event data of E-FAction, please contact yangzl_xx@outlook.com.

## EventEncoder

## Our Framework
### Train

```bash
bash shells/ablation/finetune_knowledgeModel_dwconv_fc_ablationNoOrig.sh
```
### Test

```bash
bash shells/ablation/test_knowledgeModel_dwconv_fc_ablationNoOrig.sh
```

## Citation
If you would like to use our code or dataset, please cite either
```
@inproceedings{yang2024event,
  title={Event-based Few-shot Fine-grained Human Action Recognition},
  author={Yang, Zonglin and Yang, Yan and Shi, Yuheng and Yang, Hao and Zhang, Ruikun and Liu, Liu and Wu, Xinxiao and Pan, Liyuan},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={519--526},
  year={2024},
  organization={IEEE}
}
```

### License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
