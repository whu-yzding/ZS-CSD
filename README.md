# ZS-CSD 

Zero-Shot Conversational Stance Detection: Dataset and Approaches.

## Quick Links
- [Overview](#overview)
- [Requirements](#requirements)
- [Code Usage](#code-usage)
- [Citation](#citation)


## Overview
This repository open-sources the code and datas in our [paper](https://aclanthology.org/2025.findings-acl.168/)「**Zero-Shot Conversational Stance Detection: Dataset and Approaches**」in ACL2025 findings long paper.

Please cite our paper and kindly give a star for this repository if you use our code or data.



## Requirements

This project uses PyTorch for its implementation. Please ensure your system includes the following package versions:

- Python: 3.7+ 
- PyTorch: 1.13.1+

Additional required packages can be installed via pip:
```bash
pip install -r requirements.txt
```



## Code Usage

### For Training and Evaluating on the ZS-CSD Dataset
Run the following script to train and evaluate:
```bash
python main.py
```

### Customizing Hyperparameters
Hyperparameter settings are flexible and can be adjusted within either `main.py` or `src/config.yaml`. Note that configurations in `main.py` will override any settings in `src/config.yaml`.


## Citation

The BibTex of the citation is as follows:

```bibtex
@inproceedings{ding-etal-2025-zero,
    title = "Zero-Shot Conversational Stance Detection: Dataset and Approaches",
    author = "Ding, Yuzhe  and
      He, Kang  and
      Li, Bobo  and
      Zheng, Li  and
      He, Haijun  and
      Li, Fei  and
      Teng, Chong  and
      Ji, Donghong",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2025",
    year = "2025",
    pages = "3221--3235"
}
```
