# ZS-CSD 

Zero-Shot Conversational Stance Detection: Dataset and Approaches

## Quick Links
- [Overview](#overview)
- [Requirements](#requirements)
- [Code Usage](#code-usage)


## Overview
Stance detection, which aims to identify public opinion towards specific targets using social media data, is an important yet challenging task. With the increasing number of online debates among social media users, conversational stance detection has become a crucial research area. However, existing conversational stance detection datasets are restricted to a limited set of specific targets, which constrains the effectiveness of stance detection models when encountering a large number of unseen targets in real-world applications. To bridge this gap, we manually curate a large-scale, high-quality zero-shot conversational stance detection dataset, named ZS-CSD, comprising 280 targets across two distinct target types. Leveraging the ZS-CSD dataset, we propose SITPCL, a speaker interaction and target-aware prototypical contrastive learning model, and establish the benchmark performance in the zero-shot setting. Experimental results demonstrate that our proposed SITPCL method achieves state-of-the-art performance in zero-shot conversational stance detection, ranking second only to GPT-4 while surpassing GPT-3.5 and LLaMA 3-70B. Notably, even GPT-4 attains only an F1-macro score of 48.62\%, highlighting the persistent challenges in zero-shot conversational stance detection.



## Requirements

This project uses PyTorch for its implementation. Please ensure your system includes the following package versions:

- Python: 3.7+ 
- PyTorch: 1.13.1+

Additional required packages can be installed via pip:
```bash
pip install -r requirements.txt
```



## Code Usage

### For Training and Evaluating on the CZ-CSD Dataset
Run the following script to train and evaluate:
```bash
python main.py
```

### Customizing Hyperparameters
Hyperparameter settings are flexible and can be adjusted within either `main.py` or `src/config.yaml`. Note that configurations in `main.py` will override any settings in `src/config.yaml`.
