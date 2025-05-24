Official implementation for paper "[Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling](https://arxiv.org/abs/2410.01651)" (ICML 2025)

This repository is still under construction.

### Environments
torch==2.4.0, transformers>=4.36.0, triton==3.0.0

`pip install requirements.txt`

### Data Preparation

[ArXiv-math](https://huggingface.co/datasets/hoskinson-center/proof-pile), [PG19](https://huggingface.co/datasets/emozilla/pg19)

### Pre-training

`sh scripts/pretrain_pg19.sh`

### Downstream tasks finetuning

Summarization tasks

`sh scripts/xsum_ft.sh`


NIAH tests

`sh scripts/niah_ft.sh`

### Evaluation


### Contact
aaron.hx AT antgroup.com