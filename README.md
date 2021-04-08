# EDD-third-party
The third-party implements of Encoder Dual Decoder method for table recognition  
**Paper:** [*Image-based table recognition: data, model, evaluation*][paper]  
**Official implements: [link][EDD_orig_repo_link]**
# Requirements
```bash
pip install -r requirements
```

# Training and testing on PubTabNet
## Prepare training data & Training
```bash
bash run_train.sh
```
## Prepare inference data & Inference with beam search
```bash
bash run_infer.sh
```




[EDD_orig_repo_link]:https://github.com/ibm-aur-nlp/EDD
[paper]:https://arxiv.org/pdf/1911.10683.pdf