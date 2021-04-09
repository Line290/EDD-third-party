# EDD-third-party
The third-party implements of Encoder Dual Decoder method for table recognition  
**Paper:** [*Image-based table recognition: data, model, evaluation*][paper]  
**Official implements: [link][EDD_orig_repo_link]**
# Requirements
```bash
pip install -r requirements
```

# Training and testing on PubTabNet
### Prepare training data & Training
```bash
bash run_train.sh
```
### Prepare inference data & Inference with beam search
```bash
bash run_infer.sh
```
### Model parameters
Trained model with settings as shown in `run_train.sh` can download from [google drive][model].


[EDD_orig_repo_link]:https://github.com/ibm-aur-nlp/EDD
[paper]:https://arxiv.org/pdf/1911.10683.pdf
[model]:https://drive.google.com/file/d/1e2SJ-3A5k0Q4ouaTPfmzindZ3ksjcXh9/view?usp=sharing