### University of Pretoria
#### COS711 - Assignment 3: Radio Sources Categorisation using Deep Learning Pipeline

#### Project Summary

This project focuses on classifying **MeerKAT radio galaxy sources** using three deep learning approaches implemented collaboratively by the team:

- **Basic CNN** A custom convolutional neural network built from scratch to establish a baseline for radio morphology classification.  
- **Transfer Learning (ResNet-50):** Fine-tuned a pretrained ResNet-50 model on 1,970 labeled MeerKAT images to leverage ImageNet features for feature extraction and generalisation.  
- **Pseudo-Labelling:** Semi-supervised approach that used confident predictions from the model to iteratively expand the training set with unlabeled MeerKAT data.  

The pipeline automates **data preparation**, **model training**, and **prediction** on unlabeled images.  
Final evaluation was performed and the ResNet-50 model showed stable performance and generalization across classes.  

---

### ResNet-50 Training and Prediction

```bash
# Split dataset into training and validation sets
\Assignment 3> python prepare_splits.py --labels_csv ../data/labels.csv --train_csv ../data/train.csv --val_csv ../data/val.csv

# Train ResNet-50 model
\Assignment 3> python -m src.train --train_csv data/train.csv --val_csv data/val.csv --out_dir Output/run_resnet50 --epochs 25 --batch 32 --lr 1e-4

# Run inference on unlabeled MeerKAT images
\Assignment 3> python -m src.resnet.predict --ckpt "Output/run_resnet50/best.pt" --unl_dir "data/unl" --num_classes 9 --out_csv "Output/run_resnet50/preds.csv"
```

---
 
#### Collaborator Guide

Use a **branch -> PR -> review -> merge** process Keep `main` stable.

#### 0) Initial setup
```bash
git clone https://github.com/zingisamatwana/cos711-a3radio-classification-dl.git
cd cos711-a3-radio-classification-dl
python -m venv .venv
.venv\Scripts\Activate.ps1 # windows (Powershell) / lookup on the internet cmd for macOS/Linux
pip install -r requirements.txt
```
1) Data Policy
    * Don't commit datasets or output results. .gitignore file prevent them from getting staged when `git add .`
    * Put raw files in data/ locally, put generated results in Output/.

2) Branch Naming
```bash
git checkout main
git pull origin main
git checkout -b feat/topic # like docs/readme-workflow or fix/bug-name or model/training etc..
```
3) Commit Changes
```bash
git status
git add src/files or . README.md
git commit -m "feat: brief description" # other commits {feat:, fix: docs:, test:, refactor: etc..}
```
4) Push & Open a Pull Request
```bash
git push -u origin branch-name # like "feat/topic"
```
Open a PR on GitHub:
* Base: main
* Request 1 review/approval before merge
* Resolve all PR comments

5) After Merge
```bash
git checkout main
git pull origin main
git branch -d branch-name # like "feat/topic"
git push origin --delete branch-name # like "feat/topic" (delete remote branch "feat/topic")
```
#### Folder Layout
```bash
scr/
    cnn/
        maishah_data_preparation.py
        maishah_predict.py
        maishah_train.py
    resnet/
        infer.py
        model.py
        predict.py
        train.py
    pseudolabeling/
        Pseudo (1).ipynb
    config.py
    data.py
    prepare_splits.py
data/ 
Output/
README.md
```

#### Pull Request
+ `main` up to date
+ code runs
+ docs/README updated where needed
+ don't need to commit data files






