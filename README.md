### University of Pretoria
#### COS711 - Assignment 3: Radio Sources Categorisation using Deep Learning Pipeline

#### Project Summary











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
    dataset_preprocessing.py
    model.py
    train.py
    predict.py
data/ 
Output/
README.md
```

#### Pull Request
+ `main` up to date
+ code runs
+ docs/README updated where needed
+ don't need to commit data files






