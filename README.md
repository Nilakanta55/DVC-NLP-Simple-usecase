# dvc-project-template
DVC project template

## Reference repository:
* [official reference repo](https://github.com/iterative/example-get-started)

* [DVC STUDIO](https://studio.iterative.ai/)

* [MY VIEW](https://studio.iterative.ai/user/Nilakanta55/views/DVC-NLP-Simple-usecase-ls9z5setn5)
## STEPS -

### STEP 01- Create a repository by using template repository

### STEP 02- Clone the new repository

### STEP 03- Create a conda environment after opening the repository in VSCODE

```bash
conda create --prefix ./env python=3.7 -y
```

```bash
conda activate ./env
```
OR
```bash
source activate ./env
```

### one shot create and activate environment 
```bash
conda create --prefix ./env python=3.7 -y && source activate ./env
```

### STEP 04- install the requirements
```bash
pip install -r requirements.txt
```

### STEP 05- initialize the dvc project
```bash
dvc init
```

### STEP 06- commit and push the changes to the remote repository



## extra command -

echo "*.log" >> logs/.gitignore

git rm --cached logs/running_logs.log