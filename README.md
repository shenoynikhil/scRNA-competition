# scRNA-competition
Open Problems - Multimodal Single-Cell Integration - NeurIps 2022 Challenge

### Downloading Dataset
To download the dataset, run the following commands,

```bash
# download the dataset from kaggle
kaggle competitions download -c open-problems-multimodal

# unzip the zip file
unzip open-problems-multimodal.zip -d data
```

### Installing Dependencies

```bash
# inside a conda env
conde create -n comp python=3.8
conda activate comp
pip install -r requirements.txt
```
