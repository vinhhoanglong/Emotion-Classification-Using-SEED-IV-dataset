# Emotion-Classification-Using-SEED-IV-dataset

## Download SEED-IV dataset 
request download using think link https://bcmi.sjtu.edu.cn/~seed/seed-iv.html

## Create a data directory
```bash
cd Emotion-Classification-Using-SEED-IV-dataset
sudo mkdir data
sudo mv /path/to/your/downloaded/seed-iv/ data/
```

## Create virtual environment
```bash
python -m venv .venv
```

### Install dependencies
```bash
.venv/bin/pip install -r requirements.txt
```

### Evaluation
with params model, choice = ['dense', 'lstm', '3dconv']
```bash
.venv/bin/python main.py --model=dense
```
