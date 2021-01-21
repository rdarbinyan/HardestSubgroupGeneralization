# Setup 
```` 
conda env create -f environment
````

# Run Training

```
conda activate diploma

python src/train.py dataset.data_root={Root directory where images are downloaded to}
```

# Default Configs
```
random_seed: 0
logs_root_dir: ./logs
dataset:
  name: celeba
  data_root: /home/rafayel/datasets
  download: false
  batch_size: 64
  num_workers: 4
  confounder_name: Male
  target_name: Blond_Hair
  sampler: weighted
network:
  name: resnet50
  pretrained: true
  num_target_classes: 2
optimizer:
  name: adam
  lr: 0.001
trainer:
  name: standard
  gpus: 1
  max_epochs: 30
  log_every_n_steps: 50
  progress_bar_refresh_rate: 50
scheduler:
  name: plateau
  factor: 0.1
  patience: 5
  mode: min
  threshold: 0.0001
  cooldown: 0
  eps: 1.0e-08
  verbose: false
logger:
  name: tensorboard
  run_name: default
  run_version: null
hsic:
  name: constant_weight
  weight: 10
  on_output: true
```