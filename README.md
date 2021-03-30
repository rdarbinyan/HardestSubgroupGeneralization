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
  batch_size: 128                                                                                                                                                                                                                             
  num_workers: 0                                                                                                                                                                                                                              
  confounder_name: Male                                                                                                                                                                                                                       
  target_name: Blond_Hair                                                                                                                                                                                                                     
  sampler: None                                                                                                                                                                                                                               
network:                                                                                                                                                                                                                                      
  name: resnet50                                                                                                                                                                                                                              
  pretrained: true                                                                                                                                                                                                                            
  num_target_classes: 2                                                                                                                                                                                                                       
optimizer:                                                                                                                                                                                                                                    
  name: sgd                                                                                                                                                                                                                                   
  lr: 0.0001                                                                                                                                                                                                                                  
  momentum: 0.9                                                                                                                                                                                                                               
  weight_decay: 0.0001
  nesterov: false
trainer:
  name: standard
  gpus: 1
  precision: 16
  max_epochs: 50
  checkpoint_callback: false
  group_dro: false
  group_weight_step: 0.1
  log_every_n_steps: 50
  progress_bar_refresh_rate: 50
scheduler:
  name: disabled
logger:
  name: tensorboard
  run_name: default
  run_version: null
hsic:
  name: constant_weight
  weight: 0
```