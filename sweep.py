import wandb
sweep_configuration = {
  # 搜索方法
    'method': 'bayes',
  # 在 wandb 中的名字
    'name': 'sweep',
  # 优化的指标
    'metric': {
        'goal': 'minimize', 
        'name': 'vali_loss'
        },
  # 搜索参数边界
    'parameters': {
        'd_model': {'values': [256,  1024]},
        'n_heads': {'values': [4,  16]},
        'e_layers': {'values': [2,  8]},
        'd_ff': {'values': [512, 2048]},
        'dropout': {'values': [0.1, 0.3, 0.5]},
        'learning_rate': {'max': 0.001, 'min': 0.00001}
     }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="iTransformer_first_full")
print(sweep_id)
# wandb.init() nq1jwopr

# # config 不用固定值 
# lr  =  wandb.config.lr
# bs = wandb.config.batch_size
# epochs = wandb.config.epochs

# wandb.agent(sweep_id, function=main, count=4)
