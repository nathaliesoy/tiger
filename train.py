import comet_ml

# remove local paths, so that we don't use any
import sys, os, yaml
paths = sys.path
for p in paths:
    if '.local' in p:
            paths.remove(p)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from lightning import Tiger_Lit
import yaml

# Load the config file

config_name  = sys.argv[1]
with open(config_name) as file:
    config = yaml.safe_load(file)


experiment_name = config['name']
# Create a comet logger
if len(sys.argv) == 2:
    comet_logger = CometLogger(
        api_key="your api key",
        project_name="tiger",
        workspace="your workspace",
        experiment_name=experiment_name,
    )

    # get all .py files in the current directory
    for file in os.listdir('.'):
        if file.endswith('.py'):
            comet_logger.experiment.log_asset(file)
    for file in os.listdir('./models'):
        if file.endswith('.py'):
            comet_logger.experiment.log_asset('./models/'+file)
    
    comet_logger.experiment.log_asset(config_name)

else:
     comet_logger = None

checkpoint_callback = [ModelCheckpoint(
        dirpath = '/path/to/checkpoints/'+experiment_name,
        monitor='val_total',
        save_top_k = 3, 
        save_last = True,
        filename = '{epoch}-{val_total:.4f}'),  ModelCheckpoint(
        dirpath = '/path/to/checkpoints/'+experiment_name,
        monitor='val_stage1',
        save_top_k = 3, 
        save_last = True,
        filename = '{epoch}-{val_stage1:.4f}')]


# Create a lightning model
model = Tiger_Lit(config=config, comet_logger=comet_logger)

if comet_logger is not None:
    # log number of parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    comet_logger.experiment.log_parameter("trainable_params", trainable_params)

# Create a trainer
trainer = Trainer(
    max_epochs=config['num_epochs'],
    accelerator='gpu',
    devices=1,
    logger=comet_logger,
    callbacks= checkpoint_callback,
    check_val_every_n_epoch = config['check_val_every_n_epoch'],
    )

# Train the model
trainer.fit(model)