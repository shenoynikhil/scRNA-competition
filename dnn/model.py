import pytorch_lightning as pl
from pl_models import BaseNet, ContextConditioningNet
from datamodule import DataModule
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping


class DNNSetup:
    '''Setup for training deep learning networks setup using Pytorch Lightning. This interface basically
    helps call,
    ```
    trainer = Trainer()
    model = Model()
    datamodule = DataModule()

    trainer.fit(model, datamodule)
    '''
    def __init__(self, config):
        self.config = config
        self.output_dir = self.config['output_dir']

    def setup_model(self, model_config: dict):
        model_type = model_config.get('model_type', 'BaseNet')
        if model_type == 'BaseNet':
            return BaseNet(
                input_dim = model_config.get('input_dim', 128), 
                output_dim = model_config.get('output_dim', 100),
                hp = {
                    'layers': model_config.get('layers', [170, 300, 480, 330, 770]),
                    'dropout': model_config.get('dropout', 0.2),
                }   
            )
        elif model_type == 'ContextConditioningNet':
            return ContextConditioningNet(
                context_dim=model_config.get('context_dim', 10),
                input_dim=model_config.get('input_dim', 128),
                output_dim=model_config.get('output_dim', 100),
                beta=model_config.get('beta', 1e-3),
                hp={
                    'layers': model_config.get('layers', [170, 300, 480, 330, 770]),
                    'dropout': model_config.get('dropout', 0.2), 
                }
            )
        else:
            return NotImplementedError

    def setup_trainer(self, trainer_config: dict):
        params = {
            'accelerator':'gpu',
            'devices':1,
            'logger':False,
            'num_sanity_val_steps':trainer_config.get('max_epochs', 0),
            'max_epochs':trainer_config.get('max_epochs', 200),
            'callbacks':[
                TQDMProgressBar(refresh_rate=1000),
                EarlyStopping(monitor="val/pcc", mode="max", patience=20)
            ],
        }
        return pl.Trainer(**params)

    def setup_datamodule(self, datamodule_config: dict):
        return DataModule(
            x_path=datamodule_config.get('x_path'),
            y_path=datamodule_config.get('y_path'),
            x_test_path=datamodule_config.get('x_test_path'),
            splits_path=datamodule_config.get('splits_path', None),
            batch_size=datamodule_config.get('batch_size', 128), 
        )
    
    def run_experiment(self):
        '''Performs the experiment'''
        trainer = self.setup_trainer(self.config.get('trainer_config', {}))
        model = self.setup_model(self.config.get('model_config', {}))
        datamodule = self.setup_model(self.config.get('datamodule_config'))

        trainer.fit(model, datamodule)
