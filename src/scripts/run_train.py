import os

from src.dataset.datamodule import get_datamodule
from src.train.train import train, test
from src.train.training_module import Training
from src.utils.mlflow_utils import mlflow_session
from src.utils.utils import setup_seed

os.environ['TORCH_CUDA_ARCH_LIST'] = "9.0+PTX"  # per nuove GPU

from src.config import initialize_configuration

# Params
run_params = initialize_configuration()
setup_seed(run_params.seed)
print('Configuration settled!')

with mlflow_session(run_params):
    # Data
    dataModuleInstance, run_params = get_datamodule(run_params)
    print('Data imported!')

    # Model
    training_module = Training(run_params); print('Training module defined!')

    # Train
    print('Start training!'); train(training_module, dataModuleInstance, run_params); print('End training!')

    # Test
    print('Start testing!'); test(training_module, dataModuleInstance, run_params); print('End testing!')


