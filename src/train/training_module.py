import torch
from torch import optim

from src.models.hipatch_adapter import to_hipatch_batch, compute_error
from src.models.mtand_adapter import to_mtand_batch
from src.models.tpatchgnn_adapter import to_tpatchgnn_batch
from src.utils.utils import get_model


class Training:
    def __init__(self, hparams):
        super(Training, self).__init__()
        self.args = hparams

        # Import model
        self.model, self.model_args = get_model(hparams)
        if hparams.compile_model:
            self.model = torch.compile(self.model)
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f'Parameters: {num_params}')

        # Training params
        self.scheduler = None
        self.avg_accuracy = None
        self.debug = False
        self.optimizer = None
        self.best_mse, self.best_mae, self.best_mape, self.best_rmse = float('inf'), float('inf'), float('inf'), float('inf')

    def forward(self, batch):
        if 'hi-patch' in self.args.model:
            bd = to_hipatch_batch(batch, self.model_args, self.args.device)
            pred_y = self.model.forecasting(bd["tp_to_predict"],
                                            bd["observed_data"],
                                            bd["observed_tp"],
                                            bd["observed_mask"])

        elif 'tpatch-gnn' in self.args.model:
            bd = to_tpatchgnn_batch(batch, self.model_args, self.args.device)
            pred_y = self.model.forecasting(bd["tp_to_predict"],
                                            bd["observed_data"],
                                            bd["observed_tp"],
                                            bd["observed_mask"],
            )
        elif 'mtand' in self.args.model:
            bd = to_mtand_batch(batch, self.args.device)
            pred_y = self.model.forecasting(bd["tp_to_predict"],
                                            bd["observed_data"],
                                            bd["observed_tp"],
                                            bd["observed_mask"],
            )
        elif self.args.model in ('lstm', 'dlinear', 'nbeats'):
            bd = None
            pred_y = self.model(batch[0])
        else:
            raise ValueError(f"Model '{self.args.model}' non riconosciuto.")
        return {'pred_y': pred_y, 'bd': bd}


    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.lr,
                                          weight_decay=self.args.w_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              mode='min',
                                                              patience=self.args.lr_patience,
                                                              factor=self.args.lr_factor)

    def training_step(self, train_batch):
        self.optimizer.zero_grad()
        res_forward = self.forward(train_batch)
        if self.args.model in ['hi-patch', 'tpatch-gnn', 'mtand']:
            results = self.compute_metrics_and_losses(train_batch,
                                                      res_forward,
                                                      'train')
            results["train_loss"].backward(retain_graph=False)
        elif self.args.model in ('lstm', 'dlinear', 'nbeats'):
            results = self.compute_metrics_and_losses_regular(train_batch,
                                                             res_forward,
                                                             'train')
            results["train_loss"].backward(retain_graph=False)
        else:
            raise Exception(f"Model '{self.args.model}' non riconosciuto.")

        self.optimizer.step()

        # Update lr
        results['learning_rate'] = self.optimizer.param_groups[0]['lr']
        results["train_loss"] = results["train_loss"].detach().cpu().float()
        return results


    def compute_metrics_and_losses(self,
                                   batch,
                                   res_forward, set_):
        results = {}

        mse = compute_error(batch["data_to_predict"],
                            res_forward['pred_y'],
                            mask=batch["mask_predicted_data"],
                            func="MSE",
                            reduce="mean")
        rmse = torch.sqrt(mse)
        loss = mse

        with torch.no_grad():  # mae non serve per backprop
            mae = compute_error(batch["data_to_predict"],
                                res_forward['pred_y'],
                                mask=batch["mask_predicted_data"],
                                func="MAE",
                                reduce="mean")

        # Store the loss and error metrics
        results[f'{set_}_loss'] = loss
        results[f'{set_}_mse'] = mse.item()
        results[f'{set_}_rmse'] = rmse.item()
        results[f'{set_}_mae'] = mae.item()
        return results

    def compute_metrics_and_losses_regular(self,
                                           batch,
                                           res_forward, set_):
        results = {}

        y_true = batch[1]
        y_pred = res_forward['pred_y']

        # Loss per backprop
        mse = torch.mean((y_pred - y_true) ** 2)
        rmse = torch.sqrt(mse)
        loss = mse

        # MAE (no grad)
        with torch.no_grad():
            mae = torch.mean(torch.abs(y_pred - y_true))

        # Store the loss and error metrics
        results[f'{set_}_loss'] = loss
        results[f'{set_}_mse'] = mse.item()
        results[f'{set_}_rmse'] = rmse.item()
        results[f'{set_}_mae'] = mae.item()
        return results

    def validation_step(self, val_batch):
        res_forward = self.forward(val_batch)

        # Compute metrics
        if self.args.model in ['hi-patch', 'tpatch-gnn', 'mtand']:
            results = self.compute_metrics_and_losses(val_batch,
                                                      res_forward,
                                                      'val')
            results["val_loss"] = results["val_loss"].detach().cpu().float()
        elif self.args.model in ('lstm', 'dlinear', 'nbeats'):
            results = self.compute_metrics_and_losses_regular(val_batch,
                                                             res_forward,
                                                             'val')
            results["val_loss"] = results["val_loss"].detach().cpu().float()
        else:
            raise Exception(f"Model '{self.args.model}' non riconosciuto.")
        return results

    def test_step(self, test_batch,):
        res_forward = self.forward(test_batch)

        # Compute losses and optimize DGM
        if self.args.model in ['hi-patch', 'tpatch-gnn', 'mtand']:
            results = self.compute_metrics_and_losses(test_batch,
                                                      res_forward,
                                                      'test')
            results["test_loss"] = results["test_loss"].detach().cpu().float()
        elif self.args.model in ('lstm', 'dlinear', 'nbeats'):
            results = self.compute_metrics_and_losses_regular(test_batch,
                                                             res_forward,
                                                             'test')
            results["test_loss"] = results["test_loss"].detach().cpu().float()
        else:
            raise Exception(f"Model '{self.args.model}' non riconosciuto.")
        return results

    def on_validation_epoch_end(self, val_metrics):
        actual_loss = val_metrics['val_mse']
        actual_rmse = val_metrics['val_rmse']
        actual_mae = val_metrics['val_mae']
        # actual_mape = val_metrics['val_mape']
        if actual_loss < self.best_mse:
            self.best_mse = actual_loss
            self.best_rmse = actual_rmse
            self.best_mae = actual_mae
            # self.best_mape = actual_mape