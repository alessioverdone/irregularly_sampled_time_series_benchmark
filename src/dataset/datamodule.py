import sys
from pathlib import Path

from src.dataset.datasets import build_dataloaders
from src.dataset.irregular_datasets import SparsifyConfig, build_irregular_dataloaders
from src.dataset.irregular_datasets_static import build_static_irregular_dataloaders
from src.utils.utils import set_params_wrt_dataset


def get_datamodule(run_params):
    # Parse dataset and initialize model
    if run_params.dataset in run_params.DATASET_REGISTRY:
        if run_params.irregular_time_series:
            if run_params.irregular_time_series_pattern == 'dynamic':
                dataset_cls, csv_path, default_D = run_params.DATASET_REGISTRY[run_params.dataset]

                if not Path(csv_path).exists():
                    print(f"[error] CSV non trovato: {csv_path}. Esegui prima `python download.py`.")
                    sys.exit(1)

                sparsify_cfg = SparsifyConfig(
                    mechanism=run_params.mechanism,
                    sparsity=run_params.sparsity,
                    seed=run_params.mask_seed,
                )
                loaders = build_irregular_dataloaders(
                    dataset_cls,
                    csv_path=csv_path,
                    seq_len=run_params.seq_len,
                    label_len=run_params.label_len,
                    pred_len=run_params.pred_len,
                    sparsify_cfg=sparsify_cfg,
                    batch_size=run_params.batch_size,
                    num_workers=run_params.num_workers,
                )

                train_ld, val_ld, test_ld = loaders["train"], loaders["val"], loaders["test"]
                D = train_ld.dataset.n_features
                print(f"[dataset] {run_params.dataset} D={D}  L_in={run_params.seq_len}  L_pred={run_params.pred_len}")
                print(f"[dataset] sparsity={run_params.sparsity} mechanism='{run_params.mechanism}'")
                print(f"[dataset] batches: train={len(train_ld)} val={len(val_ld)} test={len(test_ld)}")

            elif run_params.irregular_time_series_pattern == 'static':
                # Usa il CSV regolare come sorgente; il dataset irregolare viene
                # generato una sola volta e salvato in data/irregular/<dataset>/<config>/
                dataset_cls, csv_path, default_D = run_params.DATASET_REGISTRY[run_params.dataset]

                if not Path(csv_path).exists():
                    print(f"[error] CSV non trovato: {csv_path}. Esegui prima `python download.py`.")
                    sys.exit(1)

                sparsify_cfg = SparsifyConfig(
                    mechanism=run_params.mechanism,
                    sparsity=run_params.sparsity,
                    seed=run_params.mask_seed,
                )

                split_config = run_params.split_configs[run_params.dataset]
                save_dir     = run_params.get_irr_save_dir()

                loaders = build_static_irregular_dataloaders(
                    csv_path=csv_path,
                    save_dir=save_dir,
                    split_config=split_config,
                    seq_len=run_params.seq_len,
                    pred_len=run_params.pred_len,
                    sparsify_cfg=sparsify_cfg,
                    batch_size=run_params.batch_size,
                    num_workers=run_params.num_workers,
                )

                train_ld, val_ld, test_ld = loaders["train"], loaders["val"], loaders["test"]
                D = train_ld.dataset.n_features
                print(f"[dataset] {run_params.dataset} D={D}  L_in={run_params.seq_len}  L_pred={run_params.pred_len}")
                print(f"[dataset] sparsity={run_params.sparsity} mechanism='{run_params.mechanism}'")
                print(f"[dataset] save_dir={save_dir}")
                print(f"[dataset] batches: train={len(train_ld)} val={len(val_ld)} test={len(test_ld)}")
            else:
                raise ValueError(f'{run_params.irregular_time_series_pattern} not valid name for run_params.irregular_time_series_pattern!')
        else:
            dataset_cls, csv_path, default_D = run_params.DATASET_REGISTRY[run_params.dataset]

            loaders = build_dataloaders(dataset_cls,
                                        csv_path=csv_path,
                                        seq_len=run_params.seq_len,
                                        label_len=run_params.label_len,
                                        pred_len=run_params.pred_len,
                                        features="M",
                                        batch_size=run_params.batch_size,
                                        num_workers=run_params.num_workers,)
            train_ld, val_ld, test_ld = loaders["train"], loaders["val"], loaders["test"]
            D = train_ld.dataset.data_x.shape[1]
            print(f"[dataset] {run_params.dataset} D={D}  L_in={run_params.seq_len}  L_pred={run_params.pred_len}")
            print(f"[dataset] batches: train={len(train_ld)} val={len(val_ld)} test={len(test_ld)}")


    else:
        raise ValueError('Define dataset name correct!')

    run_params = set_params_wrt_dataset(run_params, loaders)

    return loaders, run_params


if __name__ == '__main__':
    from src.utils.utils import setup_seed
    from src.config import initialize_configuration

    # Params
    run_params = initialize_configuration()
    run_params.irregular_time_series = False
    run_params.irregular_time_series_pattern = 'static'
    run_params.dataset = "etth1"

    setup_seed(run_params.seed)
    print('Configuration settled!')

    # Data
    dataModuleInstance, run_params = get_datamodule(run_params)
    print('Data imported!')
