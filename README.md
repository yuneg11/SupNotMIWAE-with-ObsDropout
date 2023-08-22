# Probabilistic Imputation for Time-series Classification with Missing Data

**Currently, we are refactoring the code. We will update README and the refactored code soon. This repository is not ready for use yet.**

This repository contains the implementation for _Probabilistic Imputation for Time-series Classification with Missing Data_ (ICML 2023).

SeungHyun Kim, Hyunsu Kim, [EungGu Yun](https://github.com/yuneg11), Hwangrae Lee, Jaehun Lee, Juho Lee

[[`Paper`](https://arxiv.org/abs/2308.06738)][[`ICML`](https://icml.cc/virtual/2023/poster/23522)][[`BibTeX`](#citation)]

## Installation

```bash
pip install -r requirements.txt
```

## Datasets

Please see [Set_Functions_for_Time_Series](https://github.com/BorgwardtLab/Set_Functions_for_Time_Series) for the details and the preparation of the datasets for now.
We will add the details of the datasets soon.

## Usage

### Train model

```bash
python scripts/train.py \
    -f, --config-file CONFIG_FILE \
    [-o, --output-dir OUTPUT_DIR] \
    [--dev] \
    [additional options]
```

You can change the values of the parameters in the config file using `--<config-key> <value>` options.
For example, if you want to change the `early_stopping` to 10, you can use `--train.early_stopping 10`.
Similarly, if you want to change the `learning_rate` to 0.1, you can use `--train.learning_rate 0.1` or `-lr 0.1`, because `-lr` is registered as an alias of `--train.learning_rate` in `scripts/train.py`.

**NOTE**: The outputs are actually saved in `outs/_/<date>-<time>-<id>/` directory.
          The `--output-dir` option just makes the link to the directory.

#### See help

```bash
python scripts/train.py -f CONFIG_FILE --help
```

#### Example

- Basic case:

    ```bash
    python scripts/train.py -f configs/physionet2012/SupNotMIWAE.yaml
    ```

  - Use config file `configs/physionet2012/SupNotMIWAE.yaml` to train a SupNotMIWAE model.
  - Save outputs under `outs/physionet2012/SupNotMIWAE/` (default output defined in `scripts/train.py`)

- Advanced case:

    ```bash
    python scripts/train.py \
        -f configs/physionet2012/SupNotMIWAE.yaml \
        -o outs/physionet2012/SupNotMIWAE/test1 \
        -lr 0.005 \
        --model.n_units 64 \
        --dev
    ```

  - Use config file `configs/physionet2012/SupNotMIWAE.yaml` to train a SupNotMIWAE model.
  - Save outputs under `outs/physionet2012/SupNotMIWAE/test1/`.
  - Set learning rate `0.005`.
  - Change the model arguments `n_units` to `64`.
  - Mark this as a development run.

## Acknowledgement

Our code is based on [Set_Functions_for_Time_Series](https://github.com/BorgwardtLab/Set_Functions_for_Time_Series) and includes [medical_ts_datasets](https://github.com/ExpectationMax/medical_ts_datasets) with some modifications.

## License

See [LICENSE](LICENSE).

## Citation

```
@inproceedings{kim2023probabilistic,
    title     = {Probabilistic Imputation for Time-series Classification with Missing Data},
    author    = {Kim, SeungHyun and Kim, Hyunsu and Yun, EungGu and Lee, Hwangrae and Lee, Jaehun and Lee, Juho},
    booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML 2023)},
    year      = {2023},
}
```
