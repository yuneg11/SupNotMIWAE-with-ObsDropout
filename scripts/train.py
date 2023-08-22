import sys
sys.path.append(".")

import os
import random
import logging
import inspect
import warnings
from itertools import chain

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from typing import TYPE_CHECKING
if TYPE_CHECKING:  # This is a hack to make VS Code intellisense work
    # from tensorflow.python import keras
    from keras.api._v2 import keras
else:
    keras = tf.keras

try:
    import wandb
    is_wandb_enabled = True
except ImportError:
    is_wandb_enabled = False

import yaml
import nxcl
from nxcl.config import load_config, save_config, add_config_arguments, ConfigDict
from nxcl.experimental import utils
from nxcl.rich import Progress

from missing import models, data, callbacks, metrics


def generate_random_seed():
    rand_bytes = os.urandom(4)
    return int.from_bytes(rand_bytes, byteorder="little", signed=False)


def link_output_dir(output_dir: str, link_dir: str):
    os.makedirs(os.path.dirname(link_dir), exist_ok=True)
    relpath = os.path.relpath(output_dir, os.path.dirname(link_dir))
    os.symlink(relpath, link_dir)


def build_callbacks(config, output_dir, valid_iterator):
    cb_list = []

    # Evaluation callback
    # Repeat epochs + 1 times as we run an additional validation step at the end of training afer recovering the model.
    metric_dict = {k: eval(v) for k, v in config.dataset.metrics.items()}
    cb_list.append(callbacks.EvaluationCallback(valid_iterator, "Valid/", metrics=metric_dict))

    # Early stopping callback
    if "monitor_quantity2" in config.train:
        monitors = [f"Valid/{config.train.monitor_quantity}",
                    f"Valid/{config.train.monitor_quantity2}"]
    else:
        monitors = [f"Valid/{config.train.monitor_quantity}"]

    cb_list.append(callbacks.CustomEarlyStopping(
        monitors,
        mode=config.train.direction_of_improvement,
        patience=config.train.early_stopping,
        min_delta=0.0001,
    ))

    # LR scheduling
    cb_list.append(keras.callbacks.ReduceLROnPlateau(
        monitor=f"Valid/{config.train.monitor_quantity}",
        factor=0.5,
        patience=(config.train.early_stopping // 2),
        # patience=2,
        verbose=1,
        mode=config.train.direction_of_improvement,
        min_delta=0.0001,
        cooldown=(config.train.early_stopping // 2),
        # cooldown=2,
        min_lr=1e-5,
    ))
    cb_list.append(callbacks.WarmUpScheduler(
        config.train.learning_rate,
        warmup_steps=config.train.warmup_steps,
        verbose=1,
    ))

    # Logging
    cb_list.append(keras.callbacks.CSVLogger(os.path.join(output_dir, "metrics.csv")))
    cb_list.append(keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, "weights.h5"),
        save_best_only=True,
        save_weights_only=True,
        monitor=f"Valid/{config.train.monitor_quantity}",
        mode=config.train.direction_of_improvement,
    ))

    # Progress
    cb_list.append(callbacks.ProgressCallback(verbose=1, metric_prefix=["Train/", "Valid/"]))

    if is_wandb_enabled:
        cb_list.append(callbacks.SimpleWandbCallback(metric_prefix=["Train/", "Valid/"]))

    return cb_list


def build_imputation_callbacks(config, output_dir, valid_iterator):
    cb_list = []

    # Evaluation callback
    # Repeat epochs + 1 times as we run an additional validation step at the end of training afer recovering the model.
    cb_list.append(callbacks.EvaluationCallback_Imputation(valid_iterator, "Valid/"))

    monitors = [f"val_{config.train.monitor_quantity}"]

    cb_list.append(callbacks.CustomEarlyStopping(
        monitors,
        mode=config.train.direction_of_improvement,
        patience=config.train.early_stopping,
        min_delta=0.0001,
    ))

    cb_list.append(callbacks.WarmUpScheduler(
        config.train.learning_rate,
        warmup_steps=config.train.warmup_steps,
        verbose=1,
    ))

    # Logging
    cb_list.append(keras.callbacks.CSVLogger(os.path.join(output_dir, "metrics.csv")))
    cb_list.append(keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, "weights.h5"),
        save_best_only=True,
        save_weights_only=True,
        monitor=f"val_{config.train.monitor_quantity}",
        mode=config.train.direction_of_improvement,
    ))

    # Progress
    cb_list.append(callbacks.ProgressCallback(verbose=1, metric_prefix=["Train/", "Valid/"]))

    if is_wandb_enabled:
        cb_list.append(callbacks.SimpleWandbCallback(metric_prefix=["Train/", "Valid/"]))

    return cb_list


def main(config, output_dir):
    # Random seed
    if config.train.get("seed", None) is None:
        config.train.seed = generate_random_seed()

    if config.test.get("seed", None) is None:
        config.test.seed = generate_random_seed()

    random.seed(int(config.train.seed))
    np.random.seed(int(config.train.seed))
    os.environ["PYTHONHASHSEED"] = str(config.train.seed)
    tf.random.set_seed(int(config.train.seed))

    # Create model
    if not hasattr(models, config.model.name):
        raise ValueError(f"Unknown model: {config.model.name}")

    config.model.setdefault("output_activation", config.dataset.output_activation)
    config.model.setdefault("output_dims", config.dataset.output_dims)

    model_class = getattr(models, config.model.name)
    model_argnames = inspect.signature(model_class).parameters.keys()
    model: keras.Model = model_class(**{k: v for k, v in config.model.items() if k in model_argnames})
    model_config = model.get_config()

    for k in model_argnames:
        if k not in config.model:
            warnings.warn(f"Using default model argument: {k} = {model_config[k]}")

    # Update config with default values
    config.model.update(model_config)
    save_config(config, os.path.join(output_dir, "config.yaml"))
    if is_wandb_enabled:
        wandb.config.update(config.to_dict(), allow_val_change=True)

    # Create dataloader

    dataset_stat = data.get_dataset_statistics(config.dataset.name)
    class_balance = ([dataset_stat.class_balance[str(i)] for i in range(2)] if config.dataset.balance else None)
    class_weights = config.dataset.get("class_weights")
    if class_weights is not None:
        class_weights = {int(k[1:]): v for k, v in class_weights.items()}

    normalize_fn = data.get_dataset_normalize_fn(config.dataset.name)
    model_preprocess_fn = getattr(model, "data_preprocessing_fn", lambda: None)()

    train_preprocess_fn = data.build_preprocess_fn(normalize_fn, model_preprocess_fn, class_weights=class_weights)
    valid_preprocess_fn = data.build_preprocess_fn(normalize_fn, model_preprocess_fn, class_weights=None)

    train_iterator, train_steps = data.build_train_iterator(
        dataset_name=config.dataset.name,
        epochs=config.train.max_epochs,
        batch_size=config.train.batch_size,
        preprocess_fn=train_preprocess_fn,
        balance=config.dataset.balance,
        class_balance=class_balance,
        data_dir=config.dataset.get("data_dir"),
    )

    valid_iterator, valid_steps = data.build_valid_iterator(
        dataset_name=config.dataset.name,
        batch_size=config.test.batch_size,
        preprocess_fn=valid_preprocess_fn,
        data_dir=config.dataset.get("data_dir"),
    )

    x, y = next(iter(valid_iterator))
    args_keys = inspect.signature(model_class.call).parameters.keys()

    model(
        x,
        **({"training": True} if "training" in args_keys else {}),
        **({"output": y} if "output" in args_keys else {}),
    )

    logging.info("")
    model.summary(print_fn=logging.info, show_trainable=True, expand_nested=True, line_length=70)
    logging.info("")


    # add pretrain type
    if (not ('train_type' in config.model)) or (config.model.train_type =='joint'):
        eager_run=False

        sample_weight_mode = (None if config.dataset.get("class_weights") is None else "temporal")
        if not ('optimizer' in config.train) or config.train.optimizer == 'Adam':
            optimizer = keras.optimizers.Adam(learning_rate=config.train.learning_rate)
        elif config.train.optimizer == 'AdamW':
            # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            #     config.train.learning_rate,
            #     decay_steps=100000,
            #     decay_rate=0.96,
            #     staircase=True)
            # optimizer = keras.optimizers.experimental.AdamW(learning_rate=lr_schedule, weight_decay=config.train.weight_decay)

            optimizer = keras.optimizers.experimental.AdamW(learning_rate=config.train.learning_rate, weight_decay=config.train.weight_decay)
        train_cbs = build_callbacks(config, output_dir, valid_iterator)

        # with strategy.scope():
        model.compile(
            optimizer=optimizer,
            run_eagerly=config.model.get("run_eagerly", False), # should use true for gp-vae
            # run_eagerly=True,
            loss=config.dataset.loss,
            metrics=["accuracy"],
            sample_weight_mode=sample_weight_mode,
        )

        model.fit(
            train_iterator,
            epochs=config.train.max_epochs,
            callbacks=train_cbs,
            steps_per_epoch=train_steps,
            # NOTE: Pass a iterator over dataset with repeat, otherwise the cache is reset after each epoch.
            #       This has the disadvantage that we need to also pass validation_steps.
            validation_data=valid_iterator,
            validation_steps=valid_steps,
            verbose=0,
        )

    elif config.model.train_type == 'imputation':
        eager_run=False

        sample_weight_mode = (None if config.dataset.get("class_weights") is None else "temporal")
        optimizer = keras.optimizers.Adam(learning_rate=config.train.learning_rate)
        train_pretrain_cbs = build_imputation_callbacks(config, output_dir, valid_iterator)

        pretrain_epoch= config.train.pretrain_epoch

        model.compile(
            optimizer=optimizer,
            run_eagerly=config.model.get("run_eagerly", False), # True for GPVAE
            loss=config.dataset.loss,
            metrics=["accuracy"],
            sample_weight_mode=sample_weight_mode,
        )

        model.fit(
            train_iterator,
            callbacks=train_pretrain_cbs,
            epochs=pretrain_epoch,                                 # 다른 메트릭으로 얼리스타핑하는거 짜기   callbacks=train_cbs,
            steps_per_epoch=train_steps,
            # NOTE: Pass a iterator over dataset with repeat, otherwise the cache is reset after each epoch.
            #       This has the disadvantage that we need to also pass validation_steps.
            validation_data=valid_iterator,
            validation_steps=valid_steps,
            verbose=0,
        )

    else:
        raise ValueError(f"Unknown train_type: {config.model.train_type}")

    ensemble_size = config.test.ensemble_size
    model.load_weights(os.path.join(output_dir, "weights.h5"))

    random.seed(int(config.test.seed))
    np.random.seed(int(config.test.seed))
    os.environ["PYTHONHASHSEED"] = str(config.test.seed)
    tf.random.set_seed(int(config.test.seed))

    def select_data(data, labels):
        return data

    def select_labels(data, labels):
        return labels

    test_iterator, _ = data.build_test_iterator(
        dataset_name=config.dataset.name,
        batch_size=config.test.batch_size,
        preprocess_fn=valid_preprocess_fn,
        data_dir=config.dataset.get("data_dir"),
    )
    label_batches = list(tfds.as_numpy(test_iterator.map(select_labels)))

    if label_batches[0].ndim == 3:
        # Online prediction scenario
        def remove_padding(label_batch):
            labels = []
            for instance in label_batch:
                is_padding = np.all((instance == -100), axis=-1)
                labels.append(instance[~is_padding])
            return labels
        labels = list(chain.from_iterable([remove_padding(label_batch) for label_batch in label_batches])) # list of (50,11) array
        # label_idx = list(chain.from_iterable([remove_padding(label_batch)[1] for label_batch in label_batches]))
        batch_predictions = []

        for batch in tfds.as_numpy(test_iterator.map(select_data)):
            prediction = np.array([model.predict_on_batch(batch) for _ in range(ensemble_size)])
            # print(prediction.shape)
            predictions = np.mean(prediction, axis=0).astype("float64") # (32,221,11) for every time
            batch_predictions.append(predictions)

        predictions = list(chain.from_iterable(batch_predictions)) #np.array(list(chain.from_iterable(batch_predictions)))
        predictions = [prediction[:len(label)] for prediction, label in zip(predictions, labels)]  # Split off invalid predictions

    else:
        with Progress(speed_estimate_period=300) as p:
            # Whole time series classification scenario
            labels = np.concatenate(label_batches, axis=0)
            prediction = np.array([
                model.predict(test_iterator, verbose=0) for _ in p.trange(ensemble_size, description="Test")
            ])
            predictions = np.mean(prediction, axis=0).astype("float64")

    test_metrics = {}

    for metric_name, metric_fn_str in config.dataset.metrics.items():
        score = eval(metric_fn_str)(labels, predictions)
        test_metrics[f"Test/{metric_name}"] = round(score.item(), 4)
        logging.info(f"Test/{metric_name}: {score:.4f}")

    with open(os.path.join(output_dir, "test_metric.yaml"), "w") as f:
        yaml.dump(test_metrics, f, sort_keys=False)

    if is_wandb_enabled:
        for k, v in test_metrics.items():
            wandb.summary[k] = v

    logging.info("Finished")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config-file", type=str, required=True)
    args, rest_args = parser.parse_known_args()

    config: ConfigDict = load_config(args.config_file)
    # You can add alias to config here.
    # ex) In command line, you can use "-e 100" or "--max-epochs 100" instead of "--train.max_epochs 100".
    #     But the config key "train.max_epochs" should exist in the config file (at least with a dummy value).

    parser = argparse.ArgumentParser()
    add_config_arguments(parser, config, aliases={
        # General
        "model.name":               ["-m",  "--model"],
        "train.seed":               ["-s",  "--seed"],
        "train.max_epochs":         ["-e",  "--max-epochs"],
        "train.batch_size":         ["-bs", "--batch-size"],
        "train.learning_rate":      ["-lr", "--learning-rate"],
        "dataset.name":             ["-d",  "--dataset"],
        "dataset.balance":          ["--balance"],
        "train.early_stopping":     ["--early-stopping"],
        # Model specific
        ## SupNotMIWAEModel
        "model.recurrent_dropout":  ["-rd", "--recurrent-dropout"],
        "model.observe_dropout":    ["-od", "--observe-dropout"],
        "model.impute_type":        ["-it",   "--impute-type"],
        "model.n_train_latents":    ["-ntrl", "--n-train-latents"],
        "model.n_train_samples":    ["-ntrs", "--n-train-samples"],
        "model.n_test_latents":     ["-ntsl", "--n-test-latents"],
        "model.n_test_samples":     ["-ntss", "--n-test-samples"],
    })
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("-o", "--output-dir", "--log_dir", type=str, default=None)
    parser.add_argument("-name", "--wandb-name", default=argparse.SUPPRESS, dest="wandb.name")
    parser.add_argument("-group", "--wandb-group", default=argparse.SUPPRESS, dest="wandb.group")
    parser.add_argument("-tags", "--wandb-tags", default=argparse.SUPPRESS, dest="wandb.tags", nargs="*")
    parser.add_argument("-notes", "--wandb-notes", default=argparse.SUPPRESS, dest="wandb.notes")
    args = parser.parse_args(rest_args)

    link_dir = args.output_dir
    dev = args.dev
    del args.output_dir
    del args.dev

    config.update(vars(args))

    # Setup logger
    log_name = utils.get_experiment_name()

    if dev:
        log_name = "dev-" + log_name
        is_wandb_enabled = False

    base_dir = "outs" if link_dir is None else link_dir.split("/")[0]
    output_dir = os.path.join(base_dir, "_", log_name)

    if link_dir is None:
        link_dir = os.path.join(base_dir, config.dataset.name, config.model.name, os.path.basename(output_dir))
    else:
        link_dir = os.path.join(link_dir, os.path.basename(output_dir))

    os.makedirs(output_dir, exist_ok=True)
    link_output_dir(output_dir, link_dir)

    save_config(config, os.path.join(output_dir, "config.yaml"))

    root_logger = utils.setup_logger(None, output_dir, suppress=[tf, nxcl])
    root_logger.removeHandler(root_logger.handlers[0])  # Remove the automatically added stream handler

    keras.utils.disable_interactive_logging()

    # Log configurations
    logging.debug("python " + " ".join(sys.argv))
    logging.info("Configs:")
    for k, v in config.items(flatten=True):
        logging.info(f"    {k:<25}: {v}")
    logging.info(f"Output directory: \"{output_dir}\"" + (f", \"{link_dir}\"" if link_dir else ""))

    # Wandb
    if is_wandb_enabled:
        wandb.init(
            name=config.get("wandb.name", f"{config.model.name} - {log_name.split('-')[-1]}"),
            group=config.get("wandb.group", f"{config.model.name}"),
            tags=config.get("wandb.tags", [config.dataset.name, config.model.name]),
            notes=config.get("wandb.notes", None),
        )
        wandb.config.update(config.to_dict())
        wandb.define_metric("epoch", hidden=True)

        for split in ("Train", "Valid", "Test"):
            wandb.define_metric(f"{split}/*", step_metric="epoch")
            wandb.define_metric(f"{split}/auprc",    summary="max")
            wandb.define_metric(f"{split}/auroc",    summary="max")
            wandb.define_metric(f"{split}/brier",    summary="min")
            wandb.define_metric(f"{split}/ece",      summary="min")
            wandb.define_metric(f"{split}/logloss",  summary="min")
            wandb.define_metric(f"{split}/accuracy", summary="max")

    try:
        main(config, output_dir)
        exit_code = 0
    except KeyboardInterrupt:
        logging.info("Interrupted")
        exit_code = 1
    except Exception as e:
        logging.exception(e)
        exit_code = 2

    if is_wandb_enabled:
        wandb.finish(exit_code=exit_code)
