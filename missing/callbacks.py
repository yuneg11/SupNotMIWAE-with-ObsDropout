# Original implementation: https://github.com/BorgwardtLab/Set_Functions_for_Time_Series
#
# Copyright 2020 Max Horn
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AD CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.N


import logging
from itertools import chain
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # Support auto-completion in IDEs.
    from keras.api._v2 import keras
else:
    from tensorflow import keras

import tensorflow_datasets as tfds
from keras.utils import io_utils

from nxcl.rich import Progress

try:
    import wandb
except ImportError:
    pass


class CustomEarlyStopping(keras.callbacks.Callback):
    def __init__(self,
                 monitor='val_loss', # list
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None, # list
                 restore_best_weights=False):
        super(CustomEarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights=None
        self.length = len(monitor)

        if not isinstance(mode, str):
            raise ValueError("mode should be a string")
        if mode not in ['auto', 'min', 'max']:
            logging.warning('EarlyStopping mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'
        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            raise ValueError("Invalid mode")

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def _are_improvement(self, monitor_value, reference_value):
        improvement = []
        for mv, rv in zip(monitor_value, reference_value):
            if mv is None or rv is None:
                improvement.append(None)
                continue
            improved = self.monitor_op(mv-self.min_delta, rv)
            improvement.append(improved)

        return improvement

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = [(np.Inf if self.monitor_op == np.less else -np.Inf) for _ in range(self.length)]
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = [logs.get(m) for m in self.monitor]
        for i, m in enumerate(self.monitor):
            value = logs.get(m)
            if  i==0 and value is None:
                logging.warning('Early stopping conditioned on metric `%s` '
                                'which is not available. Available metrics are: %s',
                                self.monitor, ','.join(list(logs.keys())))
            current.append(value)
        if self.restore_best_weights and self.best_weights is None:
            self.best_weights = self.model.get_weights()

        self.wait += 1
        improvement = self._are_improvement(current, self.best)
        any_improvement = False
        for i, imp in enumerate(improvement):
            if imp:
                self.best[i] = current[i]
                any_improvement = True
        if any_improvement:
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            if self.baseline is None or any(self._are_improvement(current, self.baseline)):
                self.wait = 0

        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    keras.utils.io_utils.print_msg(
                        'Restoring model weights from the end of the best epoch: '
                        f'{self.best_epoch + 1}.')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            keras.utils.io_utils.print_msg(
                f'Epoch {self.stopped_epoch + 1}: early stopping')



class WarmUpScheduler(keras.callbacks.Callback):
    def __init__(self, final_lr, warmup_learning_rate=0.0, warmup_steps=0, verbose=0):
        """Constructor for warmup learning rate scheduler.

        Args:
            learning_rate_base: base learning rate.
            warmup_learning_rate: Initial learning rate for warm up. (default: 0.0)
            warmup_steps: Number of warmup steps. (default: 0)
            verbose: 0 -> quiet, 1 -> update messages. (default: {0})

        """

        super().__init__()
        self.final_lr = final_lr
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.verbose = verbose

        # Count global steps from 1, allows us to set warmup_steps to zero to
        # skip warmup.
        self.global_step = 1
        self._increase_per_step = self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step += 1
        lr = keras.backend.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.global_step <= self.warmup_steps:
            increase = (self.final_lr - self.warmup_learning_rate) / self.warmup_steps
            new_lr = self.warmup_learning_rate + (increase * self.global_step)
            keras.backend.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose > 0:
                logging.info(f"Warmup - learning rate: {new_lr:.6f}/{self.final_lr:.6f}")


class EvaluationCallback(keras.callbacks.Callback):
    def __init__(self, dataset, prefix, metrics=None):
        """Initialize evaluation callback.

        Args:
            dataset: The dataset, should be a tensorflow dataset outputting tuples of (X, y).
            prefix: Name to prepend metric name in log.
            metrics: Dictionary of metrics {name: function(y_true, pred)}.
            print_evaluations: Print the result of the evaluations.

        """
        super().__init__()

        self.dataset = dataset
        self.prefix = prefix
        self.metrics = metrics or {}

        label_batches = list(tfds.as_numpy(dataset.map(lambda _, labels: labels)))
        if label_batches[0].ndim == 3:
            # Online prediction scenario
            def remove_padding(label_batch):
                # Online prediction scenario
                labels = []
                for instance in label_batch:
                    is_padding = np.all((instance == -100), axis=-1)
                    labels.append(instance[~is_padding])
                return labels

            self.labels = np.concatenate(list(chain.from_iterable([
                remove_padding(label_batch) for label_batch in label_batches
            ])), axis=0)

            self.online = True
        else:
            # Whole time series classification scenario
            self.labels = np.concatenate(label_batches, axis=0)
            self.online = False

        self.data_iterator = dataset

    def on_epoch_end(self, epoch, log={}):
        ensemble_size = 1

        if self.online:
            batch_predictions = []

            for batch in tfds.as_numpy(self.data_iterator.map(lambda features, _: features)):
                prediction = np.array([self.model.predict_on_batch(batch) for _ in range(ensemble_size)])
                predictions = np.mean(prediction, axis=0).astype("float64")
                batch_predictions.append(predictions)

            predictions = np.concatenate(list(chain.from_iterable(batch_predictions)), axis=0)
            predictions = np.array([prediction[:len(label)] for prediction, label in zip(predictions, self.labels)])

        else:
            prediction = np.array([self.model.predict(self.data_iterator) for _ in range(ensemble_size)])
            predictions = np.mean(prediction, 0).astype("float64")

        for metric_name, metric_fn in self.metrics.items():
            score = metric_fn(self.labels, predictions)
            log[f"{self.prefix}{metric_name}"] = score


# class EvaluationCallback(keras.callbacks.Callback):
#     def __init__(self, dataset, prefix, metrics=None):
#         """Initialize evaluation callback.
#         Args:
#             dataset: The dataset, should be a tensorflow dataset outputting tuples of (X, y).
#             prefix: Name to prepend metric name in log.
#             metrics: Dictionary of metrics {name: function(y_true, pred)}.
#             print_evaluations: Print the result of the evaluations.
#         """
#         super().__init__()

#         self.dataset = dataset
#         self.prefix = prefix
#         self.metrics = metrics or {}
#         label_batches = list(tfds.as_numpy(dataset.map(lambda _, labels: labels)))
#         if label_batches[0].ndim == 3:
#             # Online prediction scenario
#             def remove_padding(label_batch):
#                 # Online prediction scenario
#                 labels = []
#                 label_idx = []
#                 for instance in label_batch:
#                     is_padding = np.all((instance == 0.), axis=-1)
#                     labels.append(instance[~is_padding])
#                     label_idx.append(~is_padding)
#                 return labels, label_idx
#             self.labels = list(chain.from_iterable([
#                 remove_padding(label_batch)[0] for label_batch in label_batches
#             ]))
#             self.label_idx = list(chain.from_iterable([
#                 remove_padding(label_batch)[1] for label_batch in label_batches
#             ]))
#             self.online = True
#         else:
#             # Whole time series classification scenario
#             self.labels = np.concatenate(label_batches, axis=0)
#             self.online = False
#         self.data_iterator = dataset

#     def on_epoch_end(self, epoch, log={}):
#         ensemble_size = 1
#         if self.online:
#             batch_predictions = []
#             for batch in tfds.as_numpy(self.data_iterator.map(lambda features, _: features)):
#                 prediction = np.array([self.model.predict_on_batch(batch) for _ in range(ensemble_size)])
#                 predictions = np.mean(prediction, axis=0).astype("float64")
#                 batch_predictions.append(predictions)
#             predictions = chain.from_iterable(batch_predictions)
#             # Split off invalid predictions
#             predictions = [prediction[idx] for prediction, idx in zip(predictions, self.label_idx)]
#         else:
#             # for batch in tfds.as_numpy(self.data_iterator.map(lambda features, _: features)):
#             #      x= self.model.predict_on_batch(batch)
#             #      print(x.shape)
#             #k = self.model.predict(self.data_iterator, verbose=0)
#             #print(k.shape) (1917,1)
            
#             prediction = np.array([self.model.predict(self.data_iterator, verbose=0) for _ in range(ensemble_size)])
#             predictions = np.mean(prediction, 0).astype("float64")
#         for metric_name, metric_fn in self.metrics.items():
#             score = metric_fn(self.labels, predictions)
#             log[f"{self.prefix}{metric_name}"] = score






class EvaluationCallback_Imputation(keras.callbacks.Callback):
    def __init__(self, dataset, prefix, metrics=None):
        """Callback function for imputation experiments
        Args:
            dataset: The dataset, should be a tensorflow dataset outputting tuples of (X, y).
            prefix: Name to prepend metric name in log.
            metrics: Dictionary of metrics {name: function(y_true, pred)}.
            print_evaluations: Print the result of the evaluations.
        """
        super().__init__()

        self.dataset = dataset
        self.prefix = prefix
        self.metrics = metrics or {}
        # true_data = list(tfds.as_numpy(dataset.map(lambda inputs, _: inputs[2]))) # input을 가져와야함
        # self.true_data = true_data
        self.online = False
        self.data_iterator = dataset

    def on_epoch_end(self, epoch, log={}):
        ensemble_size = 1
        # prediction = np.array([self.model.predict(self.data_iterator, verbose=0) for _ in range(ensemble_size)])
        # print(prediction.shape)
        # predictions = np.mean(prediction, 0).astype("float64") 
        # 아래 코드는 여러개의 metric을 계산하는것을 지원하지 않습니다. 그러려면 score를 딕셔너리로 바꿔주세요
        # score=[]
        # for batch in tfds.as_numpy(self.data_iterator.map(lambda features, _: features)):
        #      x_hat =  self.model.predict_on_batch(batch)

        #      for metric_name, metric_fn in self.metrics.items():
        #          score.append(metric_fn(batch[2].numpy(), x_hat))
        # score = np.mean(score)
        # # for metric_name, metric_fn in self.metrics.items():
        # #     score = metric_fn(self.true_data, predictions)
        # for metric_name, metric_fn in self.metrics.items():

        #     log[f"{self.prefix}{metric_name}"] = score







class ProgressCallback(keras.callbacks.Callback):
    def __init__(self, progress: Progress = None, verbose = 1, metric_prefix=None):
        super().__init__()

        self.progress = progress or Progress(speed_estimate_period=300)
        self.train_id = self.progress.add_task("Epoch")
        self.batch_id = self.progress.add_task("Batch")

        self.verbose = verbose

        self.epochs = None
        self.batches = None
        self.epoch_format = None
        if metric_prefix is None:
            self.metric_prefix = [""]
        elif isinstance(metric_prefix, Sequence):
            self.metric_prefix = metric_prefix
        else:
            self.metric_prefix = [metric_prefix]

    def set_params(self, params):
        # self.verbose = params["verbose"]

        if "epochs" in params:
            self.progress.update(self.train_id, total=params["epochs"])
            self.epochs = params["epochs"]
            self.epoch_format = f"{len(str(self.epochs))}d"

        if "steps" in params:
            self.progress.update(self.batch_id, total=params["steps"])
            self.batches = params["steps"]

    def on_train_begin(self, logs=None):
        self.progress.start()

    def on_train_end(self, logs=None):
        self.progress.stop()
        self.progress.remove_task(self.batch_id)

    def on_epoch_begin(self, epoch, logs=None):
        self.progress.reset(self.batch_id)

    def on_epoch_end(self, epoch, logs=None):
        self.progress.advance(self.train_id)

        if self.verbose > 0:
            logs = logs or {}

            epoch_str = f"Epoch {epoch + 1:{self.epoch_format}}"
            epoch_pad = " " * len(epoch_str)

            log_keys = [k for k in logs.keys() if k != "loss" and any([k.startswith(p) for p in self.metric_prefix])]
            max_key_len = max([len(k) for k in log_keys]) + 1

            io_utils.print_msg(
                f"{epoch_str} | Train/Loss: {logs['loss']:9.5f} LR: {keras.backend.get_value(self.model.optimizer.lr):.4e}"
            )

            for k1, k2 in zip(log_keys[:-1:2], log_keys[1::2]):
                io_utils.print_msg(
                    f"{epoch_pad} | {f'{k1}:':<{max_key_len}} {logs[k1]:8.4f}   {f'{k2}:':<{max_key_len}} {logs[k2]:8.4f}"
                )

            if len(log_keys) % 2 == 1:
                io_utils.print_msg(
                    f"{epoch_pad} | {f'{log_keys[-1]}:':<{max_key_len}}: {logs[log_keys[-1]]:8.4f}"
                )

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        self.progress.advance(self.batch_id)


class SimpleWandbCallback(keras.callbacks.Callback):
    def __init__(self, metric_prefix=None):
        super().__init__()

        if metric_prefix is None:
            self.metric_prefix = [""]
        elif isinstance(metric_prefix, Sequence):
            self.metric_prefix = metric_prefix
        else:
            self.metric_prefix = [metric_prefix]

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        log_keys = [k for k in logs.keys() if k != "loss" and any([k.startswith(p) for p in self.metric_prefix])]

        if logs.get("pred_error") is not None:
            wandb.log({
                "epoch": epoch + 1,
                "Train/Loss": round(logs["loss"], 4),
                "Train/PredtionError": round(logs["pred_error"], 4),
                "Train/ReconstructionError": round(logs["recon_error"], 4),
                "Train/Regularization": round(logs["regular"], 4),
                "Train/ESS": round(logs["avg_ess"], 2),
                "Train/QuizMSE": round(logs["quiz_mse"], 4),
                "Train/AwkwardMSE": round(logs["awkward"], 4),
                **{k: round(logs[k], 4) for k in log_keys},
            })
        else:
            wandb.log({
                "epoch": epoch + 1,
                "Train/Loss": round(logs["loss"], 4),
                **{k: round(logs[k], 4) for k in log_keys},
            })
