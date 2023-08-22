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


import math
import pickle
import functools

from collections.abc import Sequence

import tensorflow as tf
import tensorflow_datasets as tfds
import medical_ts_datasets

from absl import logging as absl_logging


# TODO: Change deprecated functions to the new ones
get_output_shapes = tf.compat.v1.data.get_output_shapes
get_output_types = tf.compat.v1.data.get_output_types


def disable_absl_logging(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        absl_logging.set_verbosity(absl_logging.ERROR)
        outs = func(*args, **kwargs)
        absl_logging.set_verbosity(absl_logging.DEBUG)
        return outs
    return wrapper


def map_to_zero(dtypes):
    if isinstance(dtypes, Sequence):
        return tuple((map_to_zero(d) for d in dtypes))
    return tf.cast(0., dtypes)


def map_to_label_padding(dtypes, label_padding):
    if isinstance(dtypes, Sequence):
        return tuple((map_to_zero(d) for d in dtypes))
    return tf.cast(label_padding, dtypes)


def get_padding_values(input_dataset_types, label_padding=-100):
    """Get a tensor of padding values fitting input_dataset_types.

    Here we pad everything with 0. and the labels with `label_padding`. This
    allows us to be able to recognize them later during the evaluation, even
    when the values have already been padded into batches.

    Args:
        tensor_shapes: Nested structure of tensor shapes.

    Returns:
        Nested structure of padding values where all are 0 except teh one
        corresponding to tensor_shapes[1], which is padded according to the
        `label_padding` value.

    """

    if len(input_dataset_types) == 2:
        data_type, label_type = input_dataset_types
        return (
            map_to_zero(data_type),
            map_to_label_padding(label_type, label_padding)
        )

    if len(input_dataset_types) == 3:
        data_type, label_type, sample_weight_type = input_dataset_types
        return (
            map_to_zero(data_type),
            map_to_label_padding(label_type, label_padding),
            map_to_zero(sample_weight_type)
        )


def positive_instances(*args):
    if len(args) == 2:
        data, label = args
    elif len(args) == 3:
        data, label, sample_weights = args
    else:
        raise ValueError("Wrong number of arguments")
    return tf.math.equal(tf.reduce_max(label), 1)


def negative_instances(*args):
    if len(args) == 2:
        data, label = args
    elif len(args) == 3:
        data, label, sample_weights = args
    else:
        raise ValueError("Wrong number of arguments")
    return tf.math.equal(tf.reduce_max(label), 0)


UHA_SPLIT_DICT = {
    tfds.Split.TRAIN:      slice(   0, 4588),
    tfds.Split.VALIDATION: slice(4588, 5243),
    tfds.Split.TEST:       slice(5243, 6554),
}


class NumExamples:
    def __init__(self, num_examples):
        self.num_examples = num_examples


class UHADatasetInfo:
    splits = {
        tfds.Split.TRAIN:      NumExamples(4588),
        tfds.Split.VALIDATION: NumExamples(655),
        tfds.Split.TEST:       NumExamples(1311),
    }


def load_uci_human_activity(data_dir, split, with_info=False):
    if split not in (tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST):
        raise ValueError("Invalid split")
    else:
        slc = UHA_SPLIT_DICT[split]

    with open(f"{data_dir}/uci_data.pickle", "rb") as f:
        data = pickle.load(f)

    split_examples = (
        data["demo"][slc].astype("float32"),
        data["times"][slc].astype("float32"),
        data["values"][slc].astype("float32"),
        data["measurements"][slc].astype("bool"),
        data["lengths"][slc].astype("float32"),
    )
    split_labels = data["labels"][slc]

    dataset = tf.data.Dataset.from_tensor_slices((split_examples, split_labels))
    dataset_info = UHADatasetInfo()

    if with_info:
        return dataset, dataset_info
    else:
        return dataset


@disable_absl_logging
def build_train_iterator(
    dataset_name, epochs, batch_size, data_dir=None, filter_fn=None, preprocess_fn=None, split=None,
    balance=False, class_balance=None,
):
    # if dataset_name == "uci_human_activity":
    #     dataset, dataset_info = load_uci_human_activity(
    #         data_dir=data_dir,
    #         split=(tfds.Split.TRAIN if split is None else split),
    #         with_info=True,
    #     )
    # else:
    dataset, dataset_info = tfds.load(
        dataset_name,
        data_dir=data_dir,
        split=(tfds.Split.TRAIN if split is None else split),
        as_supervised=True,
        with_info=True,
    )

    if filter_fn is not None:
        dataset = dataset.filter(filter_fn)

    if filter_fn is None and split is None:
        num_samples = dataset_info.splits[tfds.Split.TRAIN].num_examples
    else:
        num_samples = len(list(dataset.as_numpy_iterator()))

    steps_per_epoch = int(math.floor(num_samples / batch_size))

    if preprocess_fn is not None:
        dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if balance:
        majority_class = max(range(len(class_balance)), key=(lambda i: class_balance[i]))
        minority_class = min(range(len(class_balance)), key=(lambda i: class_balance[i]))

        num_majority = class_balance[majority_class] * num_samples
        num_minority = class_balance[minority_class] * num_samples

        # Generate two separate datasets using filter
        pos_data = dataset \
            .filter(positive_instances) \
            .shuffle(int(class_balance[1] * num_samples), reshuffle_each_iteration=True) \
            .repeat()
        neg_data = dataset \
            .filter(negative_instances) \
            .shuffle(int(class_balance[0] * num_samples), reshuffle_each_iteration=True) \
            .repeat()

        # And sample from them
        dataset = tf.data.Dataset.sample_from_datasets([pos_data, neg_data], weights=[0.5, 0.5])

        # One epoch should at least contain all negative examples or max
        # each instance of the minority class 3 times
        steps_per_epoch = min(
            math.ceil(    2 * num_majority / batch_size),
            math.ceil(3 * 2 * num_minority / batch_size),
        )

    else:
        # Shuffle repeat and batch
        dataset = dataset \
            .shuffle(num_samples, reshuffle_each_iteration=True) \
            .repeat(epochs)

    batched_dataset = dataset.padded_batch(
        batch_size,
        get_output_shapes(dataset),
        padding_values=get_padding_values(get_output_types(dataset)),
        drop_remainder=True,
    ).prefetch(tf.data.AUTOTUNE)

    return batched_dataset, steps_per_epoch


@disable_absl_logging
def build_valid_iterator(dataset_name, batch_size, data_dir=None, filter_fn=None, preprocess_fn=None, split=None):
    # if dataset_name == "uci_human_activity":
    #     dataset, dataset_info = load_uci_human_activity(
    #         data_dir=data_dir,
    #         split=(tfds.Split.VALIDATION if split is None else split),
    #         with_info=True,
    #     )
    # else:
    dataset, dataset_info = tfds.load(
        dataset_name,
        data_dir=data_dir,
        split=(tfds.Split.VALIDATION if split is None else split),
        as_supervised=True,
        with_info=True,
    )

    if filter_fn is not None:
        dataset = dataset.filter(filter_fn)

    if filter_fn is None and split is None:
        num_samples = dataset_info.splits[tfds.Split.VALIDATION].num_examples
    else:
        num_samples = len(list(dataset.as_numpy_iterator()))

    steps_per_epoch = int(math.ceil(num_samples / batch_size))

    if preprocess_fn is not None:
        dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

    batched_dataset = dataset.padded_batch(
        batch_size,
        get_output_shapes(dataset),
        padding_values=get_padding_values(get_output_types(dataset)),
        drop_remainder=False,
    )

    return batched_dataset, steps_per_epoch


@disable_absl_logging
def build_test_iterator(dataset_name, batch_size, data_dir=None, filter_fn=None, preprocess_fn=None, split=None):
    # if dataset_name == "uci_human_activity":
    #     dataset, dataset_info = load_uci_human_activity(
    #         data_dir=data_dir,
    #         split=(tfds.Split.TEST if split is None else split),
    #         with_info=True,
    #     )
    # else:
    dataset, dataset_info = tfds.load(
        dataset_name,
        data_dir=data_dir,
        split=(tfds.Split.TEST if split is None else split),
        as_supervised=True,
        with_info=True,
    )

    if filter_fn is not None:
        dataset = dataset.filter(filter_fn)

    if filter_fn is None and split is None:
        num_samples = dataset_info.splits[tfds.Split.TEST].num_examples
    else:
        num_samples = len(list(dataset.as_numpy_iterator()))

    steps_per_epoch = int(math.floor(num_samples / batch_size))

    if preprocess_fn is not None:
        dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch
    batched_dataset = dataset.padded_batch(
        batch_size,
        get_output_shapes(dataset),
        padding_values=get_padding_values(get_output_types(dataset)),
        drop_remainder=False,
    )

    return batched_dataset, steps_per_epoch


def build_preprocess_fn(normalize_fn, model_preprocess_fn=None, class_weights=None):
    if model_preprocess_fn is None:
        return normalize_fn

    elif class_weights is None:
        def combined_fn(ts, labels):
            normalized_ts, labels = normalize_fn(ts, labels)
            preprocessed_ts, labels = model_preprocess_fn(normalized_ts, labels)
            return preprocessed_ts, labels
        return combined_fn

    else:
        def combined_fn(ts, labels):
            normalized_ts, labels = normalize_fn(ts, labels)
            preprocessed_ts, labels = model_preprocess_fn(normalized_ts, labels)

            weights = tf.constant([class_weights[i] for i in range(len(class_weights))])
            sample_weights = tf.gather(weights, tf.reshape(labels, (-1, )), axis=0)
            sample_weights = tf.reshape(sample_weights, tf.shape(labels)[:-1])

            return preprocessed_ts, labels, sample_weights
        return combined_fn
