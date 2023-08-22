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


import os
import json
from typing import NamedTuple, Any

import numpy as np
import tensorflow as tf


__all__ = [
    "Statistic",
    "get_dataset_statistics",
    "get_dataset_normalize_fn",
]


STATISTICS_PATH = os.path.join(os.path.dirname(__file__), "statistics")
STATISTICS_MAP = {
    "mimic3_mortality":   "mimic3_mortality_v1.0.1.json",
    "mimic3_phenotyping": "mimic3_phenotyping_v1.0.2.json",
    "physionet2012":      "physionet2012_v1.0.10.json",
    "physionet2019":      "physionet2019_v1.0.3.json",
    "activity": "uci_human_activity.json",
}


class Statistic(NamedTuple):
    demo_means: tf.Tensor
    demo_stds: tf.Tensor
    series_means: tf.Tensor
    series_stds: tf.Tensor
    class_balance: dict


def get_dataset_statistics(dataset):
    with open(os.path.join(STATISTICS_PATH, STATISTICS_MAP[dataset]), "r") as f:
        statistic = json.load(f)

    return Statistic(
        demo_means    = tf.convert_to_tensor(statistic["demo_means"]),
        demo_stds     = tf.convert_to_tensor(statistic["demo_stds"]),
        series_means  = tf.convert_to_tensor(statistic["series_means"]),
        series_stds   = tf.convert_to_tensor(statistic["series_stds"]),
        class_balance = statistic["class_balance"],
    )


def get_dataset_normalize_fn(dataset):
    statistic = get_dataset_statistics(dataset)

    demo_means   = statistic.demo_means
    demo_stds    = statistic.demo_stds
    series_means = statistic.series_means
    series_stds  = statistic.series_stds

    def normalize_demo(demo):
        return (demo - demo_means) / demo_stds

    def normalize_series(series, measurements):
        normalized = (series - series_means) / series_stds
        normalized = tf.where(measurements, normalized, tf.zeros_like(normalized))  # Fill NaNs with zeros
        return normalized

    def normalize_fn(feature, label):
        demo, times, series, measurements, *others = feature
        normalized_feature = (
            normalize_demo(demo),
            times,
            normalize_series(series, measurements),
            measurements,
            *others,
        )
        return normalized_feature, label

    return normalize_fn
