"""Utility functions for MIMIC-III benchmarking datasets."""
import tensorflow as tf
import pandas as pd
import numpy as np

class activity_TSDreader:
    """Reader base class for MIMIC-III benchmarks."""

    demographics = ['None']
    # values = ['0','1','2','3','4','5','6','7','8','9','10']
    values = []
    # Phoneme spectra - 11
    #LSST - 6
    for i in range(12):
        values.extend([str(i)])

    def __init__(self, dataset_dir, listfile, blacklist=None):

        self.dataset_dir = dataset_dir
        with tf.io.gfile.GFile(listfile, 'r') as f:
            self.instances = pd.read_csv(f, header=0, sep=',')


    def _read_data_for_instance(self, filename):
        """Read a single instance from file.
        Args:
            filename: Filename from which to read data.
        """
        with tf.io.gfile.GFile(filename, 'r') as f:
            data = pd.read_csv(f, header=0, sep=',')
        time = data['Hours']

        demographics = (
            data[self.demographics]
            .replace({-1: float('NaN')})
            .mean()
        )
        demographics = demographics.fillna(value=-1)
        values = data[self.values]
        return time,demographics, values


    def __len__(self):
        """Get number of instances that can be read."""
        return len(self.instances)