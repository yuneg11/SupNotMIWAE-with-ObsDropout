"""Module containing mortality prediction dataset of MIMIC-III benchmarks."""
from os.path import join

import tensorflow_datasets as tfds

from .activity_reader import activity_TSDreader
from .TSD_util import TsDatasetBuilder, TsDatasetInfo
import pdb
import tensorflow as tf

_CITATION = """
None
"""

_DESCRIPTION = """
None
"""


class TsDataReader(activity_TSDreader):
    """Reader for mortality prediction of the MIMIC-III benchmarks."""

    # Blacklisted instances due to unusually many observations compared to the
    # overall distribution.
    blacklist = [
        # Criterion for exclusion: more than 1000 distinct timepoints
        # In training data
        ]

    def __init__(self, dataset_dir, listfile):
        """Initialize MIMIC-III mortality reader."""
        super().__init__(dataset_dir, listfile, self.blacklist)

    def __getitem__(self, index):
        """Get instance with index."""
        instance = self.instances.iloc[index]
        target = list(map(int, instance['y_true'].split(',')))
        data_file = join(self.dataset_dir, instance['stays'])
        time, demographics, values = \
            self._read_data_for_instance(data_file)
        _id = int(instance['stays'].split('_')[0])
        # pdb.set_trace()
        return instance['stays'], {
            'demographics': demographics,
            'time': time,
            'values': values,
            'targets': {
                'target': target
            },
            'metadata': {
                'id': _id
            }
        }


class activity(TsDatasetBuilder):
    """In hospital mortality task dataset of the MIMIC-III benchmarks."""
    VERSION = tfds.core.Version('1.0.1')
    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    manual_dir should contain the file `activity.tar.gz`\
    """

    has_demographics = True
    has_values = True
    default_target = 'target'

    def _info(self):
        return TsDatasetInfo(
            builder=self,
            targets={
                'target':tfds.features.Tensor(
                    shape=(None,), dtype=tf.int32)
            },
            demographics_names=TsDataReader.demographics,
            values_names=activity_TSDreader.values,
            description=_DESCRIPTION,
            homepage='',
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Return SplitGenerators."""
        data_file = join(
            dl_manager.manual_dir, 'activity.tar.gz')
        extracted_path = dl_manager.extract(data_file)
        train_dir = join(extracted_path, 'train')
        train_listfile = join(extracted_path, 'train_listfile.csv')
        val_dir = join(extracted_path, 'val')
        val_listfile = join(extracted_path, 'val_listfile.csv')
        test_dir = join(extracted_path, 'test')
        test_listfile = join(extracted_path, 'test_listfile.csv')

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    'data_dir': train_dir,
                    'listfile': train_listfile
                }
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    'data_dir': val_dir,
                    'listfile': val_listfile
                }
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    'data_dir': test_dir,
                    'listfile': test_listfile
                }
            ),
        ]

    def _generate_examples(self, data_dir, listfile):
        """Yield examples."""
        reader = TsDataReader(data_dir, listfile)
        for patient_id, instance in reader:
            yield patient_id, instance