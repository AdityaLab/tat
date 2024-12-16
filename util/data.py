import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

import data_formatters.base
InputTypes = data_formatters.base.InputTypes

def to_one_hot(x, cat_map):
    # Get the shape of the original input
    B, L, _ = x.shape
    num_classes = len(cat_map)

    # Initialize the one-hot encoded array
    one_hot_encoded = np.zeros((B, L, num_classes), dtype=np.float32)

    # Iterate over the batch and length dimensions
    for b in range(B):
        for l in range(L):
            category = x[b, l, 0]  # Get the category value
            if category in cat_map:  # Ensure the category is in the mapping
                idx = cat_map[category]  # Get the index from the mapping
                one_hot_encoded[b, l, idx] = 1  # Set the appropriate index to 1

    return one_hot_encoded


class TimeSeriesDataset(Dataset):
    def __init__(self, xt, xs, horizon, lookback):
        self.xt = xt
        self.xs = xs

        self.horizon = horizon
        self.lookback = lookback
        # slicing
        self.XT, self.Y, self.XF, self.XS = self._create_dataset()

    def _create_dataset(self):
        XT = self.xt[:, :self.lookback, :1]
        Y = self.xt[:, -self.horizon:, :1]
        XF = self.xt [:, :, 1:]
        XS = self.xs
        return XT, Y, XF, XS

    def __len__(self):
        return len(self.XT)

    def __getitem__(self, idx):
        xt, y = self.XT[idx], self.Y[idx]
        xf, xs = self.XF[idx], self.XS[idx]
        return xt, xf, xs, y
    

class RetailDataset(Dataset):
    def __init__(self, xt, horizon, lookback, ):
        self.xt = xt

        self.horizon = horizon
        self.lookback = lookback
        # slicing
        self.XT, self.Y, self.XF, self.XS = self._create_dataset()

    def _create_dataset(self):
        XT = self.xt[:, :self.lookback, :3]
        Y = self.xt[:, -self.horizon:, :1]
        XF = torch.cat((self.xt [:, :, 3:6], self.xt [:, :, -5:]), dim=-1)
        XS = self.xt[:, :, 6:15]
        return XT, Y, XF, XS

    def __len__(self):
        return len(self.XT)

    def __getitem__(self, idx):
        xt, y = self.XT[idx], self.Y[idx]
        xf, xs = self.XF[idx], self.XS[idx]
        return xt, xf, xs, y
        

def TimeSeriesDataLoader(dataset, batchsize, num_workers=10, shuffle=True):
    data_loader = DataLoader(dataset, batch_size=batchsize, num_workers=num_workers, shuffle=shuffle, drop_last=False)
    return data_loader

# From TFT
def get_single_col_by_input_type(input_type, column_definition):
    """Returns name of single column.

    Args:
        input_type: Input type of column to extract
        column_definition: Column definition list for experiment
    """

    l = [tup[0] for tup in column_definition if tup[2] == input_type]

    if len(l) != 1:
        raise ValueError('Invalid number of columns for {}'.format(input_type))

    return l[0]


def extract_cols_from_data_type(data_type, column_definition, excluded_input_types):
    """Extracts the names of columns that correspond to a define data_type.

    Args:
        data_type: DataType of columns to extract.
        column_definition: Column definition to use.
        excluded_input_types: Set of input types to exclude

    Returns:
        List of names for columns with data type specified.
    """
    return [
        tup[0]
        for tup in column_definition
        if tup[1] == data_type and tup[2] not in excluded_input_types
    ]


def batch_sampled_data(data, max_samples, time_steps, input_size, output_size, num_encoder_steps, column_definition):
    """Samples segments into a compatible format.

    Args:
        data: Sources data to sample and batch
        max_samples: Maximum number of samples in batch

    Returns:
        Dictionary of batched data with the maximum samples specified.
    """

    if max_samples < 1:
        raise ValueError('Illegal number of samples specified! samples={}'.format(max_samples))

    id_col = get_single_col_by_type(InputTypes.ID, column_definition)
    time_col = get_single_col_by_type(InputTypes.TIME, column_definition)

    data.sort_values(by=[id_col, time_col], inplace=True)

    print('Getting valid sampling locations.')
    valid_sampling_locations = []
    split_data_map = {}
    for identifier, df in data.groupby(id_col):
        print('Getting locations for {}'.format(identifier))
        num_entries = len(df)
        if num_entries >= time_steps:
            valid_sampling_locations += [
                (identifier, time_steps + i)
                for i in range(num_entries - time_steps + 1)
            ]
        split_data_map[identifier] = df

    inputs = np.zeros((max_samples, time_steps, input_size))
    outputs = np.zeros((max_samples, time_steps, output_size))
    time = np.empty((max_samples, time_steps, 1), dtype=object)
    identifiers = np.empty((max_samples, time_steps, 1), dtype=object)

    if max_samples > 0 and len(valid_sampling_locations) > max_samples:
        print('Extracting {} samples...'.format(max_samples))
        ranges = [
            valid_sampling_locations[i] for i in np.random.choice(
                len(valid_sampling_locations), max_samples, replace=False)
        ]
    else:
        print('Max samples={} exceeds # available segments={}'.format(
            max_samples, len(valid_sampling_locations)))
        ranges = valid_sampling_locations

    id_col = get_single_col_by_type(InputTypes.ID, column_definition)
    time_col = get_single_col_by_type(InputTypes.TIME, column_definition)
    target_col = get_single_col_by_type(InputTypes.TARGET, column_definition)
    input_cols = [
        tup[0]
        for tup in column_definition
        if tup[2] not in {InputTypes.ID, InputTypes.TIME}
    ]

    for i, tup in enumerate(ranges):
        if (i + 1 % 1000) == 0:
            print(i + 1, 'of', max_samples, 'samples done...')
        identifier, start_idx = tup
        sliced = split_data_map[identifier].iloc[start_idx - time_steps:start_idx]
        inputs[i, :, :] = sliced[input_cols]
        outputs[i, :, :] = sliced[[target_col]]
        time[i, :, 0] = sliced[time_col]
        identifiers[i, :, 0] = sliced[id_col]

    sampled_data = {
        'inputs': inputs,
        'outputs': outputs[:, num_encoder_steps:, :],
        'active_entries': np.ones_like(outputs[:, num_encoder_steps:, :]),
        'time': time,
        'identifier': identifiers
    }

    return sampled_data

 
def get_single_col_by_type(input_type, column_definition):
    """Returns name of single column for input type."""
    return get_single_col_by_input_type(input_type, column_definition)



def batch_data(data, time_steps, num_encoder_steps, column_definition):
    """Batches data for training.

    Converts raw dataframe from a 2-D tabular format to a batched 3-D array
    to feed into Keras model.

    Args:
        data: DataFrame to batch

    Returns:
        Batched Numpy array with shape=(?, self.time_steps, self.input_size)
    """

    # Functions.
    def _batch_single_entity(input_data, lag):
      time_steps = len(input_data)
      lags = lag
      x = input_data.values
      if time_steps >= lags:
          return np.stack([x[i:time_steps - (lags - 1) + i, :] for i in range(lags)], axis=1)
      else:
          return None

    id_col = get_single_col_by_type(InputTypes.ID, column_definition)
    time_col = get_single_col_by_type(InputTypes.TIME, column_definition)
    target_col = get_single_col_by_type(InputTypes.TARGET, column_definition)
    input_cols = [
        tup[0]
        for tup in column_definition
        if tup[2] not in {InputTypes.ID, InputTypes.TIME}
    ]

    data_map = {}
    for _, sliced in data.groupby(id_col):
        col_mappings = {
            'identifier': [id_col],
            'time': [time_col],
            'outputs': [target_col],
            'inputs': input_cols
        }

        for k in col_mappings:
            cols = col_mappings[k]
            arr = _batch_single_entity(sliced[cols].copy(), time_steps)

            if k not in data_map:
                data_map[k] = [arr]
            else:
                data_map[k].append(arr)

    # Combine all data
    for k in data_map:
        data_map[k] = np.concatenate(data_map[k], axis=0)

    # Shorten target so we only get decoder steps
    data_map['outputs'] = data_map['outputs'][:, num_encoder_steps:, :]

    active_entries = np.ones_like(data_map['outputs'])
    if 'active_entries' not in data_map:
        data_map['active_entries'] = active_entries
    else:
        data_map['active_entries'].append(active_entries)

    return data_map