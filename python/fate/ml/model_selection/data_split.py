#
#  Copyright 2023 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import logging

from fate.interface import Context, Dataframe
from ..abc.module import Module

logger = logging.getLogger(__name__)


class DataSplitModuleGuest(Module):
    def __init__(
            self,
            train_size=0.8,
            validate_size=0.2,
            test_size=0.0,
            stratified=False,
            random_state=None,
            ctx_mode="hetero"
    ):
        self.train_size = train_size
        self.validate_size = validate_size
        self.test_size = test_size
        self.stratified = stratified
        self.random_state = random_state
        self.ctx_mode = ctx_mode

    def fit(self, ctx: Context, train_data, validate_data=None):
        data_count = train_data.shape[0]
        train_size, validate_size, test_size = get_split_data_size(self.train_size,
                                                                   self.validate_size,
                                                                   self.test_size,
                                                                   data_count)

        if self.stratified:
            train_data_set = sample_per_label(train_data, label_count=train_size, random_state=self.random_state)
        else:
            train_data_set = train_data.sample(n=train_size, replace=False)
        train_sid = train_data_set.get_indexer(target="sample_id")
        validate_test_data_set = train_data.drop(train_sid)
        validate_data_set = validate_test_data_set.sample(n=validate_size, replace=False)
        validate_sid = validate_data_set.get_indexer(target="sample_id")
        test_data_set = validate_test_data_set.drop(validate_sid)

        if self.ctx_mode == "hetero":
            ctx.hosts.put("train_data_sid", train_data_set.get_indexer(target="sample_id"))
            ctx.hosts.put("validate_data_sid", validate_data_set.get_indexer(target="sample_id"))
            ctx.hosts.put("test_data_sid", test_data_set.get_indexer(target="sample_id"))

        return train_data_set, validate_data_set, test_data_set


class DataSplitModuleHost(Module):
    def __init__(
            self,
            train_size=0.8,
            validate_size=0.2,
            test_size=0.0,
            stratified=False,
            random_state=None,
            ctx_mode="hetero"
    ):
        self.train_size = train_size
        self.validate_size = validate_size
        self.test_size = test_size
        self.stratified = stratified
        self.random_state = random_state
        self.ctx_mode = ctx_mode

    def fit(self, ctx: Context, train_data, validate_data=None):
        if self.ctx_mode == "hetero":
            train_data_sid = ctx.guest.get("train_data_sid")
            validate_data_sid = ctx.guest.get("validate_data_sid")
            test_data_sid = ctx.guest.get("test_data_sid")
            train_data_set = train_data.loc(train_data_sid, preserve_order=True)
            validate_data_set = train_data.loc(validate_data_sid, preserve_order=True)
            test_data_set = train_data.loc(test_data_sid, preserve_order=True)
        elif self.ctx_mode in ["homo", "local"]:
            data_count = train_data.shape[0]
            train_size, validate_size, test_size = get_split_data_size(self.train_size,
                                                                       self.validate_size,
                                                                       self.test_size,
                                                                       data_count)

            if self.stratified:
                train_data_set = sample_per_label(train_data, label_count=train_size, random_state=self.random_state)
            else:
                train_data_set = train_data.sample(n=train_size, replace=False)
            train_sid = train_data_set.get_indexer(target="sample_id")
            validate_test_data_set = train_data.drop(train_sid)
            validate_data_set = validate_test_data_set.sample(n=validate_size, replace=False)
            validate_sid = validate_data_set.get_indexer(target="sample_id")
            test_data_set = validate_test_data_set.drop(validate_sid)
        else:
            raise ValueError(f"Unknown ctx_mode: {self.ctx_mode}")
        return train_data_set, validate_data_set, test_data_set


def sample_per_label(train_data, label_count=None, random_state=None):
    labels = train_data[train_data.schema.label_name].unique()
    sampled_data_df = []
    for label in labels:
        label_data = train_data[train_data[train_data.schema.label_name] == label]
        label_sampled_data = label_data.sample(n=label_count, replace=False, random_state=random_state)
        sampled_data_df.append(label_sampled_data)
    sampled_data = Dataframe.vstack(sampled_data_df)
    return sampled_data


def get_split_data_size(train_size, validate_size, test_size, data_count):
    """
    Validate & transform param inputs into all int
    """
    # check & transform data set sizes
    if isinstance(test_size, float) or isinstance(train_size, float) or isinstance(validate_size, float):
        total_size = 1.0
    else:
        total_size = data_count
    if train_size is None:
        if validate_size is None:
            train_size = total_size - test_size
            validate_size = total_size - (test_size + train_size)
        else:
            if test_size is None:
                test_size = 0
            train_size = total_size - (validate_size + test_size)
    elif test_size is None:
        if validate_size is None:
            test_size = total_size - train_size
            validate_size = total_size - (test_size + train_size)
        else:
            test_size = total_size - (validate_size + train_size)
    elif validate_size is None:
        if train_size is None:
            train_size = total_size - test_size
        validate_size = total_size - (test_size + train_size)

    if abs((abs(train_size) + abs(test_size) + abs(validate_size)) - total_size) > 1e-6:
        raise ValueError(f"train_size, test_size, validate_size should sum up to 1.0 or data count")

    if isinstance(train_size, float):
        train_size = round(train_size * data_count)
        validate_size = round(validate_size * data_count)
        test_size = total_size - train_size - validate_size
    return train_size, validate_size, test_size
