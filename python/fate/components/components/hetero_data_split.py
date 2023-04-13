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

from typing import Union

from fate.components import (
    GUEST,
    HOST,
    DatasetArtifact,
    Input,
    DataSplitMetrics,
    Output,
    Role,
    cpn,
    params,
)


@cpn.component(roles=[GUEST, HOST])
def hetero_data_split(ctx, role):
    ...


@hetero_data_split.train()
@cpn.artifact("train_data", type=Input[DatasetArtifact], roles=[GUEST, HOST], desc="training data")
@cpn.parameter("train_size", type=Union[params.conint(ge=0), params.confloat(ge=0.0)], default=None,
               desc="size of output training data, should be either int for exact sample size or float for fraction")
@cpn.parameter("validate_size", type=Union[params.conint(ge=0), params.confloat(ge=0.0)], default=None,
               desc="size of output validation data, should be either int for exact sample size or float for fraction")
@cpn.parameter("test_size", type=Union[params.conint(ge=0), params.confloat(ge=0.0)], default=None,
               desc="size of output test data, should be either int for exact sample size or float for fraction")
@cpn.parameter(
    "stratified", type=bool, default=False,
    desc="whether sample with stratification, should not use this for data with continuous label values"
)
@cpn.parameter("random_state", type=params.conint(ge=0), default=None,
               desc="random state")
@cpn.artifact("train_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
@cpn.artifact("validate_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
@cpn.artifact("test_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
@cpn.artifact("train_output_metric", type=Output[DataSplitMetrics], roles=[GUEST, HOST])
def train(
        ctx,
        role: Role,
        train_data,
        train_size,
        validate_size,
        test_size,
        stratified,
        random_state,
        train_output_data,
        validate_output_data,
        test_output_data,
        train_output_metric,
):
    if isinstance(train_size, float) or isinstance(validate_size, float) or isinstance(test_size, float):
        if train_size + validate_size + test_size > 1:
            raise ValueError("(train_size + validate_size + test_size) should be less than or equal to 1.0")
    if train_size is None and validate_size is None and test_size is None:
        train_size = 0.8
        validate_size = 0.2
        test_size = 0.0

    data_split_train(
        ctx, role, train_data, train_output_data, validate_output_data, test_output_data, train_output_metric,
        train_size, validate_size, test_size, stratified, random_state
    )


def data_split_train(ctx, role, train_data, train_output_data, validate_output_data, test_output_data,
                     train_output_metric, train_size, validate_size, test_size, stratified, random_state):
    ctx.metrics.handler.register_metrics(sample_summary=ctx.writer(train_output_metric))

    from fate.ml.model_selection.data_split import DataSplitModuleGuest, DataSplitModuleHost

    with ctx.sub_ctx("train") as sub_ctx:
        if role.is_guest:
            module = DataSplitModuleGuest(train_size, validate_size, test_size,
                                          stratified, random_state, ctx_mode="hetero")
        elif role.is_host:
            module = DataSplitModuleHost(train_size, validate_size, test_size,
                                         stratified, random_state, ctx_mode="hetero")
        train_data = sub_ctx.reader(train_data).read_dataframe()

        train_data_set, validate_data_set, test_data_set = module.fit(sub_ctx, train_data)
        # train_data_set, validate_data_set, test_data_set = module.split_data(train_data)
        sub_ctx.writer(train_output_data).write_dataframe(train_data_set)
        sub_ctx.writer(validate_output_data).write_dataframe(validate_data_set)
        sub_ctx.writer(test_output_data).write_dataframe(test_data_set)
