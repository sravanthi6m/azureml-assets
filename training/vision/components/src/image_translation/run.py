# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json

from azureml.automl.core.shared.constants import Tasks
from azureml.automl.dnn.vision.common import utils as vision_utils

from common import utils

class InputSettings:

    def __init__(self,
                 training_data: str,
                 validation_data: str) -> None:
        # Input settings
        self.training_data = training_data
        self.validation_data = validation_data

class LimitSettings:
    def __init__(self,
                 timeout_minutes: int,
                 max_trials: int,
                 max_concurrent_trials: int) -> None:
        # Limit Settings
        self.timeout_minutes = timeout_minutes
        self.max_trials = max_trials
        self.max_concurrent_trials = max_concurrent_trials
    
    def get_dict(self):
        return {"timeout": self.timeout_minutes,
                "maxTotalTrials": self.max_trials,
                "maxConcurrentTrials": self.max_concurrent_trials}

class SweepSettings:
    def __init__(self,
                 sampling_algorithm: str) -> None:
        self.sampling_algorithm = sampling_algorithm
    
    def get_dict(self):
        return {"samplingAlgorithmType": self.sampling_algorithm}


class EarlyTerminationSettings:
    def __init__(self,
                 early_termination_policy: str,
                 delay_evaluation: int,
                 evaluation_interval: int) -> None:
        self.early_termination_policy = early_termination_policy
        self.delay_evaluation = delay_evaluation
        self.evaluation_interval = evaluation_interval
    
    def get_dict(self):
        return {"policyType": self.early_termination_policy,
                "delayEvaluation": self.delay_evaluation,
                "evaluationInterval": self.evaluation_interval}


class SearchSpaceSettings:
    def __init__(self,
                 timeout_minutes: int) -> None:
        # Limit Settings
        self.timeout_minutes = timeout_minutes


# Parse command line args
def create_from_parsing_current_cmd_line_args():
    parser = argparse.ArgumentParser()
    
    # Input settings
    parser.add_argument(vision_utils._make_arg('training_data'), type=str)
    parser.add_argument(vision_utils._make_arg('validation_data'), type=str)

    # Limit Settings
    parser.add_argument(vision_utils._make_arg('timeout_minutes'), type=int)
    parser.add_argument(vision_utils._make_arg('max_trials'), type=int)
    parser.add_argument(vision_utils._make_arg('max_concurrent_trials'), type=int)

    # Sweep Settings
    parser.add_argument(vision_utils._make_arg('sampling_algorithm'), type=str)

    # Early Termination Policy
    parser.add_argument(vision_utils._make_arg('early_termination_policy'), type=str)
    parser.add_argument(vision_utils._make_arg('delay_evaluation'), type=int)
    parser.add_argument(vision_utils._make_arg('evaluation_interval'), type=int)
    
    # Training Parameters

    # Output
    parser.add_argument(vision_utils._make_arg('sweep_json'), type=str)
    args, _ = parser.parse_known_args()
    return args


@utils.create_component_telemetry_wrapper(Tasks.IMAGE_OBJECT_DETECTION)
def run():
    args = create_from_parsing_current_cmd_line_args()
    json_dict = {}

    input_data = InputSettings(args.training_data, args.validation_data)
    json_dict["jobType"] = "sweep"

    limit_settings = LimitSettings(args.timeout_minutes, args.max_trials, args.max_concurrent_trials)
    json_dict["limits"] = limit_settings.get_dict()

    sweep_settings = SweepSettings(args.sampling_algorithm)
    json_dict["samplingAlgorithm"] = sweep_settings.get_dict()

    early_termination_settings = EarlyTerminationSettings(args.early_termination_policy, args.delay_evaluation, args.evaluation_interval)
    json_dict["earlyTermination"] = early_termination_settings.get_dict()

    utils.logger.info("Component settings: {}".format(input_data))
    utils.logger.info("Training Data: {}".format(input_data.training_data))
    utils.logger.info("Timeout Minutes: {}".format(limit_settings.timeout_minutes))
    utils.logger.info("Sampling algo: {}".format(sweep_settings.sampling_algorithm))


    args.sweep_json = json.dumps(json_dict)
    utils.logger.info("Sweep Json: {}".format(args.sweep_json))


if __name__ == "__main__":
    utils.validate_running_on_gpu_compute()

    # Run the component.
    # (If multiple processes are spawned on the same node, only run the component on one process
    # since AutoML will spawn child processes as appropriate.)
    if utils.get_local_rank() == 0:
        run()
