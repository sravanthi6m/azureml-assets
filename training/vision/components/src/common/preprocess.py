# imports
import argparse
import json
import os

from typing import Any, Dict

from azureml.automl.core.shared.constants import Tasks

import utils

# define functions
def parse_automl_settings(automl_settings_json: str):

    print(automl_settings_json)

    # Opening JSON file
    with open(automl_settings_json, 'r') as openfile:
        # Reading from json file
        automl_settings = json.load(openfile)

    print(automl_settings)
    return automl_settings


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add input arguments
    parser.add_argument("--training_data", type=str, default="")
    parser.add_argument("--validation_data", type=str, default="")
    parser.add_argument("--automl_settings", type=str)

    # add output arguments
    parser.add_argument("--output_folder", type=str)

    # parse args
    args = parser.parse_args()

    return args

@utils.create_component_telemetry_wrapper(Tasks.IMAGE_OBJECT_DETECTION)
def run():
    args = parse_args()

    # Parse Settings
    automl_settings = parse_automl_settings(args.automl_settings)

    # Process the settings
    # TODO: Add automode here.

    # Write the final hyperdrive setting to json for next component to read
    with open(os.path.join(args.output_folder, "hd_settings.json"), "w") as outfile:
        json.dump(automl_settings, outfile)

# run pre-process
if __name__ == "__main__":
    utils.validate_running_on_gpu_compute()

    # Run the component.
    # (If multiple processes are spawned on the same node, only run the component on one process
    # since AutoML will spawn child processes as appropriate.)
    # if utils.get_local_rank() == 0:
    run()
    
    """
    parse_automl_settings({
        "training_parameters": {
            "model_name": "yolov5",
            "number_of_epochs": 15,
        },
        "search_space": [
            {
                "learning_rate": "Uniform(0.001, 0.01)"
            }
        ],
        "job_limits": {
            "max_trials": 5,
            "max_concurrent_trials": 5
        }
    })
    """