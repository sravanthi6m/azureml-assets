# import required libraries
import json

from azure.ai.ml import MLClient
from azure.ai.ml import command, Input
from azure.identity import DefaultAzureCredential

from azure.ai.ml import command, Input, Output, load_component

# import required libraries
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Enter details of your AML workspace
subscription_id = "381b38e9-9840-4719-a5a0-61d9585e1e91"
resource_group = "automlimage_eastus2_rg"
workspace = "risha-ws-eastus2"
compute_cluster = "gpu-cluster-nc6s"

# connect to the workspace
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

json_dict = {
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
    }

with open("automl_settings.json", "w") as outfile:
    json.dump(json_dict, outfile)

# Define image pre-processing component
pre_process_job = command(
    code="../../components/src/common/",
    command="python preprocess.py \
        --training_data ${{inputs.training_data}} \
        --validation_data ${{inputs.validation_data}} \
        --automl_settings ${{inputs.automl_settings}}",
    environment="azureml:AzureML-AutoML-DNN-Vision-GPU:98",
    inputs={
        "training_data": Input(
            type="mltable",
            path="azureml://datastores/workspaceblobstore/paths/vision-od/train/",
        ),
        "validation_data": Input(
            type="mltable",
            path="azureml://datastores/workspaceblobstore/paths/vision-od/valid/",
        ),
        "automl_settings": Input(
            type="uri_file",
            path="./automl_settings.json",
        ),
    },
    outputs={
        "output_folder": Output(
            path="azureml://datastores/workspaceblobstore/paths/azureml",
        ),
    },
    compute=compute_cluster,
)

# Load the image object detection component
train_od_component_func = load_component(source="../../components/object_detection/spec.yaml")

train_model = train_od_component_func(
    training_data=pre_process_job.inputs.training_data,
    model_name='yolov5',
    learning_rate=Choice([0.01, 0.001]),
)

"""

score_component_func = load_component(source="./predict.yml")

# define a pipeline
@pipeline()
def pipeline_with_hyperparameter_sweep():
    train_model = train_component_func(
        data=Input(
            type="uri_file",
            path="wasbs://datasets@azuremlexamples.blob.core.windows.net/iris.csv",
        ),
        c_value=Uniform(min_value=0.5, max_value=0.9),
        kernel=Choice(["rbf", "linear", "poly"]),
        coef0=Uniform(min_value=0.1, max_value=1),
        degree=3,
        gamma="scale",
        shrinking=False,
        probability=False,
        tol=0.001,
        cache_size=1024,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=42,
    )
    sweep_step = train_model.sweep(
        primary_metric="training_f1_score",
        goal="minimize",
        sampling_algorithm="random",
        compute="cpu-cluster",
    )
    sweep_step.set_limits(max_total_trials=20, max_concurrent_trials=10, timeout=7200)

    score_data = score_component_func(
        model=sweep_step.outputs.model_output, test_data=sweep_step.outputs.test_data
    )


pipeline_job = pipeline_with_hyperparameter_sweep()

# set pipeline level compute
pipeline_job.settings.default_compute = "cpu-cluster"

"""

#returned_job = ml_client.jobs.create_or_update(pre_process_job)