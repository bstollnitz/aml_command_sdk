"""Creates and invokes a managed online endpoint."""

import logging
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint
from azure.identity import DefaultAzureCredential

from common import MODEL_NAME, ENDPOINT_NAME

DEPLOYMENT_NAME = "blue"
TEST_DATA_PATH = Path(
    Path(__file__).parent.parent, "test_data", "images_azureml.json")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)

    # Create the managed online endpoint.
    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        auth_mode="key",
    )
    registered_endpoint = ml_client.online_endpoints.begin_create_or_update(
        endpoint).result()

    # Get the latest version of the registered model.
    registered_model = ml_client.models.get(name=MODEL_NAME, label="latest")

    # Create the managed online deployment.
    deployment = ManagedOnlineDeployment(name=DEPLOYMENT_NAME,
                                         endpoint_name=ENDPOINT_NAME,
                                         model=registered_model,
                                         instance_type="Standard_DS4_v2",
                                         instance_count=1)
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    # Set deployment traffic to 100%.
    registered_endpoint.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(
        registered_endpoint).result()

    # Invoke the endpoint.
    result = ml_client.online_endpoints.invoke(endpoint_name=ENDPOINT_NAME,
                                               request_file=TEST_DATA_PATH)
    logging.info(result)


if __name__ == "__main__":
    main()
