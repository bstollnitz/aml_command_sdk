"""Deletes an endpoint."""

import logging

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from common import ENDPOINT_NAME


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)

    ml_client.online_endpoints.begin_delete(ENDPOINT_NAME)


if __name__ == "__main__":
    main()
