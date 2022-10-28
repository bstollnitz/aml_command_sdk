# How to train and deploy in Azure ML, using the Python SDK

This project shows how to train a Fashion MNIST model with an Azure ML job, and how to deploy it using an online managed endpoint. It uses the Azure ML Python SDK API, and MLflow for tracking and model representation.


## Blog post

To learn more about the code in this repo, check out the accompanying blog post: https://bea.stollnitz.com/blog/aml-command/


## Azure setup

* You need to have an Azure subscription. You can get a [free subscription](https://azure.microsoft.com/en-us/free?WT.mc_id=aiml-67316-bstollnitz) to try it out.
* Create a [resource group](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal?WT.mc_id=aiml-67316-bstollnitz).
* Create a new machine learning workspace by following the "Create the workspace" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources?WT.mc_id=aiml-67316-bstollnitz). Keep in mind that you'll be creating a "machine learning workspace" Azure resource, not a "workspace" Azure resource, which is entirely different!
* If you have access to GitHub Codespaces, click on the "Code" button in this GitHub repo, select the "Codespaces" tab, and then click on "New codespace."
* Alternatively, if you plan to use your local machine:
  * Install the Azure CLI by following the instructions in the [documentation](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?WT.mc_id=aiml-67316-bstollnitz).
  * Install the ML extension to the Azure CLI by following the "Installation" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?WT.mc_id=aiml-67316-bstollnitz).
* In a terminal window, login to Azure by executing `az login --use-device-code`. 
* Add a `config.json` file to the root of your project (or somewhere in the parent folder hierarchy) containing your Azure subscription ID, resource group, and workspace:
```
{
    "subscription_id": "<YOUR_SUBSCRIPTION_ID>",
    "resource_group": "<YOUR_RESOURCE_GROUP>",
    "workspace_name": "<YOUR_WORKSPACE>"
}
```
* You can now open the [Azure Machine Learning studio](https://ml.azure.com/?WT.mc_id=aiml-67316-bstollnitz), where you'll be able to see and manage all the machine learning resources we'll be creating.
* Although not essential to run the code in this post, I highly recommend installing the [Azure Machine Learning extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai), and logging in to it. This will allow you to see the resources you create in the Azure Machine Learning studio from within VS Code.



## Project setup

If you have access to GitHub Codespaces, click on the "Code" button in this GitHub repo, select the "Codespaces" tab, and then click on "New codespace." Once your codespace is ready, log into Azure by typing the following command in the terminal:

```
az login --use-device-code
```

If you're not using Codespaces, you can set up your local machine with the right conda environment using the following steps.

Install conda environment:

```
conda env create -f environment.yml
```

Activate conda environment:

```
conda activate aml_command_sdk
```


## Train and predict locally

* Under "Run and Debug" on VS Code's left navigation, choose the "Train locally" run configuration and press F5.
* You can analyze the metrics logged in the "mlruns" directory with the following command:

```
mlflow ui
```

* Make a local prediction using the trained mlflow model. You can use either csv or json files:

```
cd aml_command_sdk
mlflow models predict --model-uri "model" --input-path "test_data/images.csv" --content-type csv
mlflow models predict --model-uri "model" --input-path "test_data/images.json" --content-type json
```


## Train and deploy in the cloud


### Create and run the job, which outputs a model

Select the run configuration "Train in the cloud" and press F5 to train in the cloud.


### Create and invoke the endpoint for the model

Select the run configuration "Create endpoint" and press F5 to create an endpoint in the cloud and invoke it.


### Clean up the endpoint

Once you're done working with the endpoint, you can clean it up to avoid getting charged by selecting the "Delete endpoint" run configuration and pressing F5.


## Related resources
* [YAML schema for Command Job](https://docs.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-command?WT.mc_id=aiml-67316-bstollnitz)
* [az ml job commands](https://docs.microsoft.com/en-us/cli/azure/ml/job?view=azure-cli-latest#az-ml-job-create?WT.mc_id=aiml-67316-bstollnitz)
* [Output formats for Azure CLI commands](https://docs.microsoft.com/en-us/cli/azure/format-output-azure-cli?WT.mc_id=aiml-67316-bstollnitz)
