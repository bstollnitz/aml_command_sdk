"""Generates images for inference."""

import logging
import os
from pathlib import Path

import numpy as np
import pandas
from PIL import Image
from torchvision import datasets

DATA_DIR = "aml_command_sdk/test_data/data"
IMAGES_DIR = "aml_command_sdk/test_data/images"
TEST_DATA_DIR = "aml_command_sdk/test_data"


def generate_images(num_images: int) -> None:
    """
    Generates images for inference.
    """
    test_data = datasets.FashionMNIST(DATA_DIR, train=False, download=True)

    images_dir = Path(IMAGES_DIR)
    if images_dir.exists():
        for f in Path(IMAGES_DIR).iterdir():
            f.unlink()
    else:
        os.makedirs(IMAGES_DIR)

    for i, (image, _) in enumerate(test_data):
        if i == num_images:
            break
        image.save(f"{IMAGES_DIR}/image_{i+1:0>3}.png")


def generate_csv_from_images() -> None:
    """
    Generates CSV file from the images.
    """
    delimiter = ","
    fmt = "%.6f"
    image_paths = [f for f in Path(IMAGES_DIR).iterdir() if Path.is_file(f)]
    image_paths.sort()

    X = np.empty((0, 0))
    for (i, image_path) in enumerate(image_paths):
        with Image.open(image_path) as image:
            if len(X) == 0:
                size = image.height * image.width
                X = np.empty((len(image_paths), size))
            x = np.asarray(image).reshape((-1)) / 255.0
            X[i, :] = x

    header = delimiter.join([f"col_{i}" for i in range(X.shape[1])])
    np.savetxt(fname=Path(TEST_DATA_DIR, "images.csv"),
               X=X,
               delimiter=delimiter,
               fmt=fmt,
               header=header,
               comments="")


def get_dataframe_from_images() -> pandas.DataFrame:
    """
    Returns a pandas.DataFrame object that contains the images.
    """
    image_paths = [f for f in Path(IMAGES_DIR).iterdir() if Path.is_file(f)]
    image_paths.sort()

    df = pandas.DataFrame()
    for (i, image_path) in enumerate(image_paths):
        with Image.open(image_path) as image:
            x = np.asarray(image).reshape((1, -1)) / 255.0

            column_names = [f"col_{i}" for i in range(x.shape[1])]
            indices = [i]
            new_row_df = pandas.DataFrame(data=x,
                                          index=indices,
                                          columns=column_names)
            df = pandas.concat(objs=[df, new_row_df])

    return df


def generate_json_from_images() -> None:
    """
    Generates a json file from the images.
    """
    df = get_dataframe_from_images()

    data_json = df.to_json(orient="split")
    with open(Path(TEST_DATA_DIR, "images.json"), "wt",
              encoding="utf-8") as file:
        file.write(data_json)


def generate_json_for_azureml_from_images() -> None:
    """
    Generates a json file from the images, to be used when invoking the Azure ML
    endpoint.
    """
    df = get_dataframe_from_images()

    # pylint: disable=inconsistent-quotes
    data_json = '{"input_data":' + df.to_json(orient="split") + '}'
    with open(Path(TEST_DATA_DIR, "images_azureml.json"),
              "wt",
              encoding="utf-8") as file:
        file.write(data_json)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    generate_images(2)
    generate_csv_from_images()
    generate_json_from_images()
    generate_json_for_azureml_from_images()


if __name__ == "__main__":
    main()
