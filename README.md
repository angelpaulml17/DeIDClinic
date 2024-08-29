# DeIDClinic

DeIDClinic is a tool designed for de-identifying clinical documents using state-of-the-art NLP models such as ClinicalBERT. This project allows you to train the model, analyze datasets, and deploy a web application for de-identification tasks.

## Table of Contents

1. [Download the Project from GitHub](#1-download-the-project-from-github)
2. [Setup Python Environment](#2-setup-python-environment)
3. [Dataset Preparation](#3-dataset-preparation)
4. [Model Training and Analysis](#4-model-training-and-analysis)
5. [Running the Website](#5-running-the-website)

## 1. Download the Project from GitHub

1. **Clone the repository** or download the ZIP file from the following GitHub link:
   - [GitHub Repository Link](https://github.com/angelpaulml17/DeIDClinic/tree/main)

2. **Unzip the downloaded file**.

## 2. Setup Python Environment

### 2.1 Python Installation

1. Download and install Python 3.7.9 from the official Python website:
   - [Download Python 3.7.9](https://www.python.org/downloads/release/python-379/)
   - Choose the appropriate installer for your operating system (Windows or macOS).
   - Note the installation path.

2. **Open the project folder** in your preferred code editor (e.g., Visual Studio Code).

3. **Navigate to the project directory**:

    ```bash
    cd .\DeIDClinic-main\
    ```

4. **Create a virtual environment** for Python 3.7.9 using the following command:

    ```bash
    & "PATH TO PYTHON 3.7 .exe FILE" -m venv myenvpytest
    ```

    Replace `PATH TO PYTHON 3.7 .exe FILE` with the actual path to `python.exe` where Python 3.7 was installed.

5. **Activate the virtual environment** so that the code runs within it:

    ```bash
    .\myenvpytest\Scripts\Activate
    ```

6. **Install all required packages**:

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file contains all the required packages with versions to run the application seamlessly.

7. **Verify that all required packages are installed successfully**. You should see a screen similar to Figure 1.

![Figure 1: Successful package installation completion screen](path_to_image)

## 3. Dataset Preparation

The dataset used in this project is provided by the i2b2/UTHealth initiative and managed under the n2c2 challenges hosted by the Department of Biomedical Informatics at Harvard Medical School. Access to this dataset requires approval due to the sensitive nature of the data.

### 3.1 Accessing the Dataset

1. **Open the N2C2 NLP portal**:

    - Create an account and apply for access to the datasets.
    - Once access is granted, download the following datasets from the “2014 De-identification and Heart Disease Risk Factors Challenge” folder:
        - Training Data: PHI Gold Set 1
        - Training Data: PHI Gold Set 2

2. **Combine the datasets**:

    - Copy the contents of `Training Data: PHI Gold Set 2` and paste them into `Training Data: PHI Gold Set 1`.

3. **Copy the consolidated dataset** to the downloaded GitHub folder.

### 3.2 Data Pre-processing

The files in the dataset are in XML format. A pre-processing function extracts only the text within the `<TEXT>` elements of the XML files, converting them into plain text format.

### 3.3 Exploratory Data Analysis (EDA)

1. **Perform EDA on the dataset**:

    - Open the `extractcounts.py` file.
    - Replace the path of the `xml_directory` variable with the path where the dataset is placed.
    - Save the file and run the following command on the terminal:

        ```bash
        python extractcounts.py
        ```

    This script processes XML files to extract and count entities, then aggregates the counts and generates a bar chart comparing the total and unique counts of each entity type. The output should look like Figure 2.

![Figure 2: Entity occurrence distribution](path_to_image)

## 4. Model Training and Analysis

### 4.1 Training on a Local Machine

The model can be trained using the following hyperparameters:

- Batch size: 32
- Learning rate: 3e-5
- Epochs: 15

1. **To train the ClinicalBERT model**, run the following command:

    ```bash
    python train_framework.py --source_type i2b2 --source_location "[PATH TO THE DATASET]" --algorithm NER_ClinicalBERT --do_test yes --save_model yes --epochs 15
    ```

    Replace `"[PATH TO THE DATASET]"` with the actual path to the dataset (Training Data: PHI Gold Set 1).

2. **Monitor the training** process. Once training is complete, the model will be saved.

### 4.2 Training on Google Colab

Alternatively, the model can be trained on Google Colab with GPU for faster training (T4).

1. **Upload the folder** on Google Drive.
2. **Open a new Google Colab Python notebook**.
3. **Mount the drive** using the following command:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

4. **Navigate to the project directory**:

    ```bash
    %cd “[PATH TO PROJECT ON GOOGLE DRIVE]”
    ```

5. **Run the following commands** to create the Python 3.7 virtual environment and install the required packages:

    ```bash
    !apt-get update -y
    !apt-get install python3.7 python3.7-venv python3.7-dev -y
    !apt-get install python3-pip -y
    !python3.7 -m ensurepip --upgrade
    !python3.7 -m pip install --upgrade pip
    !python3.7 -m pip install pipenv
    !pipenv install
    !pipenv run pip install -r requirements.txt
    ```

    Once the environment is set up, you should see a response similar to Figure 3.

![Figure 3: Google Colab Successful Shell Run screen](path_to_image)

6. **Train the model** using the below command:

    ```bash
    !pipenv run python3.7 train_framework.py --source_type i2b2 --source_location “[PATH TO THE DATASET]" --algorithm NER_ClinicalBERT --do_test yes --save_model yes --epochs 15
    ```

7. **Once the training is complete**, the model will be saved in the `Models` folder with a `.pt` extension. A classification report similar to Figure 4 will be visible.

![Figure 4: Classification report of ClinicalBERT](path_to_image)

8. **Download the trained model** and place it under the `Models` folder on the local machine.

## 5. Running the Website

1. **Navigate to the website directory**:

    ```bash
    cd '.\dataset and website\Website copy'
    ```

2. **Run the application**:

    ```bash
    python app.py
    ```

3. **Monitor the terminal output**. Once you see a screen similar to Figure 8, it indicates that the app ran successfully and is hosted locally.

![Figure 8: Website Run Screen](path_to_image)

4. **Copy the provided URL** (e.g., http://127.0.0.1:5000) and paste it into your web browser.

5. **The website will load** and can be used to de-identify documents.
