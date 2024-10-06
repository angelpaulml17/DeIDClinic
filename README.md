# DeIDClinic

DeIDClinic is a tool designed for de-identifying clinical documents using state-of-the-art NLP models such as ClinicalBERT. This project allows you to train the model, analyze datasets, and deploy a web application for de-identification tasks.

## Table of Contents

1. [Download the Project from GitHub](#1-download-the-project-from-github)
2. [Setup Python Environment](#2-setup-python-environment)
3. [Dataset Preparation](#3-dataset-preparation)
4. [Model Training](#4-model-training)
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

   **NOTE**: The following commands are applicable only on Windows systems.
   
4. **Navigate to the project directory**:

    ```bash
    cd .\DeIDClinic-main\
    ```

5. **Create a virtual environment** for Python 3.7.9 using the following command:

    ```bash
    & "PATH TO PYTHON 3.7 .exe FILE" -m venv myenvpytest
    ```

    Replace `PATH TO PYTHON 3.7 .exe FILE` with the actual path to `python.exe` where Python 3.7 was installed.

6. **Activate the virtual environment** so that the code runs within it:

    ```bash
    .\myenvpytest\Scripts\Activate
    ```

7. **Install all required packages**:

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file contains all the required packages with versions to run the application seamlessly.



## 3. Dataset Preparation

The dataset used in this project is provided by the i2b2/UTHealth initiative and managed under the n2c2 challenges hosted by the Department of Biomedical Informatics at Harvard Medical School. Access to this dataset requires approval due to the sensitive nature of the data.

### 3.1 Accessing the Dataset

1. **Open the [N2C2 NLP portal](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/)**:

    - Create an account and apply for access to the datasets.
    - Once access is granted, download the following datasets from the “2014 De-identification and Heart Disease Risk Factors Challenge” folder:
        - Training Data: PHI Gold Set 1
        - Training Data: PHI Gold Set 2

2. **Combine the datasets**:

    - Copy the contents of `Training Data: PHI Gold Set 2` and paste them into `Training Data: PHI Gold Set 1`.

3. **Copy the consolidated dataset** to the downloaded GitHub folder.


## 4. Model Training

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
    !apt-get update –y
    !apt-get install python3.7 python3.7-venv python3.7-dev -y
    !apt-get install python3-pip –y
    !python3.7 -m ensurepip --upgrade
    !python3.7 -m pip install --upgrade pip
    !python3.7 -m pip install pipenv
    !python3.7 -m pip install httplib2
    !python3.7 -m pipenv install
    !python3.7 -m pipenv run pip install keras==2.6.0
    !pipenv run pip install pandas
    !pipenv run pip install spacy
    !pipenv run pip install medcat
    !pip install scikit-learn
    !pip install nltk
    !pip install tensorflow_hub
    !pip install tensorflow
    !pip install transformers
    !pip install sklearn_crfsuite
    !pipenv install flask
    !pipenv shell
    ```

    

6. **Train the model** using the below command:

    ```bash
    !pipenv run python3.7 train_framework.py --source_type i2b2 --source_location “[PATH TO THE DATASET]" --algorithm NER_ClinicalBERT --do_test yes --save_model yes --epochs 15
    ```

7. **Once the training is complete**, the model will be saved in the `Models` folder with a `.pt` extension. 

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

4. **Copy the provided URL** (e.g., http://127.0.0.1:5000) and paste it into your web browser.

5. **The website will load** and can be used to de-identify documents.





NOTE: The only folders/files changed/added to the older MASK version are 
1. "dataset and website"- where the dataset can be placed and website is built and run on flask servers.
2. "ner_plugins"- "NER_ClinicalBERT.py" file is added
3. "Models"- Once the model is trained, the NER_ClinicalBERT is saved in the Models file
4. In the root folder 4 files were added, "extractcounts.py"(to analyse the entity distribution in the dataset), "name extraction.py"(extract all the names, locations and professions from the dataset to create base dictionaries), requirements.txt (to save all the required packages with their respective versions compatible with Python 3.7.9) and model_performance.py (to create heatmap on the entity level performance of multiple models)


## References

- Paul, A., Shaji, D., Han, L., Del-Pinto, W., & Nenadic, G. (2024). *DeIDClinic: A Multi-Layered Framework for De-identification of Clinical Free-text Data*. arXiv. https://arxiv.org/abs/2410.01648



