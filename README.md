## Machine Learning Project Template

| | |
| --- | --- |
| **Description** | Template for a machine learning project |
| **Author** | [Moreno La Quatra](https://mlaquatra.me) |
| **Course** | [Machine Learning @ UniKore](https://unikore.it) |
| **License** | [MIT](https://opensource.org/licenses/MIT) |

---

### Table of Contents

- [Machine Learning Project Template](#machine-learning-project-template)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Requirements](#requirements)
  - [Code structure](#code-structure)
  - [License](#license)

---

### Introduction

This template is primarly intended for the students of the Machine Learning course at UniKore. It provides a basic structure for a machine learning project, including a `README.md` file, a `requirements.txt` file, and a structure for the code.

The template covers a simple use-case of [MNIST](http://yann.lecun.com/exdb/mnist/) dataset classification using a simple Feedforward Neural Network.

Students are asked to implement the assignment following the structure provided in this template. The results **must be** reproducible using **only** the code provided in the template. The code must be well-documented and easy to read.

The project is divided into two main scripts:
- `train.py` for training the model.
- `test.py` for testing the model.

The dataset is managed by the `mnist_dataset.py` class, while the model is defined in the `ff_model.py` class.

> [!IMPORTANT]  
> The code provided in this template is a simple **minimal** example. Students are encouraged to follow the structure provided in this template and expand it with models and considerations for the personal assignment.


The main idea is that, the project can be reproduced by running the following commands:

```bash
git clone [student-repository-url]
cd [student-repository]
bash prepare.sh
python train.py
python test.py
```

The `prepare.sh` script is used to install the requirements for the project and, optionally, to set up the environment (e.g., download the dataset, etc.). The project should be self-contained and reproducible by running the above commands. The code will be evaluated on [Google Colab](https://colab.research.google.com/), so it is recommended to test the code on that platform.

> [!CAUTION]
> Use the `.gitignore` file to exclude unnecessary files from the repository. For example, the dataset should not be included in the repository. The `.gitignore` file should exclude the pertinent files and directories.

---

### Requirements

The project is based on **Python 3.11** - one of the latest versions of Python at the time of writing. A few considerations:
- It is recommended to use a virtual environment to manage the dependencies of the project. For example [conda](https://docs.conda.io/en/latest/).
- The requirements are listed in the `requirements.txt` file and can be installed using `pip install -r requirements.txt`.

You may want to modify the requirements file to remove unnecessary dependencies or add new ones. This template is based on the following libraries:
- `torch` for PyTorch.
- `torchvision` for PyTorch vision.
- `yaml_config_override` for configuration management.
- `addict` for configuration management.
- `tqdm` for progress bars.

---

### Code structure

The code is structured as follows:

```
main_repository/
│
├── data_classes/
│   ├── mnist_dataset.py
│
├── model_classes/
│   ├── ff_model.py
│
├── ...
│
├── train.py
├── test.py
└── ...
```

- `data_classes/` contains the classes for managing the dataset.
- `model_classes/` contains the classes for the model design.
- `train.py` is the script for training the model.
- `test.py` is the script for testing the model.
- `prepare.sh` is a script for setting up the environment - at the moment it only installs the requirements.
- `requirements.txt` contains the list of dependencies for the project.
- `README.md` is the file you are currently reading.

---

### License

This project is licensed under the terms of the MIT license. You can find the full license in the `LICENSE` file.


