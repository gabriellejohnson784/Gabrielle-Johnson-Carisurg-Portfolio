
# Emergency Department AI-Assisted Triage System 🏥 📝
### By: Gabrielle Johnson Carisurg AI Healthcare Cohort 2026

This project is an amalgamation and reimagining of an AI-assisted triage system built for a low-resource Caribbean hospital, where emergency departments often carry heavy patient loads with limited staff, limited digital infrastructure, and very little room for delay.

## Table of Contents 📖

- [Purpose](#purpose)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Purpose 🩺

The purpose of this project is to explore whether a small but powerful set of routinely collected patient triage variables can help support accurate urgency categorisation in the emergency department where digital data is extremely limited.

Rather than replacing nurses, this system is designed to act as a support tool, a second set of digital aid that helps reduce the burden on triage staff, improve consistency, and support faster decision-making during busy or high pressure periods.

The goal is to identify the least amount of patient information needed to achieve strong triage performance, with a target of approximately 90% accuracy in urgency categorisation and <5 minutes of processing.

## Installation ⚙️

Clone the repository using the command below:

```bash

git clone https://github.com/gabriellejohnson784/Gabrielle-Johnson-Carisurg-Portfolio.git

```

## Usage 💻📈

The notebooks in this project were completed using Google Colab. Before any data manipulation can begin, the required libraries must be imported and Google Drive must be mounted so the dataset can be accessed.

```python

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from google.colab import drive

drive.mount('/content/drive')

FILE_PATH = '/content/drive/MyDrive/add_your_file_path.csv'

df = pd.read_csv(FILE_PATH)

```


## Contributing 🤝

Pull requests are welcome. Any form of contributions that help make the project more practical for low-resource healthcare settings are appreciated.

## License 🧑‍⚖️

This project is not currently licensed.
