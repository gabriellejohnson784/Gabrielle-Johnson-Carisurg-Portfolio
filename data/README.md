## Data Folder

This folder is currently empty as I am just in week 2 of my project and haven't yet received the dataset. However below is a dataset containing raw emergency department triage variables that was used for data cleaning and analysis in week 0.

data/EmergencyTriageDataset_Reduced_Dirty.csv

When working in Google Colab, the dataset may also be stored in Google Drive instead of this local data/ folder. In that case, update the file path in the notebook to match the dataset location in Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
FILE_PATH = '/content/drive/MyDrive/EmergencyTriageDataset_Reduced_Dirty.csv' 
```



