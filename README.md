# monkeypox-disease-prediction
I am addressing the problem related to the recent outbreak of Monkeypox disease. Provided the different type of symptoms of a patient, we need to predict if the person has the Monkeypox disease or not.
For this problem, I am taking the dataset from Kaggle website. 
Link for dataset: https://www.kaggle.com/datasets/muhammad4hmed/monkeypox-patients-dataset

I used different machine learning models to apply over this data.
1. Perceptron
2. SVM
3. Logistic Regression 
4. KNN
5. Decision Trees
6. Neural Networks

And compared their performance with different evaluation criteria


## Steps to run the project: 
1. Find the python notebook and DATA.csv file in this repository.
2. Open the python notebook in google colab.
3. Upload the DATA.csv file in any folder of Google drive.
4. Change the code in initial two cells of notebook as : 
```
#for fetching data set from google drive
from google.colab import drive
drive.mount('/content/drive')
```
```
import pandas as pd
mp_data =pd.read_csv("drive/My Drive/<YourFolderContainingCSV>/DATA.csv")
mp_data.head()
```
5. you can run rest of the cells of notebook now.


## For read only view of project:
Open this link : https://colab.research.google.com/drive/1zzuy5bQPLUZTUsMlKuvffNFvtDASQ7p9?usp=sharing
