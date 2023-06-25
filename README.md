# monkeypox-disease-prediction

## problem/dataset:

Addressing the problem related to the recent outbreak of Monkeypox disease. Provided the different type of symptoms of a patient, to predict if the person has the Monkeypox disease or not.
For this problem, the dataset was taken from:

Dataset link: https://www.kaggle.com/datasets/muhammad4hmed/monkeypox-patients-dataset


## Context of the Problem:
### 2022 monkeypox outbreak in the United States:
Outbreak in US, is part of the larger outbreak of human monkeypox caused by the West African clade of the monkeypox virus. The United States was the fourth country, outside of the African countries with endemic monkeypox, to experience an outbreak in 2022. The first case was documented in Boston, Massachusetts, on May 19, 2022. As of August 22, monkeypox has spread to all 50 states in the United States, as well as Washington, D.C., and Puerto Rico. The United States had the highest number of monkeypox cases in the world. California had the highest number of monkeypox cases in the United States.


## Dealing with the problem:


The data contains a total of 25,000 rows with 9 features. Each row contains unique data belonging to each patient, their encountered symptoms and corresponding disease possibility (Positive or Negative).
Each row contains unique data belonging to each patient, their encountered symptoms and corresponding disease possibility (Positive or Negative).

I used different machine learning models to apply over this data.
1. Perceptron
2. SVM
3. Logistic Regression 
4. KNN
5. Decision Trees
6. Neural Networks

Compared their performance with different evaluation criteria. 

## Steps to run the project: 
1. Project code and results can be viewed directly without running the project(refer to the link in the end) but if you want to actually run the project, follow the below steps.
2. Find the python notebook and DATA.csv file in this repository.
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


## For read-only view of project:
Open this link : https://colab.research.google.com/drive/1zzuy5bQPLUZTUsMlKuvffNFvtDASQ7p9?usp=sharing

## Results and analysis: 

### 1. Perceptron:
* Accuracy for this model is found out to be almost 60%.
* Hyperparameter: Learning rate
* Tuning results: Smaller learning rate gave better performance but took more
time to train.

<img width="450" alt="image" src="https://github.com/tantuwaySourabh/monkeypox-disease-prediction/assets/26655938/bdbe921e-1368-4702-99aa-fecab0c73c1d">

### 2. Support Vector Machines: 
  1)	Linear: Accuracy: ~70%
  2)	Kernel SVM(Polynomial): Accuracy: ~69%
  3)	Kernel SVM(Sigmoid) : Accuracy: ~56%
  4)	Kernel SVM(Gaussian) : Accuracy: ~68%
  5)	After hyperparameter tuning:  For regularization parameter C = 10 best accuracy is achieved for polynomial SVM almost 70%.

  <img width="449" alt="image" src="https://github.com/tantuwaySourabh/monkeypox-disease-prediction/assets/26655938/7851a845-3801-4852-ad74-50b48191e03a">

### 3. K-Nearest Neighbors: 
  1) For K = 5, accuracy is nearly 65%
  2) After Hyperparameter tuning for K:
     * Larger values of K are giving good prediction.
     * A range of K values (after 15) are good choice for our model.
     * Optimal value is 33(Accuracy 0.6738)
    
     <img width="518" alt="image" src="https://github.com/tantuwaySourabh/monkeypox-disease-prediction/assets/26655938/86956fee-ca28-475d-80a2-7761d8db0798">

     


### 4. Logistic Regression:
  For “liblinear” solver, accuracy found to be nearly 69%
  After Hyperparameter tuning for C and using different solver: 
  * For lower value of C (e.g., 0.01) model is performing best.
  * Accuracy for this value reached almost 70%
  * Means high regularized model are better for any type of solver.

    <img width="531" alt="image" src="https://github.com/tantuwaySourabh/monkeypox-disease-prediction/assets/26655938/4c049d96-bb13-427c-a603-76e73278f5aa">


### 5. Decision Trees: 
* For max depth of 5 and min leaf samples = 5, accuracy found to be nearly 68%.
* entropy criterion is used here.
After Hyperparameter tuning for depth and number of leaf samples: 
  * For depth = 5 and min leaf samples = 1, accuracy incremented up to 69%
  * <img width="311" alt="image" src="https://github.com/tantuwaySourabh/monkeypox-disease-prediction/assets/26655938/d1968284-e388-40b7-b7d1-26f1b4617bb0">

### 6. Neural Network: 
* Around 70% accuracy is achieved for baseline model:
* There are many hyperparameters involved in the neural networks. Like number of neurons, activation units per layer, number of layers, optimizer type etc.
  
  <img width="570" alt="image" src="https://github.com/tantuwaySourabh/monkeypox-disease-prediction/assets/26655938/717c7546-ea8b-476b-8a87-bf4ddbde41e8">


## Summary: 
As we have seen that performance of Linear SVM, Logistic Regression, and Neural Networks is found to be better.
Also, hyperparameters play an important role in improving the efficacy of the algorithms. 
So, it is very important to study different types of hyperparameters involved in ML algorithms. And how they affect the ML models.

## References: 
*	https://machinelearningmastery.com/perceptron-algorithm-for-classification-in-python/
*	https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
*	https://towardsdatascience.com/how-to-find-the-optimal-value-of-k-in-knn-35d936e554eb
* https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
*	https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
*	https://www.kaggle.com/code/gauravduttakiit/hyperparameter-tuning-in-decision-trees
*	https://plainenglish.io/blog/hyperparameter-tuning-of-decision-tree-classifier-using-gridsearchcv-2a6ebcaffeda
*	https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
*	https://www.analyticsvidhya.com/blog/2021/05/tuning-the-hyperparameters-and-layers-of-neural-network-deep-learning/


