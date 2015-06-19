---
title: "Estimating the quality of health activities with Ensemble approach"
author: "Oabba Kana"
output: html_document
---

In this report, we develope prediction algorithms for estimating the quality of health activities. Our analysis and prediction performance is evaluated over benchmark **Human Activity Recognition** datasets that can accessed from <a href="http://groupware.les.inf.puc-rio.br/har"> this website </a>.

##Required packages and multicore processing
We utilize *caret* and *gbm* packages to make our coding efficient. For quicker analysis, we utilize multiple cores of our system which required us to load the **doParallel** package.


```
## Loading required package: lattice
## Loading required package: ggplot2
## Loading required package: foreach
## foreach: simple, scalable parallel programming from Revolution Analytics
## Use Revolution R for scalability, fault tolerance and more.
## http://www.revolutionanalytics.com
## Loading required package: iterators
## Loading required package: parallel
```
##Loading the datasets
Firstly, we loaded the training and the testing datasets into R. We had a quick look over the training set since this dataset is used in building models. 

##Preprocessing and identifying useful features

We utilize **summary()** function to get an insight into the variables used in this dataset. Summary of the variables in the training dataset reveals that there are a lot of variables with many missing records. For each variable, we count the number of missing observations and for each row, we count the number of missing variables by using *colSums()* and *rowSums()* function. It was found that there are 67 variables with 19216 missing values i.e., for 67 variables, 98\% of the observations are missing. As is saying, **data is king**, we cannot use these variables for any useful insights. 

In addition to features with missing values, various datasets include variables that cannot differentiate between the classes. We utilize *nearZeroVar()* function to identify such variables. This function returns the index of such variables. 

###Useless Variables
We identify indexes of variables with 98\% missing observations with *which()* function. We combine the indexes of such variables wtih near-zero-variability and identify them as indexes of **useless variables** that cannot be productive in the prediction task. 

##Creating new datasets
We remove the useless variables, identified from the training dataset from both the training as well as the testing dataset. Note that we also remove the first variable since it is just an observation index that always increases. These new datasets have the total number of variables reduced to 57, down from a total of 159 variables.



###Why not PCA?
In our new dataset, we found that though the variables are correlated, their relation with respect to the classes is nonlinear. Therefore, it is not a good idea to apply *PCA* to further reduce the dimensionality since it is a linear transformation. As an example, we draw a scatter plot between two variables and color the samples by their respective class. In the figure below, it is clear that there are nonlinear structures in the data that cannot be possibly captured with the linear PCA. 


![plot of chunk unnamed-chunk-3](figure/unnamed-chunk-3-1.png) 

##Creating training, testing and validation datasets
For model building over the training dataset, we split the training dataset into 60\% training, 20\% testing and 20\% validation datasets. We make this partition by utilizing *createFolds()* function and create 5 folds. We assign 3 of them to the training data and 1 to each of the testing and the validation datasets. 



###Assessing Out of sample accuracy over validation set
We utilize the training set with 60\% of the data for model building. Since a model would be trained over this data, model performance over this data will be very good. In order to assess the model performance, we would utilize the test set that contains 20\% of the data. We build multiple models over the training set and test all of them over the test set. The best model amongst the tested models is then chosen. Note that the best model is chosen out of its performnace over the test set, therefore, test set is also considered part of the training data. 

The best model is then used to make predictions over the validation set that contains 20\% of the data. Since the validation set is not utilized in the process of model building, the performance of the model over validation set is taken as the model estimate of **out of sample** accuracy. 


##Prediction model

Our prediction model is based on the **ensemble** of 10 models and the final prediction is decided based on the **majority vote**. 

###Basic model
Let us first describe a basic model that can become 1 of the 10 models in the **ensemble**. We randomly choose 10 variables and 1000 observations in the training set. Recall that this set consist of 60\% of the original training set. Using 1000 observations, we construct a model using **stochastic gradient boosting** algorithm. The constructed model is then tested over the testing data. Recall that this testing data consist of 20\% of the original training dataset. Based on the model performance over the test set, we decided whether the model can be a part of the ensemble. If the model yields both the **sensitivity** and **specificity** of more than 0.7 for each of the 5 classes, it becomes the a **basic model** that is the part of the **ensemble**. 

A model with 10 variables and 1000 observations can have irrelevant variables or its rows can consist of outliers. Assessing its performance over the test set ensures model's real promise. 

###Formation of ensemble and predictions
An ensemble consist of 10 basic models constructed using the procedure,as described in the previous subsection. 
For making a prediction over a new input, each model in the ensemble is utilised to make a prediction. The final prediction is decided based on the **majority vote**. Our prediction algorithm as described can be achieved in the following *R* code.

###**Out of sample accuracy** over validation set
Once the ensemble is constructed using 10 **Basic models**, we use the ensemble to test the performance over validation set. Recall that validation set consist of 20\% of the original training set. We consider the performance of our ensemble prediction approach over this validation set to assess the **Out of sample accuracy**. We compute the confusion matrix of predictions over the validation set as follows. 

```
## Loading required package: gbm
## Loading required package: survival
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: splines
## Loaded gbm 2.1.1
## Loading required package: plyr
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1113    7    0    0    0
##          B    3  750    6    0    0
##          C    0    3  674    7    0
##          D    0    0    4  637    2
##          E    0    0    0    0  719
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9918          
##                  95% CI : (0.9885, 0.9944)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9897          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9973   0.9868   0.9854   0.9891   0.9972
## Specificity            0.9975   0.9972   0.9969   0.9982   1.0000
## Pos Pred Value         0.9937   0.9881   0.9854   0.9907   1.0000
## Neg Pred Value         0.9989   0.9968   0.9969   0.9979   0.9994
## Prevalence             0.2843   0.1936   0.1743   0.1641   0.1837
## Detection Rate         0.2836   0.1911   0.1717   0.1623   0.1832
## Detection Prevalence   0.2854   0.1934   0.1743   0.1638   0.1832
## Balanced Accuracy      0.9974   0.9920   0.9911   0.9937   0.9986
```

Note that the overall **Out of sample accuracy** is 98\% and the **sensitivity** for each of the 5 classes is more than 96\% and **specificity** for each of the 5 classes is more than 99\%.

##Predictions over the test set
We can consider the performance of our ensemble approach over validation set to be its estimated performance over the test set. This algorithm was utilised for predictions over the test set with 20 observations. Our algorithm yielded 100\% accuracy rate over the unseen test examples. My test score page is attached.
<br>
<img src="drawing.jpg" alt="Drawing" style="width: 1200px;"/>
<br>
