---
title: "Practical Machine Learning Project"
author: "Andrew Volz"
date: "10/28/2020"
output:
  html_document:
    keep_md: yes
---

## Synopsis

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

The goal of this project is to predict the manner in which they did the exercise. This is the `classe` variable in the training set.

## Data Description

The outcome variable is `classe`, a factor variable with 5 levels. For this data set, participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in 5 different fashions:

- exactly according to the specification (Class A)
- throwing the elbows to the front (Class B)
- lifting the dumbbell only halfway (Class C)
- lowering the dumbbell only halfway (Class D)
- throwing the hips to the front (Class E)

## Setting Things Up

First, load in some required packages and initialize some variables.


```r
#Data variables
training.file   <- './data/pml-training.csv'
test.cases.file <- './data/pml-testing.csv'
training.url    <- 'http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
test.cases.url  <- 'http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
#Directories
if (!file.exists("data")){
  dir.create("data")
}
#R-Packages
IscaretInstalled <- require("caret")
```

```
## Loading required package: caret
```

```
## Warning: package 'caret' was built under R version 3.6.3
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.6.3
```

```r
if(!IscaretInstalled){
    install.packages("caret")
    library("caret")
    }
IsrandomForestInstalled <- require("randomForest")
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.6.3
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
if(!IsrandomForestInstalled){
    install.packages("randomForest")
    library("randomForest")
    }
IsRpartInstalled <- require("rpart")
```

```
## Loading required package: rpart
```

```r
if(!IsRpartInstalled){
    install.packages("rpart")
    library("rpart")
    }
IsRpartPlotInstalled <- require("rpart.plot")
```

```
## Loading required package: rpart.plot
```

```
## Warning: package 'rpart.plot' was built under R version 3.6.3
```

```r
if(!IsRpartPlotInstalled){
    install.packages("rpart.plot")
    library("rpart.plot")
    }
# Set seed for reproducability
set.seed(1000)
```

## Receiving, Exploring, and Cleaning the Data
Next, the data is downloaded. Some EDA was performed, but omitted to keep this write up more concise. From the EDA, some manipulations and cleanup will be performed, so that `NA` values are omitted. Unneeded columns (columns 1 to 7) will be removed as well.

The `pml-training.csv` data will be split into training and testing sets.
The `pml-test.csv` data is going to be used to predict the unknown `classe` varibale for 20 participants.


```r
# Download data
download.file(training.url, training.file)
download.file(test.cases.url,test.cases.file )
# Clean data
training   <-read.csv(training.file, na.strings=c("NA","#DIV/0!", ""))
testing <-read.csv(test.cases.file , na.strings=c("NA", "#DIV/0!", ""))
training<-training[,colSums(is.na(training)) == 0]
testing <-testing[,colSums(is.na(testing)) == 0]
# Remove Unneeded columns
training   <-training[,-c(1:7)]
testing <-testing[,-c(1:7)]
```

## Cross-Validation
Cross-Validation will be performed by splitting the training data, `pml-training.csv`, into a training set and a testing set, 75% and 25% respectively, as stated previously.


```r
subSamples <- createDataPartition(y=training$classe, p=0.75, list=FALSE)
subTraining <- training[subSamples, ] 
subTesting <- training[-subSamples, ]
```

Now that the data is set up the way it needs to be, we can start looking at some prediction models.

## Prediction Models

### Decision Tree
A Decision Tree is the first prediction model explored. The model will be run using the subTraining data set and then used to predict `classe` for the subTesting data set to capture accuracy.


```r
# Fit model
modelFitDT <- rpart(classe ~ ., data=subTraining, method="class")
# Perform prediction
predictDT <- predict(modelFitDT, subTesting, type = "class")
# Plot result
rpart.plot(modelFitDT, main="Decision Tree", extra=102 , under=TRUE)
```

![](Practical-Machine-Learning-Project_files/figure-html/decisiontree-1.png)<!-- -->

A confusion matrix shows the accuracy of this prediction model.


```r
confusionMatrix(predictDT, subTesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1232  142    8   39   17
##          B   44  598   74   59   72
##          C   48   93  694  122  111
##          D   44   73   63  525   51
##          E   27   43   16   59  650
## 
## Overall Statistics
##                                          
##                Accuracy : 0.7543         
##                  95% CI : (0.742, 0.7663)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.689          
##                                          
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8832   0.6301   0.8117   0.6530   0.7214
## Specificity            0.9413   0.9370   0.9076   0.9437   0.9638
## Pos Pred Value         0.8567   0.7060   0.6498   0.6944   0.8176
## Neg Pred Value         0.9530   0.9135   0.9580   0.9327   0.9389
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2512   0.1219   0.1415   0.1071   0.1325
## Detection Prevalence   0.2932   0.1727   0.2178   0.1542   0.1621
## Balanced Accuracy      0.9122   0.7836   0.8597   0.7983   0.8426
```

With the Decision Tree only predicting at 75.43% accuracy, other models will be explored to try to capture a better prediction model.

### Random Forest
A Random Forest is the next prediction model explored.Again, the model will be run using the subTraining data set and then used to predict `classe` for the subTesting data set to capture accuracy.


```r
# Fit model
modelFitRF <- randomForest(classe ~ ., data=subTraining, method="class")
# Perform prediction
predictRF <- predict(modelFitRF, subTesting, type = "class")
```

A confusion matrix shows the accuracy of this prediction model.


```r
confusionMatrix(predictRF, subTesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    5    0    0    0
##          B    0  939    7    0    0
##          C    0    5  846   10    0
##          D    0    0    2  793    6
##          E    0    0    0    1  895
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9927          
##                  95% CI : (0.9899, 0.9949)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9907          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9895   0.9895   0.9863   0.9933
## Specificity            0.9986   0.9982   0.9963   0.9980   0.9998
## Pos Pred Value         0.9964   0.9926   0.9826   0.9900   0.9989
## Neg Pred Value         1.0000   0.9975   0.9978   0.9973   0.9985
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1915   0.1725   0.1617   0.1825
## Detection Prevalence   0.2855   0.1929   0.1756   0.1633   0.1827
## Balanced Accuracy      0.9993   0.9938   0.9929   0.9922   0.9965
```

The Random Forest predicts with 99.27% accuracy. This model is very accurate and is the one that will be used to predict `classe` on the testing set.

## Conclusion

### Final Model

The confusion matrices show that the Random Forest predicts better than the Decision Tree. The accuracy for the Random Forest was 0.9927 (95% CI: (0.9899, 0.9949)) compared to 0.7543 (95% CI: (0.742, 0.7663)) for the Decision Tree. As stated previously, the Random Forest is chosen to predict `classe` on the test set.

### Expected out-of-sample error
The expected out-of-sample error is estimated at 0.0073, or 0.73%. This is calculated as 1 - accuracy for predictions made against the cross-validation set. With this expected out-of-sample error we expect to classify the testing set near perfectly.

## Prediction
The predicted `classe` values for the testing data set are generated using the Random Forest model These predicted values will then be used for the Prediction Quiz.


```r
# Perform prediction
predictFinal <- predict(modelFitRF, testing, type="class")
predictFinal
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
