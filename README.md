---
title: "Practical Machine Learning Project"
author: "Jacob Ider Chitham"
date: "10/07/2017"
output:
  pdf_document: default
  html_document:
    keep_md: yes
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```
# Goal
The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

# Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

# Load data
Set seed for reproducability and download data from the webpage if it is not already loaded in the variable does not already exist in the R session.
```{r load, ECHO = FALSE}
library(caret)
library(randomForest)
require(data.table)
set.seed(123)

DownloadIf <- function (x, dataURL) {
  if (!exists(deparse(substitute(x)))) {
    data.frame(fread(dataURL))
  }
  else {
    x
  }
}

testing <- DownloadIf(testing, paste("https://d396qusza40orc.cloudfront.net/",
                                     "predmachlearn/pml-testing.csv"))
training <- DownloadIf(training, paste("https://d396qusza40orc.cloudfront.net/",
                                       "predmachlearn/pml-training.csv"))
```

# Data Cleansing
Remove features that intutively do not contribute useful information to the algorithm or some features will give it too much infromation. These include "V1", "user\_name", "timestamp", "window", "problem\_id". It is also not a bad idea to shuffle the data

```{r clean}
CleanFeaures <- function(data){
  data <- data[sample(nrow(data), nrow(data)), ] # shuffle the data
  # remove features which will not improve the algorithm
  rmCols <- grepl("V1|user_name|timestamp|window|problem_id", colnames(data))
  data <- data[, (!rmCols)]
  # remove features which contain nulls, NAs or empties
  Filter(function(x) !any(is.na(x) || is.null(x) || x == ""), data)
  
}
```

# Split the data
Create validation set, and remove it from the original training set. This will be used when determining which classification method to chose.

```{r splitdata}
inVal <- createDataPartition(training$classe, p = 0.2, list = FALSE)
validation <- training[inVal, ]
training <- training[-inVal, ]
training.clean <-CleanFeaures(training)
```

# Remove zero-variarnce and correlated predictors
There are many models where predictors with a single unique value (also known as “zero- variance predictors”) will cause the model to fail. These so-called “near zero-variance predictors” can cause numerical problems during resampling for some models, such as linear regression. Check for low variance predictors by plotting correlation between features (excluding the classe labels which are in the final column).

The following code generates a corelation matrix between features, it is important to remove those which have a corelation near to one (i.e very blue).

```{r plorcor}
PlotCor <- function(data){       
  cor.matrix <- data
  tri <- lower.tri(cor.matrix, diag = FALSE)
  cor.matrix[tri] <- NA
  melted.cor.matrix <- melt(cor.matrix)
  ggplot(data = melted.cor.matrix, aes(x    = Var1, 
                                       y    = Var2, 
                                       fill = value)) + geom_tile()
}
```

repeat the following\:

* Find the pair of predictors with the largest absolute correlation
* For both predictors, compute the average correlation between each predictor and all of the other variables
* Flag the variable with the largest mean correlation for removal
* Remove this row and column from the correlation matrix
* until no correlations are above a threshold (in this case this is chosen to be 0.90)

```{r applyclean}
testing.clean <- CleanFeaures(testing)
training.clean <- CleanFeaures(training)
corTraining <- cor(training.clean[, -ncol(training.clean)])
highCor <- findCorrelation(corTraining, 0.90)
PlotCor(corTraining)
training.clean <- training.clean[, -highCor]
```

# Train the predictive models
Once the final set of predictors is determined, the values may require transformations before being used in a model. Some models, such as partial least squares, neural networks and support vector machines, need the predictor variables to be centered and/or scaled. The "preProcess()" function can be used to determine values for predictor transformations using the training set and can be applied to the test set or future samples. It is better to use the preProcess **argument** for the train function rather than "preProcess()" on the data prior to running the train function as this means the pre-processing will be applied to each re-sampling iteration. A function to train and analyse the data is shown below which passes a generic method as a string paramater to the train function. The technique chosen for data processing involves reapted cross validation to maximise accuracy by averaging over predictors on different samples of the training data multiple times. There is of course a trade off between algorithm performance and computational time. 

```{r trainanalysis}
train.control <- trainControl(method     = "repeatedcv",
                             number      = 3, 
                             repeats     = 1)

TrainAnalysis <- function (method){
  capture.output(model <- train(classe    ~ ., 
                               method     = method,
                               trControl  = train.control,
                               preProcess = c("center", "scale"),
                               data       = training.clean))
  capture.output(pred <- predict(model, subset(validation, select = -c(classe))))
  print(confusionMatrix(pred, validation$classe))
  model
}
```

# Results

The prediction results from neural network, gradient boosting machine and random forest are summarised below after preprocessing the data using the built in centre and scale techniques. These results also include confusion matricies using the validation data. From this information the best performing classifier is chosen as summarised in the following conclusion section.

```{r summary, quietly = TRUE}
methods <- c("nnet", "gbm", "rf")
for (method in methods){
  model.name <- paste("model.", method, sep = "")
  print(model.name)
  assign("modelAnalysis", TrainAnalysis(method))
  print("Predicted Results:")
  print(predict(modelAnalysis, testing))
}
```

# Conclusion

Clearly the random forest produces the best results against the validation set when this particular trainControl and preProcess configuration is used consistently accross teh selected classification algorithms. This is what will be used in the final test. Please note PCA was not found to improve the algorithms performance, this is most likely due to the fact that highly correlated and zero-variance faetures have already been removed therefore when PCA was included there was a tendancy to underfit the data.
