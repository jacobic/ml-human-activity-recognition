# setwd("~/johnshopkins/practicalml")
pkg.list <- c("ggplot2", "caret", "randomForest", "e1071",
              "data.table", "reshape2")
InstallLib <- function (pkgs) {
  lapply(pkgs,
         function(x) if (!(x %in% rownames(installed.packages()))) {
             install.packages(x)
             })
  lapply(pkgs,
         function(x) if (!(x %in% loadedNamespaces())) {
             library(x)
             })
  cat("\014") #clears the screen
}

# Install the require packages
InstallLib(pkg.list)
require(data.table)




# set seed
set.seed(123)

# load/split data
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

# dev.off(dev.list()["RStudioGD"])

CleanFeaures <- function(data){
  # remove features that intutively do not contribute useful information 
  # to the algorithm 
  # ...or some features will give it too much infromation
  
  # if(deparse(substitute(data)) != "testing") {
  #  data$classe<-as.factor(data$classe)
  # }
  
  data <- data[sample(nrow(data), nrow(data)), ]
  rmCols <- grepl("V1|user_name|timestamp|window|problem_id", colnames(data))
  data <- data[, (!rmCols)]
  # remove features which contain nulls, NAs or empties
  Filter(function(x) !any(is.na(x) || is.null(x) || x == ""), data)
  # shuffle the data
}

# create validation set, and remove it from thr original training det
inVal <- createDataPartition(training$classe, p = 0.2, list = FALSE)
validation <- training[inVal, ]
training <- training[-inVal, ]
training.clean <-CleanFeaures(training)

# There are many models where predictors with a single unique value (also known 
# as “zero- variance predictors”) will cause the model to fail. These so-called 
# “near zero-variance predictors” can cause numerical problems during resampling 
# for some models, such as linear regression. 

# Check for low variance predictors by plotting correlation between features 
# (excluding the classe labelswhich are in the final column )

PlotCor <- function(data){       
  cor.matrix <- data
  tri <- lower.tri(cor.matrix, diag = FALSE)
  # tri<-tri[,c(ncol(tri):1)]
  cor.matrix[tri] <- NA
  melted.cor.matrix <- melt(cor.matrix)
  ggplot(data = melted.cor.matrix, aes(x    = Var1, 
                                       y    = Var2, 
                                       fill = value)) + geom_tile()
}

# repeat
# Find the pair of predictors with the largest absolute correlation;
# For both predictors, compute the average correlation between each predictor 
# and all of the other variables;
# Flag the variable with the largest mean correlation for removal;
# Remove this row and column from the correlation matrix;
# until no correlations are above a threshold ;

training.clean <- CleanFeaures(training)
corTraining <- cor(training.clean[, -ncol(training.clean)])
highCor <- findCorrelation(corTraining, 0.90)
PlotCor(corTraining)
training.clean <- training.clean[, -highCor]

# Once the final set of predictors is determined, the values may require
# transformations before being used in a model. Some models, such as partial 
# least squares, neural networks and support vector machines, need the predictor 
# variables to be centered and/or scaled. The preProcess function can be used to
# determine values for predictor transformations us- ing the training set and 
# can be applied to the test set or future samples

# It is better to use the preProcess argument for the train function rather than 
# preProcess() on the data prior to running the train function as this means the 
# pre-processing will be applied to each re-sampling iteration.

methods <- c("nnet", "gbm", "rf")
train.control <- trainControl(method     = "repeatedcv",
                             number      = 3, 
                             repeats     = 1,
                             p           = 0.6)

TrainAnalysis <- function (method){
  capture.output(model <- train(classe    ~ ., 
                               method     = method,
                               trControl  = train.control,
                               data       = training.clean))
  pred <- predict(model, subset(validation, select = -c(classe)))
  print(confusionMatrix(pred, validation$classe))
  model
}

for (method in methods){
  model.name <- paste("model.", method, sep = "")
  assign(model.name, TrainAnalysis(method))
  print(model.name)
  predict(model.name, testing)
}