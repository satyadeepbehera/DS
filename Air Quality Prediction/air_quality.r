# Predicting AirQuality using Linear Regression model (Using R)
# Dataset available at the below UCI repositoty location
https://archive.ics.uci.edu/ml/datasets/air+quality

data(airquality)
airquality
summary(airquality)
head(airquality)
mapply(table, airquality)
table(airquality$Day)

#Preprocessing Steps
summary(airquality)
mapply(anyNA, airquality)
library(DMwR)
airquality<-knnImputation(airquality,k=5)

#Normalize the dataset

mapply(shapiro.test, airquality)

minMaxFunc<-function(x){
  return((x-min(x))/(max(x)-min(x)))
}

airquality<-minMaxFunc(airquality)


str(airquality)

#Construct Model
rows<-1:nrow(airquality)
set.seed(123)
trainIndex<-sample(rows,round(0.8*length(rows)))
train<-airquality[trainIndex,]
test<-airquality[-trainIndex,]
nrow(train)/nrow(airquality)
nrow(test)/nrow(airquality)

model1<-lm(Ozone~.,data = train)
summary(model1)
# As month and day have no effect on model so removing them
train$Month<-NULL
train$Day<-NULL
test$Month<-NULL
test$Day<-NULL

model1<-lm(Ozone~.,data = train)
plot(model1)
abline(model1)
summary(model1)

preds<-predict(model1,test)
test$preds<-preds

#Calculating RMSE
rmse<-sqrt(mean((test$preds-test$Ozone)^2))
