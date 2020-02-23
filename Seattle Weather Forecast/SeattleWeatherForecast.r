# Seattle Weather Forecast using Logistic Regression

#Table description:
#DATE = the date of the observation
#PRCP = the amount of precipitation, in inches
#TMAX = the maximum temperature for that day, in degrees Fahrenheit
#TMIN = the minimum temperature for that day, in degrees Fahrenheit
#RAIN = TRUE if rain was observed on that day, FALSE if it was not


#Fetch the dataset
full<-read.csv("C:/Users/E002891/Desktop/DayWiseTracker/Programming Concepts/Data Science/DayWiseClasses/1stJun18_LogReg_Practicals_Self/seattleWeather_1948-2017.csv", na.strings = c(""," ","  ","?","NA"))
View(full)
nrow(full)


#Begin Preprocessing process
summary(full)
#Imputing the NA values of PRCP column
library(DMwR)
full<-knnImputation(full,k=5)

#Checking correlation (No changes required)
cor(full[,-c(1,5)])

str(full)
library(lubridate)
full$month<-month(full$DATE)
library(ggplot2)
ggplot(data = full, mapping = aes(x=month, y=RAIN))+geom_bar(stat = "identity")
full$RAIN<-ifelse(full$RAIN=='TRUE',1,0)
full$DATE<-NULL

#Rainy season of Seattle is between Oct and Mar. Hence updated the month based upon rainy season
full$month<-ifelse(full$month %in% c(10,11,12,1,2,3),"RainyMonth","NotRainyMonth")
full$month<-as.factor(full$month)


#Scaling
hist(log(full$PRCP))
library(forecast)
hist(BoxCox(full$PRCP,BoxCox.lambda(full$PRCP))) #Not efficient enough
full$PRCP<-log(full$PRCP)
hist((full$TMAX)^1/3)
hist((full$TMAX)) #Almost no change

hist(full$TMIN)
hist((full$TMIN)^1/3)  #Almost no change

#Scaling and Outlier management ignored as of now as we are going to make a naive model first

#Make the model
rows<-1:nrow(full)
set.seed(123)
trainRows<-sample(rows, round(0.8*length(rows)))
train<-full[trainRows,]
test<-full[-trainRows,]
nrow(train)/nrow(full)
nrow(test)/nrow(full)


model1<-glm(RAIN~.-PRCP,data=train,family = binomial(link = "logit"))
plot(model1)
summary(model1) #AIC: 20302

model2<-glm(RAIN~.-PRCP-month,data=train,family = binomial(link = "logit"))
summary(model2) #AIC: 20301

#Prediction
preds<-predict(model2,test,type='response')
range(preds)
test$preds<-preds

test$preds<-ifelse(test$preds>0.5,1,0)
View(test)

#Constructing the confusion matrix
table(test$preds,test$RAIN,dnn = c('Preds','Actuals'))
precision<-1486/(694+1486)
recall<-1486/(591+1486)
precision
recall

#By using in-built function
library(caret)
posPred<-posPredValue(as.factor(test$preds), as.factor(test$RAIN), positive="1")
sensitvty<-sensitivity(as.factor(test$preds), as.factor(test$RAIN), positive="1")
posPred
sensitvty


F1Score<-(2*precision*recall)/(precision+recall)
F1Score #0.6981442

#As the F1 Score is ~70%, we have made a decent model :):)

Accuracy<-(2339+1486)/(2339+1486+591+694)
Error<-(591+694)/(2339+1486+591+694)
Accuracy
Error



#Constructing the ROC curve
library(ROCR)
pred<-prediction(test$preds, test$RAIN)
perf<-performance(pred,'tpr','fpr')
plot(perf,colorize=TRUE,text.adj=c(0.25,0.25)) #Cut-off ~ 0.7
