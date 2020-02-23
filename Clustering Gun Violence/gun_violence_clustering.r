# K-Means Clustering on Gun violence data set

guns<-read.csv("C:/Users/E002891/Desktop/DayWiseTracker/Programming Concepts/Data Science/DataSets/Kaggle DataSets/gun-violence-data_01-2013_03-2018.csv", na.strings = c('',' ','  ','?','NA'))
nrow(guns)
set.seed(123)
guns<-guns[sample(1:nrow(guns),40000),]


#Preprocessing
summary(guns)
colnames(guns)
#Determining the columns to use
#The below cols are not required
#incident_id,date,address,incident_url,source_url,incident_url_fields_missing,incident_characteristics,location_description,
#notes,participant_age,participant_age_group,participant_name,participant_status,participant_type,sources,state_house_district,state_senate_district
for(i in colnames(guns[,c("incident_id","date","address","incident_url","source_url","incident_url_fields_missing","incident_characteristics","location_description","notes","participant_age","participant_age_group","participant_name","participant_status","participant_type","sources","state_house_district","state_senate_district")]))
{
  print(which(colnames(guns)==i))
}

guns<-guns[,-c(1,2,5,8,9,10,14,16,19,
               20,21,23,25,
               26,27,28,29)]
head(guns)

#Also the below columns are not required
guns$participant_gender<-NULL
guns$participant_relationship<-NULL
guns$gun_stolen<-NULL
View(guns)
table(guns$gun_type)
guns$gun_type<-NULL

mapply(table, guns)
table(guns$city_or_county)
guns$longitude<-NULL
guns$latitude<-NULL
guns$city_or_county<-NULL


#Factorization and bucketing
str(guns)
table(guns$n_killed)
#guns$n_killed<-ifelse(guns$n_killed<2,"Less",ifelse(guns$n_killed<4,"Medium","More"))
table(guns$n_injured)
#guns$n_injured<-ifelse(guns$n_injured<4,"Less",ifelse(guns$n_injured<10,"Medium","More"))
table(guns$congressional_district)
#guns$congressional_district<-ifelse(guns$congressional_district<20,"Less",ifelse(guns$congressional_district<40,"Medium","More"))
table(guns$n_guns_involved)
#guns$n_guns_involved<-ifelse(guns$n_guns_involved<50,"Less",ifelse(guns$n_guns_involved<150,"Medium","More"))
#guns$n_killed<-as.factor(guns$n_killed)
#guns$n_injured<-as.factor(guns$n_injured)
#guns$congressional_district<-as.factor(guns$congressional_district)
#guns$n_guns_involved<-as.factor(guns$n_guns_involved)



#Imputation
library(DMwR)
mapply(anyNA, guns)
guns<-guns[sample(1:nrow(guns),10000),]
guns<-knnImputation(guns, k = 10)



#Start Clustering
withinByBetween<-c()
for(i in 2:15)
{
  clusters<-kmeans(guns[,-c(1)], centers = i) #State not taken as its categorical
  withinByBetween<-c(withinByBetween,mean(clusters$withinss)/clusters$betweenss)
}


#Error more cluster centers than distinct data points. (As catagorical data). Hence commented the factorization codes
plot(2:15,withinByBetween, type="l")

#No Of clusters=7
clusters<-kmeans(guns[,-c(1)], centers = 7)
guns$cluster<-clusters$cluster
View(guns)
clusters$centers