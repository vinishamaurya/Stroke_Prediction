library(dplyr)
library(caret)
library(ROCR)
library(ROSE)
library(fastDummies)
library(mlbench)
library(Hmisc)
library(randomForest)
library(Information)
library(rpart)


#importing train data
data_train = read.csv("C:/Users/Vinay/Downloads/train_2v.csv")



## checking the column in the data
head(data_train)
## Summary of the data
summary(data_train)
#Structure of the data
str(data_train)

## Checking Missing Values in the dataset
sum(is.na(data_train))  ##1462 missing values in entire data set
sum(is.na(data_train$bmi)) ##1462 missing values for BMI

## Checking the values in the target variable
sum(data_train$stroke)  ## 783 records have stroke predicted as '1'  so it is imbalanced data

## Checking if there are otuliers
boxplot(data_train$bmi) #outliers are present
boxplot(data_train$avg_glucose_level)# Outliers are present
boxplot(data_train$age) # no otuliers

## Missing Value Imptataion for BMI
## boxplot for BMI so we can see then there outliers
## but there are some extreme values which is not realistic. The max of BMI can be upto 62 in extreme cases
## Conting values that are greater than 62
nrow(subset(data_train,bmi > 62))
data_train$bmi=ifelse(is.na(data_train$bmi),median(data_train$bmi ,na.rm = TRUE), data_train$bmi)

## Winsorization for BMI
bmi_matrix = matrix(data_train$bmi, 1, byrow=TRUE)
max_value_bmi <- 32.90 + 1.5*IQR(bmi_matrix)
data_train$bmi[data_train$bmi > max_value_bmi] <-max_value_bmi

###Univariate Analaysis
hist(data_train$age, xlab = 'Age', main = 'Histogram of Age')
hist(data_train$bmi, xlab = 'BMI', main = 'Histogram of BMI')
hist(data_train$avg_glucose_level, xlab = 'Average Glucose Level', main = 'Histogram of Average Glucose Level')


barplot(table(data_train$gender), main = 'Barplot for Gender', xlab = 'Gender')
barplot(table(data_train$heart_disease), xlab = 'Heart Disease', main = 'Barplot for Heart Disease')
barplot(table(data_train$hypertension), xlab = 'Hypertension', main = 'Barplot for Hypertension')

## bargrpahs for factor variables
ggplot(data_train) + aes(x = smoking_status, position = 'stack') + geom_bar()
ggplot(data_train) + aes(x = data_train$gender) + geom_bar()
ggplot(data_train) + aes(x = data_train$work_type) + geom_bar()
ggplot(data_train) + aes(x = data_train$hypertension) + geom_bar()


### Bivariate Analysis

data_train$hypertension = as.factor(data_train$hypertension)
data_train$stroke = as.factor(data_train$stroke)

ggplot(data_train, aes(hypertension, ..count..))+
  geom_bar(aes(fill = stroke), position = "dodge") +
  labs(title = "Distribution of HyperTension relative to Stroke",
       x = "Hypertension",
       y = "Count")

data_train$heart_disease = as.factor(data_train$heart_disease)

ggplot(data_train, aes(heart_disease, ..count..))+
  geom_bar(aes(fill = stroke), position = "dodge") +
  labs(title = "Distribution of Heart Disease relative to Stroke",
       x = "Heart Disease",
       y = "Count")

data_train$ever_married = as.factor(data_train$ever_married)

ggplot(data_train, aes(ever_married, ..count..))+
  geom_bar(aes(fill = stroke), position = "dodge") +
  labs(title = "Distribution of Married relative to Stroke",
       x = "Married",
       y = "Count") + theme(plot.title = element_text(hjust = .5))

data_train$gender = as.factor(data_train$gender)

ggplot(data_train, aes(gender, ..count..))+
  geom_bar(aes(fill = stroke), position = "dodge") +
  labs(title = "Distribution of Gender relative to Stroke",
       x = "Gender",
       y = "Count") + theme(plot.title = element_text(hjust = .5))


data_train$work_type = as.factor(data_train$work_type)

ggplot(data_train, aes(work_type, ..count..))+
  geom_bar(aes(fill = stroke), position = "dodge") +
  labs(title = "Distribution of Worktype relative to Stroke",
       x = "Work Type",
       y = "Count") + theme(plot.title = element_text(hjust = .5))


data_train$Residence_type = as.factor(data_train$Residence_type)

ggplot(data_train, aes(Residence_type, ..count..))+
  geom_bar(aes(fill = stroke), position = "dodge") +
  labs(title = "Distribution of Residence relative to Stroke",
       x = "Residence Type",
       y = "Count") + theme(plot.title = element_text(hjust = .5))


ggplot(data_train, aes(data_train$smoking_status, ..count..))+
  geom_bar(aes(fill = stroke), position = "dodge") +
  labs(title = "Distribution of Smoking status to Stroke",
       x = "smoking status",
       y = "Count") + theme(plot.title = element_text(hjust = .5))

ggplot(data_train, aes(as.factor(stroke), bmi))+
  geom_boxplot(col = "blue")+
  labs(title = "BMI by Stroke",
       x = "stroke")+
  theme(plot.title = element_text(hjust = .5))


ggplot(data_train, aes(as.factor(stroke), avg_glucose_level))+
  geom_boxplot(col = "blue")+
  ggtitle("Distribution of Glucose by Stroke")+
  xlab("stroke")+
  theme(plot.title = element_text(hjust = .5))


#============================================================================================
#                               DATA PREPARATION
#============================================================================================

#converting type to factors
data_train$hypertension = as.factor(data_train$hypertension)

data_train$heart_disease = as.factor(data_train$heart_disease)


#seperate model for with Smoking data and without smoking data and joining the Stroke data



train_data_With_Smoke_Data = data_train[ data_train$smoking_status != "", ]
View(train_data_With_Smoke_Data)
train_data_Without_Smoke_Data = data_train[ data_train$smoking_status == "", ]
nrow(train_data_Without_Smoke_Data)
nrow(train_data_With_Smoke_Data)
View(train_data_Without_Smoke_Data)


#check classes distribution
#for smoking status available
table(train_data_With_Smoke_Data$stroke)     
prop.table(table(train_data_With_Smoke_Data$stroke))
#for smoking status not available
table(train_data_Without_Smoke_Data$stroke)
prop.table(table(train_data_Without_Smoke_Data$stroke))


#=======================================================================
#                    TRAIN-TEST SPLIT to 80%-20%
#=======================================================================

#DATA:
set.seed(0)
rand = sample(1:nrow(train_data_With_Smoke_Data)*0.8)
train_smoke = train_data_With_Smoke_Data[rand, ]
validate_smoke = train_data_With_Smoke_Data[-rand, ]


set.seed(0)
rand = sample(1:nrow(train_data_Without_Smoke_Data)*0.8)
train_without_smoke = train_data_Without_Smoke_Data[rand, ]
validate_without_smoke = train_data_Without_Smoke_Data[-rand, ]

#=======================================================================
#                   BALANCING SMOKING STATUS
#=======================================================================

#over sampling
 data_balanced_over_smoke <- ovun.sample(stroke ~ ., data = train_smoke, method = "over",N=30000*2)$data
table(data_balanced_over_smoke$stroke)
prop.table(table(data_balanced_over_smoke$stroke))


#undersampling
data_balanced_under_smoke <- ovun.sample(stroke ~ ., data = train_smoke, method = "under", N = 638*2, seed = 1)$data
table(data_balanced_under_smoke$stroke)
prop.table(table(data_balanced_under_smoke$stroke))


#both
data_balanced_both_smoke <- ovun.sample(stroke ~ ., data = train_smoke, method = "both", p=0.5,                             N=1000, seed = 1)$data
table(data_balanced_both_smoke$stroke)
prop.table(table(data_balanced_both_smoke$stroke))


#ROSE
data.rose_smoke <- ROSE(stroke ~ ., data = train_smoke, seed = 1)$data
table(data.rose_smoke$stroke)
prop.table(table(data.rose_smoke$stroke))


#=======================================================================
#                    BALANCING NON-SMOKE DATA
#=======================================================================
#dropping smoking_status variable
drops <- c('smoking_status')
train_without_smoke<-train_without_smoke[ , !(names(train_without_smoke) %in% drops)]
View(train_without_smoke)

drops <- c('smoking_status')
validate_without_smoke<-validate_without_smoke[ , !(names(validate_without_smoke) %in% drops)]
View(validate_without_smoke)

#over sampling
data_balanced_over_nonsmoke <- ovun.sample(stroke ~ ., data = train_without_smoke, method = "over",N=13129*2)$data
table(data_balanced_over_nonsmoke$stroke)
prop.table(table(data_balanced_over_nonsmoke$stroke))

#undersampling
data_balanced_under_nonsmoke <- ovun.sample(stroke ~ ., data = train_without_smoke, method = "under", N = 162*2, seed = 1)$data
table(data_balanced_under_nonsmoke$stroke)
prop.table(table(data_balanced_under_nonsmoke$stroke))

#both
data_balanced_both_nonsmoke <- ovun.sample(stroke ~ ., data = train_without_smoke, method = "both", p=0.5,                             N=1000, seed = 1)$data
table(data_balanced_both_smoke$stroke)
prop.table(table(data_balanced_both_nonsmoke$stroke))

#ROSE
data.rose_nonsmoke <- ROSE(stroke ~ ., data = train_without_smoke, seed = 1)$data
table(data.rose_smoke$stroke)
prop.table(table(data.rose_nonsmoke$stroke))

#write to csv
write.csv(validate_without_smoke, file = "C:/Users/Vinay/Downloads/validate_nonsmoke.csv", row.names = FALSE)


#=====================================================================================
#                         FEATURE SELECTION FOR OVER-SAMPLED DATA
#=====================================================================================

#===========================uSING INFORMATION GAIN====================================

IvValues<-create_infotables(data = data_balanced_over_smoke,y="stroke")
#summary of information values for all variables
print(IvValues$Summary, row.names = F)
#checking variables which are greater than 0.02, that are not contributing to model building
aa<-IvValues$Summary > 0.02
aa
#plot of IV
plot_infotables(IvValues, IvValues$Summary$Variable, same_scales = T)


#droppin residency and ID based on IV 

drops <- c('Residence_type','id')
data_balanced_over_smoke1<-data_balanced_over_smoke[ , !(names(data_balanced_over_smoke) %in% drops)]


#================================USING STEP FUNCTION=================================

#fitting glm model
Model_With_Smoke_Data_over = glm(stroke ~., data = data_balanced_over_smoke1,family="binomial")
#getting results from step function
step(Model_With_Smoke_Data_over,direction = 'both')

# creating data frame with feauters selected from step model to build the final model
drops <- c('stroke','age','hypertension','heart_disease','work_type','avg_glucose_level','bmi','smoking_status')
data_balanced_over_smoke11<-data_balanced_over_smoke1[ , (names(data_balanced_over_smoke1) %in% drops)]


#write to csv
write.csv(data_balanced_over_smoke11, file = "C:/Users/Vinay/Downloads/data_balanced_over_smoke11.csv", row.names = FALSE)
View(data_balanced_over_smoke11)


#======================================================================================
#                        FEATURE SELECTION FOR UNDERSAMPLED DATA
#======================================================================================

#=============================USING INFORMATION VALUE==================================

IvValues<-create_infotables(data = data_balanced_under_smoke,y="stroke")
print(IvValues$Summary, row.names = F)
aa<-IvValues$Summary > 0.02
aa
plot_infotables(IvValues, IvValues$Summary$Variable, same_scales = T)

#droppin residency and id

drops <- c('Residence_type','id')
data_balanced_under_smoke1<-data_balanced_under_smoke[ , !(names(data_balanced_under_smoke) %in% drops)]

#=================================USING STEP FUCTION=====================================
Model_With_Smoke_Data_under = glm(stroke ~., data = data_balanced_under_smoke1,family="binomial")
step(Model_With_Smoke_Data_under,direction = 'both')

#==========================================================================================
#                    FEATURE SELECTION FOR NON-SMOKE DATA
#==========================================================================================

#=========================INFORMATION VALUE================================================
IvValues<-create_infotables(data = data_balanced_over_nonsmoke,y="stroke")
print(IvValues$Summary, row.names = F)
aa<-IvValues$Summary > 0.02
aa

plot_infotables(IvValues, IvValues$Summary$Variable, same_scales = T)


#droppin residency  and id
drops <- c('Residence_type','id')
data_balanced_over_nonsmoke1<-data_balanced_over_nonsmoke[ , !(names(data_balanced_over_nonsmoke) %in% drops)]

#============================USING STEP FUNCTION=========================================
Model_With_nonSmoke_Data_over = glm(stroke ~., data = data_balanced_over_nonsmoke1,family="binomial")
summary(Model_With_Smoke_Data_under)
step(Model_With_nonSmoke_Data_over,direction = 'both')

#creating a data frame to build final model
drops <- c('stroke','gender','age','hypertension','heart_disease','work_type','avg_glucose_level','bmi')
data_balanced_over_nonsmoke11<-data_balanced_over_nonsmoke1[ , (names(data_balanced_over_nonsmoke1) %in% drops)]

#write to csv
write.csv(data_balanced_over_nonsmoke1, file = "C:/Users/Vinay/Downloads/data_balanced_over_nonsmoke11.csv", row.names = FALSE)
View(data_balanced_over_nonsmoke1)



#--------------------------------------------------------------------------------------------------
#            FITTING LOGISTIC REGRESSION considering SMOKING STATUS As a Factor in Data
#---------------------------------------------------------------------------------------------------

# Calling required libraries
library(caTools)
library(caret)
library(pROC)
library(InformationValue)


# Lodaing train and test data
data=read.csv("G:\\Analytics_Praxis\\R\\R-Data\\data_balanced_over_smoke11.csv")
View(data)
testdata=read.csv("G:\\Analytics_Praxis\\R\\R-Data\\validate_smoke.csv")
View(testdata)

#Checks for proportion of strokes in both the data
prop.table(table(data$stroke))*100
prop.table(table(testdata$stroke))*100

#------------------------------------------------------------------------------------------------
# MODEL 1
#Fitting a logistic model on the train data
#-------------------------------------------------------------------------------------------------

model_smoke = glm(stroke~.,data=data,family = binomial)
#its a binomial so we are passing binomial in the family

summary(model_smoke)# Summary of the model fitted

# Predicting values for stroke in test data set
pred=predict(model_smoke,newdata =testdata[-12],type = 'response' )
# Setting a threshold of 0.5
pred<-ifelse(pred>0.5,1,0)

#---------------------------------------------------------------------------------------------------
# Evaluating the model 1
#---------------------------------------------------------------------------------------------------
# confusiom Matrix - built inside caret package
confusionMatrix(table(pred,testdata$stroke),positive = '1')

# Calculating seperately these values
# this functions are in built inside package Informationvalue
table(testdata$stroke,pred)
misClassError(testdata$stroke,pred)
sensitivity(testdata$stroke,pred)
specificity(testdata_nonsmoke$stroke,pred)
precision(testdata$stroke,pred)
pre=precision(testdata$stroke,pred)
recall=sensitivity(testdata$stroke,pred)
# Calculating F1 score for test data
F_i = (1+i^2)*pre*recall/((i^2 * pre)+recall)
F_i


#Plotting an ROC curve 
#-----------------------------------------------------------------------------------
# roc is built inside pROC package
auc <- roc(testdata$stroke, pred)
print(auc)

## Area under the curve: 0.75
plot(auc, ylim=c(0,1), main=paste('AUC:',round(auc1$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

#Concordance and Discordance
#------------------------------------------------------------------------
conc=Concordance(testdata$stroke,pred)
conc




#----------------------------------------------------------------------------------------------
#       FITTING LOGISTIC REGRESSION by NOT considering SMOKING STATUS As a Factor in Data
#---------------------------------------------------------------------------------------------

# Lodaing train and test data
data_nosmoke=read.csv("G:\\Analytics_Praxis\\R\\R-Data\\data_balanced_over_nonsmoke11.csv")
View(data_nosmoke)
testdata_nonsmoke=read.csv("G:\\Analytics_Praxis\\R\\R-Data\\validate_nonsmoke1.csv")
View(testdata_nonsmoke)
nrow(testdata_nonsmoke)


#Checks for proportion of strokes in both the data
table(data_nosmoke$stroke)
table(testdata_nonsmoke$stroke)

#------------------------------------------------------------------------------------------------
# MODEL 2
#Fitting a logistic model on the train data
#-------------------------------------------------------------------------------------------------

model_nosmoke = glm(stroke~.,data=data_nosmoke,family = binomial)
#its a binomial so we are passing binomial in the family

summary(model_nosmoke)   # Summary of the model

# Predicting values for stroke in test data set
pred_nosmoke=predict(model_nosmoke,newdata =testdata_nonsmoke[-11],type = 'response' )
pred_nosmoke<-ifelse(pred_nosmoke>0.5,1,0)

#---------------------------------------------------------------------------------------------------
# Evaluating the model 2
#---------------------------------------------------------------------------------------------------

# confusiom Matrix 
#-----------------------------------------------------------
# Calculating seperately these values
# this functions are in built inside package Informationvalue
table(testdata_nonsmoke$stroke,pred_nosmoke)
misClassError(testdata_nonsmoke$stroke,pred_nosmoke)
sensitivity(testdata_nonsmoke$stroke,pred_nosmoke)
specificity(testdata_nonsmoke$stroke,pred_nosmoke)
precision(testdata_nonsmoke$stroke,pred_nosmoke)
#Accu=(2105+19)/(2105+534+20)
#Accu



#Plotting an ROC curve (Choosing Cut-off using ROC curve)
#------------------------------------------------------------------------
# roc is built inside pROC package
auc1 <- roc(testdata_nonsmoke$stroke, pred_nosmoke)
print(auc1)
# Auc is 0.87
plot(auc1, ylim=c(0,1), main=paste('AUC:',round(auc1$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

#Concordance and Discordance
#------------------------------------------------------------------------
conc=Concordance(testdata_nonsmoke$stroke,pred_nosmoke)
conc



#------------------------------------------------------------------------------------------------
#             END of Logistic Regression
#-------------------------------------------------------------------------------------------------

#----------------------------------------------------------------
#   Tree Based Model
#------------------------------------------------------------------
library(ggplot2)   #For plots
library(lattice)   #For plots and for caret
library(caret)     #ML package
library(rpart)     #ulibrary for rpart function for CART
library(rpart.plot)#plot trees
library(pROC)      #For ROC curve
library(maptree)   #For mapping trees (plot)
library(partykit)   #for ctree, plot
library(MLmetrics)

ovdata=read.csv("C:/Users/91944/Documents/R/praxis/data/overtrain_smoke.csv")
undata=read.csv("C:/Users/91944/Documents/R/praxis/data/undertrain_smoke.csv")
test_data=read.csv("C:/Users/91944/Documents/R/praxis/data/validate_smoke.csv")
ovndata=read.csv("C:/Users/91944/Documents/R/praxis/data/overtrain_nonsmoke.csv")
unndata=read.csv("C:/Users/91944/Documents/R/praxis/data/undertrain_nonsmoke.csv")
ntest_data=read.csv("C:/Users/91944/Documents/R/praxis/data/validate_nonsmoke1.csv")


# train a Classification Tree
fit <- rpart(stroke~., data=ovdata, method="class",cp=0.013) 
fit2<- rpart(stroke~., data=undata, method="class",cp=0.02)

# display results data-1
plotcp(fit)
print(fit)
plot(fit)
text(fit)
rpart.plot(fit,main="Classification Tree",extra = 100)


# make predictions
predictions = predict(fit, test_data[,1:10],type='class')
#print(predictions)   
#Confusion Matrix

table(test_data$stroke,predictions)
Accuracy(predictions,test_data$stroke)
F1_Score(predictions,test_data$stroke)
#Convert predictions factor to numbeic
predictions=as.numeric(as.character(predictions))

#Area under curve
auc(test_data$stroke,predictions)   

plot(roc(test_data$stroke,predictions))  #Roc plot


library(ROCR)
#Plot ROC curve
ROCpred <- prediction(predictions,test_data$stroke)
ROCperf <- performance(ROCpred,"tpr","fpr")
plot(ROCperf)


#display results data-2  #########Uv data
plotcp(fit2)
print(fit2)
plot(fit2)
text(fit2)
rpart.plot(fit2,main="Classification Tree",extra = 100)

# make predictions
predictions = predict(fit2, test_data[,1:10],type='class')
#print(predictions)   
#Confusion Matrix

table(test_data$stroke,predictions)
Accuracy(predictions,test_data$stroke)
F1_Score(predictions,test_data$stroke)
#Convert predictions factor to numbeic
predictions=as.numeric(as.character(predictions))

#Area under curve
auc(test_data$stroke,predictions)   

plot(roc(test_data$stroke,predictions))  #Roc plot

###################################################NON SMOKE


# train a Classification Tree
fit3<- rpart(stroke~., data=ovndata, method="class",cp=0.014) 
fit4<- rpart(stroke~., data=unndata, method="class",cp=0.011)

# display results data-1
plotcp(fit3)
print(fit3)
plot(fit3)
text(fit3)
rpart.plot(fit3,main="Classification Tree",extra = 101)


# make predictions
predictions = predict(fit3, ntest_data[,1:9],type='class')
#print(predictions)   
#Confusion Matrix

table(ntest_data$stroke,predictions)
Accuracy(predictions,ntest_data$stroke)
F1_Score(predictions,ntest_data$stroke)
#Convert predictions factor to numbeic
predictions=as.numeric(as.character(predictions))

#Area under curve
auc(ntest_data$stroke,predictions)   

plot(roc(ntest_data$stroke,predictions))  #Roc plot


library(ROCR)
#Plot ROC curve
ROCpred <- prediction(predictions,test_data$stroke)
ROCperf <- performance(ROCpred,"tpr","fpr")
plot(ROCperf)


#display results data-4  #########Uv data
plotcp(fit4)
print(fit4)
plot(fit4)
text(fit4)
rpart.plot(fit4,main="Classification Tree",extra = 100)

# make predictions
predictions = predict(fit4, ntest_data[,1:9],type='class')
#print(predictions)   
#Confusion Matrix

table(ntest_data$stroke,predictions)
Accuracy(predictions,ntest_data$stroke)
F1_Score(predictions,ntest_data$stroke)
#Convert predictions factor to numbeic
predictions=as.numeric(as.character(predictions))

#Area under curve
auc(ntest_data$stroke,predictions)   

plot(roc(ntest_data$stroke,predictions))  #Roc plot

################################################Random Forest
library(randomForest)
fit5=randomForest(stroke~., data=ovdata )

# make predictions
predictions = predict(fit5, newdata=test_data[,1:10],type="class")
predictions=ifelse(predictions>0.07,1,0)  #Threshold increases then accuracy increases but 1s predicted decreases 
#print(predictions)   
#Confusion Matrix

table(test_data$stroke,predictions)


Accuracy(predictions,test_data$stroke)
F1_Score(predictions,test_data$stroke)
#Convert predictions factor to numbeic
predictions=as.numeric(as.character(predictions))

#Area under curve
auc(test_data$stroke,predictions)   

plot(roc(test_data$stroke,predictions))  #Roc plot
