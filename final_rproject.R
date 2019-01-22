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

table(data_train$smoking_status)
prop.table(table(validate_without_smoke$stroke))

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
nrow(data.rose_smoke)

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
nrow(data.rose_nonsmoke)
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
IvValues<-create_infotables(data = data_balanced_under_nonsmoke,y="stroke")
print(IvValues$Summary, row.names = F)
aa<-IvValues$Summary > 0.02
aa

plot_infotables(IvValues, IvValues$Summary$Variable, same_scales = T)


#droppin residency  and id
drops <- c('Residence_type','id')
data_balanced_over_nonsmoke1<-data_balanced_over_nonsmoke[ , !(names(data_balanced_over_nonsmoke) %in% drops)]

#============================USING STEP FUNCTION=========================================
Model_With_nonSmoke_Data_over = glm(stroke ~., data = data_balanced_under_nonsmoke,family="binomial")
summary(Model_With_Smoke_Data_under)
step(Model_With_nonSmoke_Data_over,direction = 'both')

#creating a data frame to build final model
drops <- c('stroke','gender','age','hypertension','heart_disease','work_type','avg_glucose_level','bmi')
data_balanced_over_nonsmoke11<-data_balanced_over_nonsmoke1[ , (names(data_balanced_over_nonsmoke1) %in% drops)]

#write to csv
write.csv(validate_without_smoke, file = "C:/Users/Vinay/Downloads/validate_nonsmoke1.csv", row.names = FALSE)
View(data_balanced_over_nonsmoke1)
