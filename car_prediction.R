#title : car prediction using knn algorithm

#this project is all about understanding the dataset and making predictions of price using
#other independent variables  by taking whole dataset at the beginning,and then by dropping 
#the variables that are highly correlated with the predictor variables thus visualizing the
#prediction in the form of line graph.

#the library() is used to load the package to an environment where "readr" is used to read csv
library("readr")

#the readr function is used to read the data from folder and it is then assigned to car 
#variable
car = read_csv("C:/Users/USER/Desktop/dataquest-R/imports-85.data")

#the head() is used to display the top rows from the dataset
head(car,4)

#the column names are found out and the col_names vector is created
col_names <- c("Symboling","Loss","make","fuel_type","aspiration","num_of_doors","body_style",
               "drive_wheels","engine_location","wheel_base","length","width","height",
               "curb_weight","engine_type","no_of_cylinders","engine_size","fuel_system",
               "bore","stroke","compression_ratio","horse_pw","peak_rpm","city_mpg",
               "highway_mpg","price")

#the names() is used to name the car attributes columns using previously built name  vector
names(car) <- col_names

#the the columns(hp,price) required for calculations are then converted to numeric types
car$horse_pw <- as.numeric(car$horse_pw)
car$price <- as.numeric(car$price)


#the  number of na values are then determined using sum collaborated with is.na()
sum(is.na(car$horse_pw))
sum(is.na(car$price))



#the tidyverse library is loaded to clean the dataset ie.remove na values 
library("tidyverse")

#the drop_na() is used to drop all the rows containing na values making easier for further
#analysis
car <- car %>% drop_na()


install.packages("caret")
#the library caret is loaded to use the latticeplot to know the type of relationship between
#the two variables
library("caret")

#the featureplot takes in the two variables x and y where x=horsepw and y=price
featurePlot(x=car$horse_pw,y=car$price)

#note: from the plot we could get the insights that, there is a positive relationship 
#between feature and price. As the horsepw increases in x axis  the price increases in y axis


#createdatapartition fuction is used to partition the data into 80% of training_dataset
train_indices <- createDataPartition(y=car$price,
                                     p=0.8,list = FALSE)
train_list <- car[train_indices,]
train_list
test_list <- car[-train_indices,]
test_list

#the trainControl() is used to specify the type of cross validation is used in training data
train_control <- trainControl(method="cv",number = 5)

#the expand.grid() is used in hyperparameter optimization so that optimum k value is found
#for the specific data containing specific pattern.
hyperparameter_grid <- expand.grid(k=1:5)

#the model is built using the price and horsepw,the method here used is knn along with
#preprocess parameter,train_control and gridsearchcv for hyperparmeter tuning
knn_model <- train(price~horse_pw,data=train_list,
                   method="knn",
                  trControl=train_control,
                  preProcess=c("center","scale"),
                  tuneGrid=hyperparameter_grid)

#the price is predicted using test_list which was the result of splitting the data along
#with the model built previously
test_pred <- predict(knn_model,newdata = test_list)

test_pred

#the error is then calculated using observed value of price and predicted value
test_list <- test_list %>% mutate(
  error = price - test_pred
)
test_list$error

#the mean squared error is then calculated using previously obtained error array
test_list <- test_list %>% mutate(
  mse = error^2
)
mean(test_list$mse)

#the root mean squared error is then calculated using previously obtained mse
rmse = sqrt(mean(test_list$mse))
rmse

###postResample(pred = test_pred,obs = test_list$price)
###RMSE             Rsquared          MAE 
###3454.0208209     0.8711627      2341.2591949 


#2nd model with all the independent values such as highway_mpg,city_mpg,peak_rpm,
#compression_ratio,engine_size,horse_pw and the dependent value price
car$horse_pw <- as.numeric(car$horse_pw)
car$price <- as.numeric(car$price)
car$highway_mpg <- as.numeric(car$highway_mpg)
car$city_mpg <- as.numeric(car$city_mpg)
car$peak_rpm <- as.numeric(car$peak_rpm)
car$compression_ratio <- as.numeric(car$compression_ratio)
car$engine_size <- as.numeric(car$engine_size)

#after using drop_na all the na values sum is equal to 0
sum(is.na(car$horse_pw))
sum(is.na(car$price))
sum(is.na(car$highway_mpg))
sum(is.na(car$city_mpg))
sum(is.na(car$peak_rpm))
sum(is.na(car$compression_ratio))
sum(is.na(car$engine_size))

#the new train indices is built from scratch along with split ratio of 50% for train and 
#50% for the test
train_indices1 <- createDataPartition(y=car$price,p=0.5,list = FALSE)
train_list1 <- car[train_indices1,]
train_list1
test_list1 <- car[-train_indices1,]
test_list1

#the model2 is built by using highway_mpg,city_mpg,peak_rpm,
#compression_ratio,engine_size,horse_pw and dependent variable of price
knn_model2 <- train(price~horse_pw+highway_mpg+city_mpg+peak_rpm+compression_ratio+
                      engine_size,data=train_list1,
                   method="knn",
                   trControl=train_control,
                   preProcess=c("center","scale"),
                   tuneGrid=hyperparameter_grid)

#the new prices are now predicted using the test_list1 and knn_model2
test_pred2 <- predict(knn_model2,newdata = test_list1)
test_pred2

#the error is then calculated using observed value of price and predicted value
test_list1 <- test_list1 %>% mutate(
  error = price - test_pred2
)
test_list1$error

#the mean squared error is then calculated using previously obtained error array
test_list1 <- test_list1 %>% mutate(
  mse = error^2
)
mean(test_list1$mse)

#the root mean squared error is then calculated using previously obtained mse
rmse1 = sqrt(mean(test_list1$mse))
rmse1

#postResample(pred = test_pred2,obs = test_list1$price)
#RMSE              Rsquared          MAE 
#3605.9024119    0.8432264      2232.6683673 

#the correlation package is installed to detect the correlation of dependent wrt independent
install.packages("corrr")
x <- car[21:25]
y <- car[26]
cor(x,y,use = "complete.obs")


#3rd model with only low correlated variables peak_rpm,city_mpg,compression_ratio,highway_mpg
train_indices3 <- createDataPartition(y=car$price,
                                      p=0.5,list = FALSE)
#the new train indices is built from scratch along with split ratio of 50% for train and 
#50% for the test
train_list2 <- car[train_indices3,]
train_list2
test_list2 <- car[-train_indices3,]
test_list2

#the model3 is built by using highway_mpg,city_mpg,peak_rpm,
#compression_ratio, and dependent variable of price
knn_model3 <- train(price~highway_mpg+city_mpg+peak_rpm+compression_ratio
                    ,data=train_list2,
                    method="knn",
                    trControl=train_control,
                    preProcess=c("center","scale"),
                    tuneGrid=hyperparameter_grid)

#the new prices are now predicted using the test_list2 and knn_model3
test_pred3 <- predict(knn_model3,newdata = test_list2)
test_pred3

#the error is then calculated using observed value of price and predicted value
test_list2 <- test_list2 %>% mutate(
  error = price - test_pred3
)
test_list2$error

#the mean squared error is then calculated using previously obtained error array
test_list2 <- test_list2 %>% mutate(
  mse = error^2
)
mean(test_list2$mse)

#the root mean squared error is then calculated using previously obtained mse
rmse2 = sqrt(mean(test_list2$mse))
rmse2

#postResample(pred = test_pred3,obs = test_list2$price)
#RMSE              Rsquared          MAE 
#4150.0092616    0.7327355      2246.8297619 



#4th model with only few low correlated values such as highway_mpg,city_mpg,peak_rpm and
#data is split into 80% training set and 20% testing set
train_indices4 <- createDataPartition(y=car$price,
                                      p=0.8,list = FALSE)

#the new train indices is built from scratch along with split ratio of 80% for train and 
#20% for the test
train_list3 <- car[train_indices4,]
train_list3
test_list3 <- car[-train_indices4,]
test_list3

#the model4 is built by using highway_mpg,city_mpg,peak_rpm,
#and dependent variable of price
knn_model4 <- train(price~highway_mpg+city_mpg+peak_rpm
                    ,data=train_list3,
                    method="knn",
                    trControl=train_control,
                    preProcess=c("center","scale"),
                    tuneGrid=hyperparameter_grid)

#the new prices are now predicted using the test_list3 and knn_model4
test_pred4 <- predict(knn_model4,newdata = test_list3)
test_pred4

test_list3 <- test_list3 %>% mutate(
  error = price - test_pred4
)
test_list3$error

test_list3 <- test_list3 %>% mutate(
  mse = error^2
)
mean(test_list3$mse)

rmse3 = sqrt(mean(test_list3$mse))
rmse3

#postResample(pred = test_pred4,obs = test_list3$price)
#RMSE             Rsquared          MAE 
#2708.6433326    0.9258342      1883.0436404 

featurePlot(x=car$highway_mpg,y=car$price) #negatively correlated
featurePlot(x=car$city_mpg,y=car$price) # negatively correlated
featurePlot(x=car$peak_rpm,y=car$price) #no correlation at all


##  RMSE             Rsquared          MAE          mse
###3454.0208209     0.8711627      2341.2591949   119,30,260
###3605.9024119     0.8432264      2232.6683673   130,02,532
###4150.0092616     0.7327355      2246.8297619   172,22,577
###2708.6433326     0.9258342      1883.0436404   73,36,749

#higher the mse,lower the r**2 poorer the model

library("ggplot2")
ggplot(data = car, 
       aes(x=horse_pw,y=price))+
      geom_line(position = "identity")+
      theme(panel.background = element_rect(fill = "white"))

#from the graph it is seen that as the horsepower is less, the price is less.But,
#as the horsepower increases, price also increases hence we can conclude that both the 
#variables are positively correlated to each other.
