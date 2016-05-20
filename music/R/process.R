#! /usr/bin/R

data <- read.csv('allplay/play/0c80008b0a28d356026f4b1097041689.csv', header = F, col.names =list('ind','id', 'plays', 'dates'))
plays <- data$plays
train <- data.frame(plays, row.names = data$dates)
fit <- arima(train, order = c(0,1,1))
library(forecast)
pre <- forecast(fit, 60)
print(pre)

fit <- arima(train, order=c(2, 0, 1))
pre <- forecast(fit, 60)
print(pre)



## randomforest
data <- read.csv('allplay/play/0c80008b0a28d356026f4b1097041689.csv', header = F, col.names =list('ind','id', 'plays', 'dates'))
plays <- data$plays
train <- data.frame(plays, row.names = data$dates)

rrf <- randomForest(train, )