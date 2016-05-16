#! /usr/bin/R

data <- read.csv('25739ad1c56a511fcac86018ac4e49bb.csv', header = F, col.names =list('id', 'plays', 'dates'))
plays <- data$plays
train <- data.frame(plays, row.names = data$dates)
fit <- arima(train, order = c(0,1,1))
library(forecast)
pre <- forecast(fit, 60)
print(pre)