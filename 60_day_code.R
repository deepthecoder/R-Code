# xgboost
library(xgboost)
library(ROCR)
s<-read.csv("trainyy60.csv")
g<-cbind(s$fersi,s$dpsch,s$dpwill,s$pricerate60,s$MACD,s$OBV)
b<-xgboost(data = g , label = s$label60, missing = NULL, params = list(),
           nrounds=25, verbose = 1, print.every.n = 1L, early.stop.round = NULL,
           maximize = NULL,ntree=10)
r<-read.csv("testyy60.csv")
pred60 <- predict(b,as.matrix(r))
print(pred60)

p<-read.csv("predictxgy60.csv")
pred <- prediction(p$PREDICTED,p$ACTUAL)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=T)
abline(a=0,b=1)
k<-attributes(performance(pred,'auc'))$y.values[[1]]
print(k)

#--------------------------------------------------------


library(xgboost)
y_train<-as.numeric(as.factor(s$label60))-1
label_train<-data.frame(s$label60,y_train)
xgb_train<-xgb.DMatrix(model.matrix(~fersi+dpsch+pricerate60+dpwill+MACD+OBV,data=s),label=y_train,missing=NA)
cv<-xgb.cv(data=xgb_train,nrounds = 5,nfold=10,metrices='mlogloss',num_class=5,objective='multi:softprob',verbose = T,
stratified = TRUE,prediction = T)
pred.cv=matrix(cv$pred,nrow=nrow(s),ncol=5)
predicted<-max.col(pred.cv,"last")
cm<-table(predicted,s$label60)

#-------------------------------

B <- c(55,79,81,90)
barplot(B,col="blue",main="comparison",ylim = c(0, 100), ylab="Accuracy(%)", names.arg=c("logistic regression","svm",
"Artificial neural network","xgboost"))
