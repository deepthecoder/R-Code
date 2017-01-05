
library(xgboost)
library(ROCR)
s<-read.csv("train28xg.csv")
g<-cbind(s$fersi,s$dpsch,s$dpwill,s$MACD,s$OBV)
b<-xgboost(data = g , label = s$label28, missing = NULL, params = list(),
           nrounds=25, verbose = 1, print.every.n = 1L, early.stop.round = NULL,
           maximize = NULL,ntree=10)
r<-read.csv("testxg28.csv")
pred28 <- predict(b,as.matrix(r))
print(pred28)

p<-read.csv("predictxg28.csv")
pred <- prediction(p$predicted,p$ACTUAL)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=T)
abline(a=0,b=1)
k<-attributes(performance(pred,'auc'))$y.values[[1]]
print(k)

#--------------------------------------------------------


library(xgboost)
y_train<-as.numeric(as.factor(s$label28))-1
label_train<-data.frame(s$label28,y_train)
xgb_train<-xgb.DMatrix(model.matrix(~fersi+dpsch+pricerate+dpwill+MACD+OBV,data=s),label=y_train,missing=NA)
cv<-xgb.cv(data=xgb_train,nrounds = 5,nfold=10,metrices='mlogloss',num_class=5,objective='multi:softprob',verbose = T,
stratified = TRUE,prediction = T)
pred.cv=matrix(cv$pred,nrow=nrow(s),ncol=5)
predicted<-max.col(pred.cv,"last")
cm<-table(predicted,s$label28)
