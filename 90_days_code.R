library(xgboost)
s<-read.csv("trainyy90.csv")
g<-cbind(s$fersi,s$dpsch,s$dpwill,s$MACD,s$pricerate90,s$OBV)
b<-xgboost(data = g , label = s$label90, missing = NULL, params = list(),
           nrounds=25, verbose = 1, print.every.n = 1L, early.stop.round = NULL,
           maximize = NULL)
r<-read.csv("testyy90.csv")
pred90 <- predict(b,as.matrix(r))
print(pred90)

p<-read.csv("predictedxgyy90.csv")
pred <- prediction(p$predicted,p$ACTUAL)
perf <- performance(pred, "tpr", "fpr")

plot(perf, colorize=T)
abline(a=0,b=1)
k<-attributes(performance(pred,'auc'))$y.values[[1]]
print(k)
#--------------------------------------------------------------

library(xgboost)
y_train<-as.numeric(as.factor(s$label90))-1
label_train<-data.frame(s$label90,y_train)
xgb_train<-xgb.DMatrix(model.matrix(~fersi+dpsch+dpwill+MACD+pricerate90+OBV,data=s),label=y_train,missing=NA)
cv<-xgb.cv(data=xgb_train,nrounds = 5,nfold=10,metrices='mlogloss',num_class=5,objective='multi:softprob',verbose = T,
stratified = TRUE,prediction = T)
pred.cv=matrix(cv$pred,nrow=nrow(s),ncol=5)
predicted<-max.col(pred.cv,"last")
cm<-table(predicted,s$label90)
