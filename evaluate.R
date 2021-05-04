
get_predictions <- function(model, test_dataset_x,threshold=0.5) {
  predsprobs<-model %>% predict(test_dataset_x, batch_size=256)
  preds<-ifelse(predsprobs>threshold,1,0)
  return (preds)
}

calculate_recall <-function(dataset){
  recall<-dataset %>% group_by(label) %>% summarise(recall=sum(predicted_class==class)/n(),support=n()) 
  return(recall)
}


# Function implemeting k-fold cross validation 
# modelfun: reference to the function that create the keras model
#            data : dataset used for crossvalidation (dataframe)
#               k : the number of folds in k-fold (default 5) (integer)
# model_parameters: a list with the hyper parameters of the models

evaluate_model_cv <- function(data,k=5, modelfun = keras_model_cnn_argencon,model_parameters=NULL,experimentname="default"){
  knum=k
  result=c()
  result_per_subclass=c()
  
  folds <- createFolds(factor(data$label), k = knum, list = FALSE)
  for (k in 1:knum){
    
    train_dataset_x<-data$encode[ which(folds !=k ),]
    train_dataset_y<-data$class[ which(folds !=k)]
    
    test_dataset_x<-data$encode[ which(folds == k),]
    test_dataset_y<-data$class[ which( folds ==k)]
    
    model_learned<-train_model(x=train_dataset_x,
                               y=train_dataset_y,
                               model=modelfun(train_dataset_x,parameters=model_parameters),
                               modelname = paste(opt$experimenttag,"_model.h5",sep="")
    )
    preds<-get_predictions(model_learned$model,test_dataset_x)
    
    confmatrix<-confusionMatrix(as.factor(preds),as.factor(test_dataset_y),positive = '1')
    result<-rbind(result,cbind(k=k,value=as.data.frame(confmatrix$byClass) %>% rownames_to_column()))
    
    #recall<-data.frame(label=data$label[ which( folds ==k)], class=test_dataset_y,predicted_class=preds) %>% 
    #  group_by(label) %>% summarise(recall=sum(predicted_class==class)/n(),support=n()) 
    recall<-calculate_recall(data.frame(label=data$label[ which( folds ==k)], class=test_dataset_y,predicted_class=preds))
    result_per_subclass=rbind(result_per_subclass,cbind(k=k,recall))
    rm(model_learned)
    gc()
    keras::k_clear_session()
  }
  names(result)<-c("k","metric","value")
  return (list(result=result, resultperclass=result_per_subclass))
}



evaluate_model_test <- function(model, test_dataset_x, test_dataset_y, original_labels) {
  preds<-get_predictions(model,test_dataset_x)
  confmatrix<-confusionMatrix(as.factor(preds),as.factor(test_dataset_y),positive = '1')
  result<-cbind(value=as.data.frame(confmatrix$byClass) %>% rownames_to_column())
  recall<-calculate_recall(data.frame(label=original_labels, class=test_dataset_y,predicted_class=preds))
  result_per_subclass<-cbind(recall)
  names(result)<-c("metric","value")
  return (list(result=result, resultperclass=result_per_subclass))
}
# 
evaluate_model_train_test <- function(train_dataset_keras,test_dataset_keras,modelfun = keras_model_cnn_argencon,  model_parameters, experimentname) {
  train_dataset_x<-train_dataset_keras$encode
  #train_dataset_y<-ifelse(grepl("Normal",train_dataset_keras$label) ,0,1)
  train_dataset_y <- train_dataset_keras$class
  test_dataset_x<- test_dataset_keras$encode
  test_dataset_y<- test_dataset_keras$class
  #test_dataset_y<-ifelse(grepl("Normal",test_dataset_keras$label) ,0,1)
  model_learned<-train_model(x=train_dataset_x,
                             y=train_dataset_y,
                             model=modelfun(train_dataset_x,
                                            parameters=model_parameters),
                             modelname=opt$experimenttag
                             
  )
  message("save")
  model_learned$model %>% save_model_tf(paste(models_dir,opt$experimenttag,"_model.tf",sep=""))
  message("OK")
  res<-evaluate_model_test(model_learned$model,test_dataset_x,test_dataset_y,test_dataset_keras$label)
  return (list(result=res$result, resultperclass=res$resultperclass,model_learned=model_learned))
}