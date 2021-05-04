source("create_csv.R")
source("preprocess.R")
source("build_model.R")
source("tune.R")


suppressPackageStartupMessages(library("optparse"))
suppressPackageStartupMessages(library("caret"))
suppressPackageStartupMessages(library("e1071"))

option_list <- list(
  make_option("--generate", action="store_true",  help = "Generate train and test files", default=FALSE),
  make_option("--experimenttag", action="store", type="character", default="default-experiment", help = "Set experiment tag id "),
  make_option("--modelid", action="store", type="numeric", default=1, help = "Select between different models"),
  make_option("--list-available-models", action="store_true", help = "List different models", dest="list_models",default=FALSE),
  make_option("--tune", action="store_true", help = "Tune the selected model",default=FALSE),
  make_option("--testonly", action="store_true", help = "Bypass training and test with previous weights",default=FALSE),
  make_option("--maxlen", action="store", type="numeric", default=45, help = "Set the maximun length of the domain name considered"),
  make_option("--modelfile", action="store", type="character", help = "A file to load model from"),
  make_option("--testfile", action="store", type="character", help = "A file to load test data from"),
  make_option("--datafile", action="store", type="character", help = "A file to load dataset from", default = "ctu19subs.csv"),
  make_option("--upsample", action="store_true", help = "Apply oversampling to  train dataset",default=FALSE)
  
)
opt <- parse_args(OptionParser(option_list=option_list))
source("config.R")
source("evaluate.R")


#set.seed(12121) # For ensuring repeatibility (not working really)
# tensorflow session setup
#library(tensorflow)
#get_session<-function(gpu_fraction=0.333){
#  gpu_options = tf$GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
#                              allow_growth=TRUE)
#return (tf$Session(config=tf$ConfigProto(gpu_options=gpu_options)))
#}

#keras::k_set_session(get_session())



## MAIN Section                                                                                 #####

maxlen=opt$maxlen         # the maximum length of the domain name considerd for input of the NN

if (opt$list_models){
  print (names(funcs))
  quit()
}

### Test a previouysly saved model                                        ####
if (opt$testonly){
  message("[] Evaluating model on testset")
  model<-load_model_hdf5(opt$modelfile) #TODO check missing
  if (!is.null(opt$testfile)){
    testset<-read_valid_csv(opt$testfile)
    if(!is.null(testset)){
  	 message("[] Tokenizing testset")
     test_dataset_keras<-build_dataset(as.matrix(testset),opt$maxlen)
	   test_dataset_x<-test_dataset_keras$encode
	   test_dataset_y<-test_dataset_keras$class
    }else{ 
      message("[] Error:")
      quit() }
  }else{
 	 load(file='datasets/.test_dataset_keras.rd')
	 test_dataset_x<-test_dataset_keras$encode
	 test_dataset_y <- test_dataset_keras$class
  }
  results<-evaluate_model_test(model,test_dataset_x,test_dataset_y,test_dataset_keras$label)
  message("[] Saving results ")
  write_csv(results$result,col_names = T,path=paste(results_dir,"results_test_",opt$experimenttag,".csv",sep=""))
  write_csv(results$resultperclass,col_names = T,path=paste(results_dir,"results_per_subclass_test_",opt$experimenttag,".csv",sep=""))
  quit()
}

#### Generate new tokenized datasets from .csv files ##########
if (!file.exists(paste0(datasets_dir,".",dataset_default, "_train_dataset_keras.rd"))){
	message(" []  train and test files not found. Generating")
	opt$generate<-TRUE
}

#### Generate new datasets from csv or load previously generated R objects #######
if ( opt$generate){
  message("[] Generating Datasets")
  dataset_csv<-paste(datasets_dir,dataset_default,sep="")
  dataset_df<-read_valid_csv(dataset_csv)
  if(!is.null(dataset_df)){
    datasets<-build_train_test(dataset_df,opt$maxlen,upsample=opt$upsample)
    train_dataset_keras<-datasets$train
    test_dataset_keras<-datasets$test
  }else{ 
    message("[] Error:")
    quit() }
} else {
  message("[] Loading Datasets ")
  load(file=paste0(datasets_dir,".",dataset_default,"_train_dataset_keras.rd"))
  load(file=paste0(datasets_dir,".",dataset_default,"_test_dataset_keras.rd"))
}


###### Tune model hyperparameters, select the best model and save CV results ####
if (opt$tune){
  message("[] Tuning model hyperparameters")
  models_results<-tune_model(dataset=train_dataset_keras,modelid=opt$modelid,experimentname=opt$experimenttag )
  write_csv(models_results,col_names = T,path=paste(results_dir,"results_tuning_",opt$experimenttag,".csv",sep=""))
  selected_parameters<-models_results %>% arrange(desc(value.F1)) %>% select(-value.F1,-value.sd) %>% head(1)
  message("[] Crossvalidation on best model")
  message(paste("Using",selected_parameters))
  results<-evaluate_model_cv(modelfun=funcs[[opt$modelid]],model_parameters=selected_parameters,data=train_dataset_keras,k=10,experimentname = opt$experimenttag)
  message("[] Saving results of best model")
  names(results$result)<-c("k","metric","value")
  write_csv(results$result,col_names = T,path=paste(results_dir,"results_tuning_best_cv",opt$experimenttag,"_cv.csv",sep=""))
  write_csv(results$resultperclass,col_names = T,path=paste(results_dir,"results_per_subclass_tuning_best_cv",opt$experimenttag,"_cv.csv",sep=""))
  quit()
}

### Train and test a model ####
message("[] Creating model and evaluating model on test ")
selected_parameters<- 
  eval(
    parse(
      text=paste("default_keras_model_",names(funcs)[opt$modelid],"_parameters",sep="") # TODO: verify existence
    )
  )
#results<-evaluate_model_cv(modelfun=funcs[[opt$modelid]],model_parameters=selected_parameters,data=train_dataset_keras,k=5,experimentname = opt$experimenttag)
results<-evaluate_model_train_test(train_dataset_keras,test_dataset_keras,modelfun=funcs[[opt$modelid]], selected_parameters,opt$experimentname)
message("[] Saving results ")
write_csv(results$result,col_names = T,path=paste(results_dir,"results_test_",opt$experimenttag,".csv",sep=""))
write_csv(results$resultperclass,col_names = T,path=paste(results_dir,"results_per_subclass_test_",opt$experimenttag,".csv",sep=""))
quit()

