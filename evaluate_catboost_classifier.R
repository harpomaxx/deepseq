suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(catboost))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(tibble))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library("optparse"))
suppressPackageStartupMessages(library("caret"))
suppressPackageStartupMessages(library("e1071"))

read_valid_csv<-function(data_csv){
  
  type_csv <- c("character","integer","character")
  data<-readr::read_csv(data_csv,col_types  =cols(
    State = col_character(),
    class = col_integer(),
    label = col_character()
  ))
  data_type_csv<-sapply(data,class)
  if (length(data_type_csv)!=3 || (FALSE %in% (data_type_csv == type_csv %>% as.list()))) { 
    message("[] Wrong file schema. The schema should be <sequence, class, label>")
    return (NULL)
  }
  return (data)
}

vectorize_seq <-function(dataset,maxlen){
#dataset_vectorized <- as.data.frame(as.character(dataset),stringsAsFactors=FALSE)
#names(dataset_vectorized)<-c("State")
dataset_vectorized <-as_tibble(dataset)
dataset_vectorized<- dataset_vectorized %>% mutate(modelsize=str_count(State,"."))
if (maxlen > 0){
  dataset_vectorized$State <- dataset_vectorized$State %>% substr(1,maxlen)
  dataset_vectorized<-dataset_vectorized %>% mutate(modelsize=nchar(State))
}
#Periodicity
dataset_vectorized = dataset_vectorized %>% mutate(strong_p = str_count(State,'[a-i]'))
dataset_vectorized = dataset_vectorized %>% mutate(weak_p = str_count(State,'[A-I]'))
dataset_vectorized = dataset_vectorized %>% mutate(weak_np = str_count(State,'[r-z]'))
dataset_vectorized = dataset_vectorized %>% mutate(strong_np = str_count(State,'[R-Z]'))
#Duration
dataset_vectorized = dataset_vectorized %>% mutate(duration_s = str_count(State,'(a|A|r|R|1|d|D|u|U|4|g|G|x|X|7)'))
dataset_vectorized = dataset_vectorized %>% mutate(duration_m = str_count(State,'(b|B|s|S|2|e|E|v|V|5|h|H|y|Y|8)'))
dataset_vectorized = dataset_vectorized %>% mutate(duration_l = str_count(State,'(c|C|t|T|3|f|F|w|W|6|i|I|z|Z|9)'))
#Size
dataset_vectorized = dataset_vectorized %>% mutate(size_s = str_count(State,'[a-c]') + str_count(State,'[A-C]') + str_count(State,'[r-t]') + str_count(State,'[R-T]') + str_count(State,'[1-3]'))
dataset_vectorized = dataset_vectorized %>% mutate(size_m = str_count(State,'[d-f]') + str_count(State,'[D-F]') + str_count(State,'[u-w]') + str_count(State,'[U-W]') + str_count(State,'[4-6]'))
dataset_vectorized = dataset_vectorized %>% mutate(size_l = str_count(State,'[g-i]') + str_count(State,'[G-I]') + str_count(State,'[x-z]') + str_count(State,'[X-Z]') + str_count(State,'[7-9]'))

#Periodicity %
dataset_vectorized <- dataset_vectorized %>% mutate(strong_p = (strong_p / modelsize))
dataset_vectorized <- dataset_vectorized %>% mutate(weak_p = (weak_p / modelsize))
dataset_vectorized <- dataset_vectorized %>% mutate(strong_np = (strong_np / modelsize))
dataset_vectorized <- dataset_vectorized %>% mutate(weak_np = (weak_np / modelsize))
#Duration %
dataset_vectorized <- dataset_vectorized %>% mutate(duration_s = (duration_s / modelsize))
dataset_vectorized <- dataset_vectorized %>% mutate(duration_m = (duration_m / modelsize))
dataset_vectorized <- dataset_vectorized %>% mutate(duration_l = (duration_l / modelsize))
#Size %
dataset_vectorized <- dataset_vectorized %>% mutate(size_s = (size_s / modelsize))
dataset_vectorized <- dataset_vectorized %>% mutate(size_m = (size_m / modelsize))
dataset_vectorized <- dataset_vectorized %>% mutate(size_l = (size_l / modelsize))

#Making feature vectors
dataset_vectorized <- dataset_vectorized %>% select('strong_p','weak_p','weak_np','strong_np','duration_s',
                                                    'duration_m','duration_l','size_s','size_m','size_l','modelsize','class','label')

names(dataset_vectorized) <- c("sp","wp","wnp","snp","ds","dm","dl","ss","sm","sl","modelsize",'class','label')


#dataset_vectorized<-dataset_vectorized %>% mutate(class=ifelse(grepl(pattern = "Normal", x = label),0,1))
#dataset_vectorized$class<-as.factor(dataset_vectorized$class)
#
dataset_vectorized %>% group_by(class) %>% summarize(tot=n())
dataset_vectorized
}

train_test_sample<-function(x,percent=0.7){
  smp_size <- floor(percent * nrow(x))
  train_ind <- sample(seq_len(nrow(x)), size = smp_size)
  return (train_ind)
}

### Function Definitions ####
get_predictions <- function(model, pool_test,threshold=0.5) {
  predsprobs<- catboost.predict(model,pool_test,prediction_type='Class')
  #preds<-ifelse(predsprobs[,1]>0.5,0,1)
  preds<-predsprobs
  return (preds)
}

calculate_recall <-function(dataset){
  recall<-dataset %>% group_by(label) %>% summarise(recall=sum(predicted_class==class)/n(),support=n()) 
  return(recall)
}

evaluate_model_test <- function(model, test_dataset_x, test_dataset_y, pool_test, original_labels) {
  preds<-get_predictions(model,pool_test)
  confmatrix<-confusionMatrix(data= as.factor(preds),reference = as.factor(test_dataset_y),positive = '1', mode="everything")
  print(confmatrix)
  result<-cbind(value=as.data.frame(confmatrix$byClass) %>% rownames_to_column())
  recall<-calculate_recall(data.frame(label=original_labels, class=test_dataset_y,predicted_class=preds))
  result_per_subclass<-cbind(recall)
  names(result)<-c("metric","value")
  return (list(result=result, resultperclass=result_per_subclass))
}

#### MAIN 

option_list <- list(
  make_option("--experimenttag", action="store", type="character", default="default-experiment", help = "Set experiment tag id "),
  make_option("--maxlen", action="store", type="numeric", default=45, help = "Set the maximun length of the seq  considered"),
  make_option("--upsample", action="store_true", help = "Apply oversampling to  train dataset",default=FALSE),
  make_option("--downsample", action="store_true", help = "Apply downsampling to  train dataset",default=FALSE),
  make_option("--datafile", action="store", type="character", help = "A file to load dataset from", default = "train.csv"),
  make_option("--testfile", action="store", type="character", help = "A file to load dataset from", default = "test.csv")
)

  


opt <- parse_args(OptionParser(option_list=option_list))

print(opt)

datasetfile=opt$datafile
results_dir='../dga-wb-r/results/'




dataset<-read_valid_csv(datasetfile)
if(is.null(dataset)){
  message("Error!")
  quit()
}
dindex<-train_test_sample(dataset,0.7)
train_dataset<-dataset[dindex,]
test_dataset<-dataset[-dindex,]


if (opt$upsample){
  train_dataset<- caret::upSample(x=train_dataset[,c(1,3)], y=as.factor(train_dataset$class),list = F,yname = "class") 
  train_dataset<-train_dataset[,c(1,3,2)]
  print(train_dataset %>% head(5))
}

if (opt$downsample){
  train_dataset<- caret::downSample(x=train_dataset[,c(1,3)], y=as.factor(train_dataset$class),list = F,yname = "class") 
  train_dataset<-train_dataset[,c(1,3,2)]
  print(train_dataset %>% head(5))
}

dataset_train_vectorized<-vectorize_seq(train_dataset,opt$maxlen)
dataset_test_vectorized<-vectorize_seq(test_dataset,opt$maxlen)
print("[] Creating model and evaluating model on test ")

dataset_train_vectorized_x<-dataset_train_vectorized %>% select(-class, -label, -modelsize)
dataset_train_vectorized_y<-dataset_train_vectorized$class

dataset_test_vectorized_x<-dataset_test_vectorized %>% select(-class, -label, -modelsize)
dataset_test_vectorized_y<-dataset_test_vectorized$class

pool_train <- catboost.load_pool( dataset_train_vectorized_x, 
                                  label = as.numeric(dataset_train_vectorized_y))

pool_test <- catboost.load_pool( dataset_test_vectorized_x, 
                                  label = as.numeric(dataset_test_vectorized_y)) 

fit_params <- list(iterations=100,
                   loss_function = 'Logloss',
                   task_type = 'CPU')

rfModel<-  catboost.train(pool_train, params = fit_params)

results<-evaluate_model_test(rfModel,
                             test_dataset_x = dataset_test_vectorized_x, 
                             test_dataset_y = dataset_test_vectorized_y, 
                             pool_test = pool_test,
                             original_labels = dataset_test_vectorized$label)

print("[] Saving results ")
write_csv(results$result,col_names = T,path=paste(results_dir,"results_test_catboost_",opt$experimenttag,".csv",sep=""))
write_csv(results$resultperclass,col_names = T,path=paste(results_dir,"results_per_subclass_test_catboost_",opt$experimenttag,".csv",sep=""))
