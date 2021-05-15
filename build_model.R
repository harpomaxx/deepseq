library(keras)
funcs<-list()
#source(list.files(pattern = "model_.*.R"))
source("model_cnn_argencon.R")#1
source("model_cnn_pablo.R") 
source("model_lstm_endgame.R")
source("model_lstm_custom.R")
source("model_mc_cnn_asai.R") #5
source("model_dense.R")
source("model_lstm_endgame_norm.R")
source("model_lstm_endgame_bi.R")
source("model_lstm_endgame_recurrent_dropout.R")
source("model_lstm_endgame_bi_att.R") #10

# list for selecting between different models
#funcs<-list( cnn_argencon=keras_model_cnn_argencon,
#              cnn_pablo=keras_model_cnn_pablo,
#              lstm_endgame=keras_model_lstm_endgame
#             )

#library(tensorflow)
#get_session <- function(gpu_fraction = 0.333) {
#  keras::k_clear_session()
#  gpu_options = tf$GPUOptions(per_process_gpu_memory_fraction = gpu_fraction,
#                              allow_growth = TRUE)
#  return (tf$Session(config = tf$ConfigProto(gpu_options = gpu_options)))
#}
#keras::k_set_session(get_session())



# Train model
train_model <- function(x,y, model,ep=60,modelname="model"){
  #message(y %>% head(5))
  keras::k_clear_session()
  es <- callback_early_stopping(monitor='val_loss', 
                                mode='min', 
                                patience = 15,
                                min_delta = 0.001)
  tb <- callback_tensorboard(paste0("logs/eval_models"))
  history<-model %>% fit(x,y,epochs = ep, batch_size = 256, validation_split = 0.2,verbose = 0,
                         callbacks=list(es,tb))
 # model %>% save_model_hdf5(paste(modelname,".h5",sep=""))
  system2(c("rm","-Rf","/home/gab/deepseq/logs/eval_models"))
  return(list(model=model,history=history))
}

