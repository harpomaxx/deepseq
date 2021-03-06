# keras model used in ENDGAME LSTM 2016 paper +  batch normalization layer

default_keras_model_lstm_endgame_norm_parameters_tune=list(
  lstm_size = c(128,64,32),
  embedingdim = c(128,50,32),
  dropout = c(0.5)
)

#default_keras_model_cnn_argencon_parameters_tune=list(
#  nb_filter = c(256,128),
#  kernel_size = c(8),
#  embedingdim = c(100),
#  hidden_size = c(1024)
#)



default_keras_model_lstm_endgame_norm_parameters=list(
  embedingdim = 128,
  lstm_size = 128,
  dropout = 0.5
)


keras_model_lstm_endgame_norm<-function(x,parameters=default_keras_model_lstm_endgame_norm_parameters){
  
  input_shape <- dim(x)[2]
  inputs<-layer_input(shape = input_shape) 
  
  embeding<- inputs %>% layer_embedding(length(valid_characters_vector), parameters$embedingdim , input_length = input_shape,mask_zero=T)
  
  lstm <- embeding %>%
    layer_lstm(units = parameters$lstm_size) %>%
    layer_batch_normalization(axis = -1) # the feature axis %>%
    layer_dropout(rate = parameters$dropout) %>%
    layer_dense(1, activation = 'sigmoid')
  
  #compile model
  model_endgame_norm <- keras_model(inputs = inputs, outputs = lstm)
  model_endgame_norm %>% compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = c('accuracy')
  )
  summary(model_endgame_norm)
  return(model_endgame_norm)
}

funcs[["lstm_endgame_norm"]]=keras_model_lstm_endgame_norm
