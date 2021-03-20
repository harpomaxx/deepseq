# keras model used in ENDGAME LSTM 2016 paper + BI directional
default_keras_model_lstm_endgame_bidirectional_att_parameters_tune=list(
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



default_keras_model_lstm_endgame_bidirectional_att_parameters=list(
  embedingdim = 128,
  lstm_size = 128,
  dropout = 0.5
)


keras_model_lstm_endgame_bidirectional_att<-function(x,parameters=default_keras_model_lstm_endgame_bidirectional_parameters){
  
  input_shape <- dim(x)[2]
  inputs<-layer_input(shape = input_shape) 
  
  embeding<- inputs %>% layer_embedding(length(valid_characters_vector), parameters$embedingdim , input_length = input_shape,mask_zero=F)
  
  lstm <- embeding %>%
    bidirectional(layer_lstm(units = parameters$lstm_size,
                             recurrent_dropout = parameters$dropout, 
                             #batch_input_shape = c(256,256,128),
                             return_sequences = T))
    
    #layer_dropout(rate = parameters$dropout) %>%
    
   att <- lstm %>% layer_dense(1, activation='tanh') %>%
    layer_flatten() %>%
    layer_activation('softmax')
   temp <- att %>% layer_repeat_vector(parameters$lstm_size * 2) %>%
     layer_permute(c(2,1))
   
   output <- layer_multiply(list(temp,lstm))
   print("OK")
   
   output <- output %>% layer_lambda( f = function(x){k_sum(x, axis =1)}) %>%
      layer_dense(1, activation = 'sigmoid')
  
  
  
  #compile model
  model_endgame_bidirectional_att <- keras_model(inputs = inputs, outputs = output)
  model_endgame_bidirectional_att %>% compile(
    optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics = c('accuracy')
  )
  summary(model_endgame_bidirectional_att)
  return(model_endgame_bidirectional_att)
}

funcs[["lstm_endgame_bidirectional_att"]]=keras_model_lstm_endgame_bidirectional_att
