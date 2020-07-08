##### deepseq basic configuration

### Set the valid characters. Required for tokenizer function ###
# for DGA
#valid_characters <- "$abcdefghijklmnopqrstuvwxyz0123456789-_.ABCDEFGHIJKLMNOPQRSTUVWXYZ+*,\""
# for Slips stratosphere models
valid_characters <- "$abcdefghiABCDEFGHIrstuvwxyzRSTUVWXYZ0123456789.,+*"

## Configuration paths ####
results_dir='/home/harpo/deepactivelearning/results/'
models_dir='/home/harpo/deepactivelearning/models/'
datasets_dir='/home/harpo/deepactivelearning/datasets/'
#dataset_default='JISA2018.csv.gz'
#dataset_default='argencon_vaclav.csv.gz'
dataset_default=opt$datafile
#dataset_default='ctu13subs.csv'
#dataset_default='train_combined_multiclass.csv.gz'




valid_characters_vector <- strsplit(valid_characters,split="")[[1]]
tokens <- 0:length(valid_characters_vector)
names(tokens) <- valid_characters_vector



