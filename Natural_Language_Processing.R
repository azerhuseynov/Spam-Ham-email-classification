library(tidyverse)
library(text2vec)
library(caTools)
library(glmnet)

df <- read_csv("emails.csv")

for (i in 1:nrow(df)){
  df$id[i] = i
} 

df <- df %>% select(id, everything())

# Split data
set.seed(123)
split <- df$spam %>% sample.split(SplitRatio = 0.8)
train <- df %>% subset(split == T)
test <- df %>% subset(split == F)

it_train <- train$text %>% 
  itoken(preprocessor = tolower, 
         tokenizer = word_tokenizer,
         ids = train$id,
         progressbar = F) 

stop_words <- c("i", "you", "he", "she", "it", "we", "they",
                "me", "him", "her", "them",
                "my", "your", "yours", "his", "our", "ours",
                "myself", "yourself", "himself", "herself", "ourselves",
                "the", "a", "an", "and", "or", "on", "by", "so",
                "from", "about", "to", "for", "of", 
                "that", "this", "is", "are")
vocab <- it_train %>% create_vocabulary(stopwords = stop_words)

vectorizer <- vocab %>% vocab_vectorizer()
dtm_train <- it_train %>% create_dtm(vectorizer)

dtm_train %>% dim()
identical(rownames(dtm_train), as.character(train$id))

# Modeling ----
glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['spam']],
            family = 'binomial', 
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,# high value is less accurate, but has faster training
            maxit = 1000)

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")


it_test <- test$text %>% tolower() %>% word_tokenizer()

it_test <- it_test %>% 
  itoken(ids = test$id,
         progressbar = F)

dtm_test <- it_test %>% create_dtm(vectorizer)

preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$spam, preds) %>% round(2)

