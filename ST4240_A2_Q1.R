#### Assignment 2 Question 1

#### Packages required
devtools::install_github("jrowen/twitteR", ref = "oauth_httr_1_0")
require(twitteR)
require(plyr)
require(dplyr)
require(tm)
require(textstem)
require(text2vec)
require(tidyverse)
require(tidytext)
require(glue)
require(data.table)

# Set your working directory first and make sure trump_raw.RDS and clinton_raw.RDS are in them.
setwd("/Users/Qinli/Google Drive/ST4240/")

#### Obtaining tweet data from twitter; output has been saved in RDS files. Ignore this section.
# consumer_key <- "m3Q1Go2rCB6rrkXadkhYwmcDK"
# consumer_secret <- "6fCnM2dhEgqgOf2rzA6BW7YtQx9k7VU0JM7MmjC5QUXcttui3S"
# access_token <- "	980854628794290176-UUlxLQXZSVSPpOz4z5uSQw73eQ3JVop"
# access_secret <- "Mtk9CzftgbyHu9lvxkgI0pMib7sfHR4b2RvmEGUdBYovM"
# setup_twitter_oauth(consumer_key, consumer_secret)
# trump_raw = twListToDF(userTimeline("realDonaldTrump", n = 3200, excludeReplies = TRUE))
# clinton_raw = twListToDF(userTimeline("HillaryClinton", n = 3200, excludeReplies = TRUE))


#### Initializing data
trump_raw = readRDS("trump_raw.RDS") %>% select(id, screenName, text)
clinton_raw = readRDS("clinton_raw.RDS") %>% select(id, screenName, text)
full = rbind(trump_raw, clinton_raw)
remap = c("realDonaldTrump" = 1, "HillaryClinton" = 0)
full$screenName = revalue(full$screenName, remap)
full$screenName = as.integer(full$screenName)

n_total = nrow(full)
n_train = floor(n_total * 0.7)  #Using 70% of dataset for training
train_ids = sample(n_total, n_train, replace = FALSE)
test_ids = (1:n_total)[-train_ids]
train = full[train_ids,]
test = full[test_ids,]

#### Creating DTM matrix

# Cleaning the tweets
word_processing = function(x) {
  process_sentence = function(ss){
    ss = ss[!str_detect(ss,"[^[:alpha:]]")]  # get rid of words that contains non-letter characters
    return(lemmatize_words(tolower(ss)))     # lower case + lemmatize
  }
  tokens = word_tokenizer(x) %>% lapply(process_sentence)
  return( tokens )
}

train_tokens = word_processing(train$text)
test_tokens = word_processing(test$text)

it_train = itoken(train_tokens, 
                  ids = train$screenName, 
                  progressbar = FALSE)

it_test = itoken(test_tokens, 
                 ids = test$screenName, 
                 progressbar = FALSE)

stop_words_to_use = c(tolower(stop_words$word), "https", "amp")
vocab = create_vocabulary(it_train, stopwords = stop_words_to_use)
pruned_vocab = prune_vocabulary(vocab, term_count_min = 5)

vectorizer_pruned = vocab_vectorizer(pruned_vocab)
dtm_train = create_dtm(it_train, vectorizer_pruned)
dtm_test = create_dtm(it_test, vectorizer_pruned)

tfidf = TfIdf$new()
dtm_train_tfidf = fit_transform(dtm_train, tfidf) # tfidf modified by fit_transform() call!
dtm_test_tfidf  = dtm_test %>% transform(tfidf)

library(glmnet)
NFOLDS = 5
glmnet_classifier = cv.glmnet(x = dtm_train_tfidf, 
                              y = train$screenName, 
                              family = 'binomial', 
                              # L1 penalty
                              alpha = 1,
                              # interested in the area under ROC curve
                              type.measure = "auc",
                              # 5-fold cross-validation
                              nfolds = NFOLDS,
                              # high value is less accurate, but has faster training
                              thresh = 1e-3,
                              # again lower number of iterations for faster training
                              maxit = 1e4)

plot(glmnet_classifier)

#### Making predictions
preds = predict(glmnet_classifier, dtm_test_tfidf, type = 'response')[,1]
auc_test = glmnet:::auc(test$screenName, preds)
cat("AUC =", auc_test, "\n")

suppressMessages(library(pROC))
plot.roc(test$screenName, 
         preds, 
         col = "firebrick1", lwd = 2, print.auc=TRUE, 
         print.auc.y = 0.2,
         main = "Prediction for Tweets")

c = as.matrix( coefficients(glmnet_classifier) )
c_summary = data.frame(var = rownames(c), coef = c[,1])
print(c_summary %>% arrange(desc(coef)))
print(c_summary %>% arrange(coef))

error_rate = mean((test$screenName == 1) != (preds > 0.5) )
cat("Error rate = ", error_rate, "\n")
