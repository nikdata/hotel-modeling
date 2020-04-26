# load libraries
library(vroom)
library(tidyverse)
library(rsample)
library(recipes)
library(keras)
library(yardstick)

# load file
hotels <- vroom::vroom(
  file = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv",
  na = c("", " ", "NA", "NULL", 'Undefined')
  )

glimpse(hotels)

# check for missing values

hotels %>%
  purrr::map_df(function(x) sum(is.na(x))) %>%
  tidyr::pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = 'missing'
  ) %>%
  dplyr::arrange(desc(missing)) %>%
  dplyr::mutate(
    percent_missing = round(missing / nrow(hotels),2)
  ) %>%
  dplyr::filter(missing > 0)

# unique values by variable
hotels %>%
  purrr::map_df(function(x) length(unique(x))) %>%
  tidyr::pivot_longer(
    cols = everything(),
    values_to = 'unique_values',
    names_to = 'variable'
  ) %>%
  arrange(desc(unique_values))

# unique values for agent

length(unique(hotels$agent))

# top 5 agents that cancelled

hotels %>%
  group_by(agent) %>%
  summarize(
    total_cancelled = sum(is_canceled)
  ) %>%
  ungroup() %>%
  arrange(desc(total_cancelled)) %>%
  head(5)

# response variable balance

hotels %>%
  dplyr::group_by(is_canceled) %>%
  dplyr::count() %>%
  dplyr::rename('observation_count' = 'n') %>%
  dplyr::mutate(
    pct = round(observation_count / nrow(hotels),2)
  )

# hotels balance

hotels %>%
  group_by(hotel) %>%
  count() %>%
  rename(observation_count = n) %>%
  mutate(
    pct = round(observation_count / nrow(hotels),2)
  )

# create new df
cln_hotel <- hotels

# feature engineering recipe
main_recipe <- recipes::recipe(~.,data = cln_hotel) %>%
  recipes::update_role(is_canceled, new_role = 'outcome') %>%
  recipes::step_naomit(children, country) %>%
  recipes::step_mutate(
    reservation_date = lubridate::ymd(paste0(arrival_date_year,"-",arrival_date_month,"-",arrival_date_day_of_month)),
    day_of_week = lubridate::wday(reservation_date, label = T),
    stay_duration = stays_in_week_nights + stays_in_weekend_nights,
    total_guests = adults + children + babies,
    total_cost = stay_duration * adr,
    meal_included = case_when(meal %in% c('SC','Undefined') ~ 0, TRUE ~ 1),
    city_hotel = ifelse(hotel == 'City Hotel', 1, 0),
    roomtype_different = case_when(reserved_room_type != assigned_room_type ~ 1, TRUE ~ 0)
  ) %>%
  recipes::step_filter(
    total_cost > 0
  ) %>%
  recipes::step_rm(
    agent,
    company,
    arrival_date_year, 
    arrival_date_month, 
    arrival_date_week_number, 
    arrival_date_day_of_month, 
    stays_in_weekend_nights, 
    stays_in_week_nights, 
    adults, 
    children, 
    babies, 
    reservation_status, 
    reservation_status_date, 
    adr, 
    meal, 
    distribution_channel, 
    reservation_date,
    hotel,
    reserved_room_type,
    assigned_room_type
  )

# apply recipe to dataframe

grande_meal <- recipes::prep(main_recipe, data = cln_hotel, strings_as_factors = FALSE)
burrito <- recipes::juice(grande_meal, everything())

glimpse(burrito)

# train/test split

set.seed(8837)
splits <- rsample::initial_split(burrito, prop = 0.8, strata = is_canceled)

df_train <- splits %>% rsample::training()
df_test <- splits %>% rsample::testing()

# detect imbalance

df_train %>%
  group_by(is_canceled) %>%
  count() %>%
  ungroup() %>%
  mutate(
    percent = round(n / sum(n),2)
  )

# recipe for processing dataframes used for modelling

model_recipe <- recipes::recipe(~., data = df_train) %>%
  recipes::update_role(is_canceled, new_role = 'outcome') %>%
  recipes::update_role(meal_included, city_hotel, is_repeated_guest, roomtype_different, new_role = 'pred_binary') %>%
  recipes::step_normalize(recipes::all_numeric(), -recipes::has_role('pred_binary'), -is_canceled) %>%
  recipes::step_string2factor(recipes::has_type(match = 'nominal')) %>%
  recipes::step_ordinalscore(day_of_week) %>%
  recipes::step_integer(recipes::all_nominal()) %>%
  recipes::step_bin2factor(is_canceled) %>%
  recipes::step_downsample(is_canceled) %>%
  recipes::step_factor2string(is_canceled) %>%
  recipes::step_mutate(is_canceled = case_when(is_canceled == 'yes' ~ 1, TRUE ~ 0)) %>%
  recipes::step_nzv(everything()) %>%
  recipes::prep(training = df_train, strings_as_factors = FALSE)

# apply to training dataframe

ds_train <- juice(model_recipe)
glimpse(ds_train)

# check balance on response variable

ds_train %>%
  group_by(is_canceled) %>%
  count() %>%
  ungroup() %>%
  mutate(
    pct = n / sum(n)
  )

# apply recipe to test dataset

new_dftest <- recipes::bake(model_recipe, df_test)
glimpse(new_dftest)

# rearrange datasets

ds_train <- ds_train %>%
  select(
    country,
    customer_type,
    day_of_week,
    market_segment,
    lead_time,
    previous_cancellations,
    booking_changes,
    deposit_type,
    total_of_special_requests,
    stay_duration,
    total_guests,
    total_cost,
    meal_included,
    city_hotel,
    roomtype_different,
    is_canceled
  )

new_dftest <- new_dftest %>%
  select(
    country,
    customer_type,
    day_of_week,
    market_segment,
    lead_time,
    previous_cancellations,
    booking_changes,
    deposit_type,
    total_of_special_requests,
    stay_duration,
    total_guests,
    total_cost,
    meal_included,
    city_hotel,
    roomtype_different,
    is_canceled
  )

# define GPU parameters

use_python('/usr/local/bin/python3')
keras::use_virtualenv("~/python-virtual-environments/deeplearning/")
use_backend('plaidml')

# define embedding sizes

embed_cntry = 50
embed_custtype = 2
embed_day = 4
embed_marketsegment = 4


# define inputs

input_country <- layer_input(shape = 1, name = 'country_embedding_input')
input_custype <- layer_input(shape = 1, name = 'customertype_embedding_input')
input_day <- layer_input(shape = 1, name = 'day_embedding_input')
input_marketsegment <- layer_input(shape = 1, name = 'marketsegment_embedding_input')
input_allvars <- layer_input(shape = 11, name = 'remaining_predictors_input')

# define embeddings

## no more than 50 or number of unique values / 2

cntry_embedding <- layer_embedding(
  input_dim = 155,
  output_dim = embed_cntry,
  input_length = 1,
  name = 'embedding_country'
)

custype_embedding <- layer_embedding(
  input_dim = 4,
  output_dim = embed_custtype,
  input_length = 1,
  name = 'embedding_customertype'
)

dayofweek_embedding <- layer_embedding(
  input_dim = 7,
  output_dim = embed_day,
  input_length = 1,
  name = 'embedding_dayofweek'
)

marketsegment_embedding <- layer_embedding(
  input_dim = 7,
  output_dim = embed_marketsegment,
  input_length = 1,
  name = 'embedding_marketsegment'
)


emb_cntry <- input_country %>%
  cntry_embedding() %>%
  layer_flatten()

emb_custype <- input_custype %>%
  custype_embedding() %>%
  layer_flatten()

emb_dayofweek <- input_day %>%
  dayofweek_embedding() %>%
  layer_flatten()

emb_marketsegment <- input_marketsegment %>%
  marketsegment_embedding() %>%
  layer_flatten()

pred_vars <- layer_concatenate(list(emb_cntry, emb_custype, emb_dayofweek, emb_marketsegment, input_allvars))

# layer definition
h1 <- layer_dense(pred_vars, units = 60, activation = 'relu')
d1 <- layer_dropout(h1, rate = 0.2)
h2 <- layer_dense(pred_vars, units = 30, activation = 'relu')
d2 <- layer_dropout(h2, rate = 0.1)
o1 <- layer_dense(d2, units = 1, activation = 'sigmoid')

# we want class probabilities

# model definition

model <- keras_model(
  inputs = list(input_country, input_custype, input_day, input_marketsegment, input_allvars),
  outputs = o1
)

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

summary(model)

# train model

history <- fit(
  object = model,
  y = ds_train %>% pull(is_canceled),
  x = list(
    as.matrix(ds_train %>% pull(country)), 
    as.matrix(ds_train %>% pull(customer_type)),
    as.matrix(ds_train %>% pull(day_of_week)),
    as.matrix(ds_train %>% pull(market_segment)),
    as.matrix(ds_train %>% select(-country, -customer_type, -day_of_week,-market_segment, -is_canceled))
  ),
  batch_size = 1000,
  epochs = 50,
  validaton_split = 0.3
)

print(history)

plot(history)

# create predictions on test dataset

yhat_class_pred <- predict(
  object = model, 
  x = list(
    as.matrix(new_dftest %>% pull(country)), 
    as.matrix(new_dftest %>% pull(customer_type)),
    as.matrix(new_dftest %>% pull(day_of_week)),
    as.matrix(new_dftest %>% pull(market_segment)),
    as.matrix(new_dftest %>% select(-country, -customer_type, -day_of_week,-market_segment, -is_canceled))
  )
) %>%
  as.vector()

# create tibble for assessing predicted values

test_tbl <- tibble::tibble(
  truth = as.factor(new_dftest$is_canceled) %>% fct_recode(cancelled = '1', not_cancelled = '0'),
  class_prob = yhat_class_pred
) %>%
  dplyr::mutate(
    predicted_class = ifelse(class_prob >= 0.5, 1, 0),
    estimate = as.factor(predicted_class) %>% fct_recode(cancelled = '1', not_cancelled = '0')
  ) %>%
  dplyr::select(-predicted_class)

# metrics: confusion matrix

test_tbl %>%
  yardstick::conf_mat(truth, estimate)

# metrics: accuracy
test_tbl %>%
  yardstick::metrics(truth, estimate) %>%
  dplyr::select(.metric, .estimate)

# metrics: AUC
test_tbl %>%
  yardstick::roc_auc(truth, class_prob) %>%
  dplyr::select(.metric, .estimate)

# metrics: precision & recall
tibble(
  precision = test_tbl %>% yardstick::precision(truth,estimate) %>% pull(.estimate),
  recall = test_tbl %>% yardstick::recall(truth, estimate) %>% pull(.estimate)
)

# metrics: F1

test_tbl %>%
  yardstick::f_meas(truth, estimate, beta = 1) %>%
  dplyr::select(.estimate)