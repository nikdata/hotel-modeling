# load libraries

library(tidyverse)
library(rsample)
library(recipes)

# load data

hotels <- vroom::vroom(
  file = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv",
  na = c("", " ", "NA", "NULL", 'Undefined'))


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

# create a new dataframe

cln_hotel <- hotels

# some basic EDA

# response variable breakout
cln_hotel %>%
  group_by(is_canceled) %>%
  count() %>%
  ungroup() %>%
  mutate(
    pct = n / sum(n)
  )

cln_hotel %>%
  mutate(
    status = case_when(is_canceled == 0 ~ 'Not Cancelled', TRUE ~ 'Cancelled')
  ) %>%
  ggplot(aes(x = status, fill = status)) +
  geom_bar(stat = 'count') +
  geom_text(stat = 'count', aes(label=..count..), vjust=1.6, color="white", size=4.5) +
  labs(x = 'Status', y = 'Observation Count') +
  ggtitle("Distribution of Cancellations") +
  theme_bw()

# define recipe to create new features

main_recipe <- recipes::recipe(~.,data = cln_hotel) %>%
  recipes::update_role(is_canceled, new_role = 'outcome') %>%
  recipes::step_naomit(children, country) %>%
  recipes::step_mutate(
    reservation_date = lubridate::ymd(paste0(arrival_date_year,"-",arrival_date_month,"-",arrival_date_day_of_month)),
    day_of_week = lubridate::wday(reservation_date, label = T),
    stay_duration = stays_in_week_nights + stays_in_weekend_nights,
    total_guests = adults + children + babies,
    total_cost = stay_duration * adr,
    meal_included = case_when(meal %in% c('SC','Undedfined') ~ 0, TRUE ~ 1),
    city_hotel = ifelse(hotel == 'City Hotel', 1, 0)
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
    reservation_date
  )

summary(main_recipe)

grande_meal <- recipes::prep(main_recipe, data = cln_hotel, strings_as_factors = FALSE)
burrito <- recipes::juice(grande_meal, everything())
glimpse(burrito)


# create training/test dataset
set.seed(8837)
splits <- rsample::initial_split(burrito, prop = 0.8, strata = is_canceled)

df_train <- splits %>% rsample::training()
df_test <- splits %>% rsample::testing()

burrito %>%
  group_by(is_canceled) %>%
  count() %>%
  ungroup() %>%
  mutate(
    pct = n / sum(n)
  )

df_train %>%
  group_by(is_canceled) %>%
  count() %>%
  ungroup() %>%
  mutate(
    pct = n / sum(n)
  )

df_test %>%
  group_by(is_canceled) %>%
  count() %>%
  ungroup() %>%
  mutate(
    pct = n / sum(n)
  )

ds_rec <- recipes::recipe(~., data = df_train) %>%
  recipes::step_bin2factor(is_canceled) %>%
  recipes::step_downsample(is_canceled) %>%
  prep(training = df_train)

ds_train <- juice(ds_rec) 

ds_train%>%
  group_by(is_canceled) %>%
  count() %>%
  ungroup() %>%
  mutate(
    pct = n / sum(n)
  )



# recipe to modify split datasets

model_recipe <- recipes::recipe(~., data = df_train) %>%
  recipes::update_role(is_canceled, new_role = 'outcome') %>%
  recipes::update_role(meal_included, city_hotel, is_repeated_guest, new_role = 'pred_binary') %>%
  recipes::step_normalize(recipes::all_numeric(), -recipes::has_role('pred_binary'), -is_canceled) %>%
  recipes::step_string2factor(recipes::has_type(match = 'nominal')) %>%
  recipes::step_ordinalscore(day_of_week) %>%
  recipes::step_integer(recipes::all_nominal()) %>%
  recipes::step_bin2factor(is_canceled) %>%
  recipes::step_downsample(is_canceled) %>%
  recipes::step_nzv(everything()) %>%
  recipes::prep(training = df_train, strings_as_factors = FALSE)

ds_train <- juice(model_recipe)
glimpse(ds_train)

ds_train %>%
  group_by(is_canceled) %>%
  count() %>%
  ungroup() %>%
  mutate(
    pct = n / sum(n)
  )

# apply the recipe to the test dataset

new_dftest <- recipes::bake(model_recipe, df_test)
glimpse(new_dftest)

new_dftest %>%
  group_by(is_canceled) %>%
  count() %>%
  ungroup() %>%
  mutate(
    pct = n / sum(n)
  )
