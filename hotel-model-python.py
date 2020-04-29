import pandas as pd
import numpy as np


hotels = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv", na_values=["", " ", "NA", "NULL", "Undefined"])

hotels.head()
hotels.columns
hotels.dtypes

# determine null values

hotels.isna().sum().sort_values(ascending = False)

# unique values in each column
hotels.nunique()

# top 5 agents
hotels.groupby(['agent']).agg(total_cancelled = ('is_canceled', 'sum')).reset_index().sort_values(by = 'total_cancelled', ascending = False).head(5)

# response variable balance
hotels.groupby(['is_canceled']).agg(obs_count = ('is_canceled','count')).reset_index().assign(pct = lambda x: x.obs_count / x.obs_count.sum())

# create new dataframe
cln_hotel = hotels.copy()

# remove country and agent columns
cln_hotel = cln_hotel.drop(['company','agent'], axis = 1)

# drop all rows that have na values except meal
cln_hotel = cln_hotel.dropna(subset = ['country','distribution_channel','children','market_segment'], axis = 0)

# create a column that is the reservation date (aka arrival date)

cln_hotel = cln_hotel.assign(arrival_date = lambda x: x['arrival_date_year'].astype(str) + '-' + x['arrival_date_month'].astype(str) + '-' + x['arrival_date_day_of_month'].astype(str))
cln_hotel['arrival_date'] = pd.to_datetime(cln_hotel['arrival_date'])

# create a column to figure out the day of week of arrival date

cln_hotel = cln_hotel.assign(dayofweek = lambda x: x.arrival_date.dt.day_name())

# calculate duration of stay

cln_hotel = cln_hotel.assign(stay_duration = lambda x: x.stays_in_week_nights + x.stays_in_weekend_nights)

# calculate total guests

cln_hotel = cln_hotel.assign(total_guests = lambda x: x.adults + x.children + x.babies)

# calculate total cost

cln_hotel = cln_hotel.assign(total_cost = lambda x: x.adr * x.stay_duration)

# classify meal as "was meal included or not?"

cln_hotel['meal_included'] = 0
cln_hotel.loc[cln_hotel['meal'].notnull(), 'meal_included'] = 1
cln_hotel.loc[cln_hotel['meal'].eq('SC'), 'meal_included'] = 0

# is the hotel a city hotel?

cln_hotel['city_hotel'] = 0
cln_hotel.loc[cln_hotel['hotel'].eq('City Hotel'), 'city_hotel'] = 1

# was assigned roomtype different than requested roomtype



