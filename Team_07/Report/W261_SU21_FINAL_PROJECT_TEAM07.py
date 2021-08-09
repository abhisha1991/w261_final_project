# Databricks notebook source
# MAGIC %md 
# MAGIC # MASE Airlines: Delayed Flight Predictions
# MAGIC W261 Machine Learning At Scale | Summer 2021, Section 002
# MAGIC 
# MAGIC Team 7 | Michael Bollig, Emily Brantner, Sarah Iranpour, Abhi Sharma
# MAGIC 
# MAGIC ##### Current airlines have too many delays and not enough lead time on notifications when delays inevitably happen. MASE aims to shake up the industry by being the premier low-delay airline. We’re accomplishing this by investing heavily in our data science team to create predictive models to know when and why delays occur so we can stop them at the source.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Section 1 - Question Formulation
# MAGIC 
# MAGIC Flight delays cost consumers over 18 billion dollars annually, and cost airlines another 10 billion. Flights are only becoming more delayed (with the effects of covid being the exception) as more an more people rely on flights as a common means of transportation. Between 2016 and 2017, the number of on-time flights decreased by 8.5% (Thiagarajan et al. 2021). Delayed and canceled flights cause a loss of trust between the consumer and the airline, which we at MASE hope to avoid. The goal of our project is to use weather and airline data to predict whether flights will be delayed- we define this as taking off fifteen minutes or more beyond the scheduled time or canceling the flight altogether. Past attempts to predict delays based on weather and flight data have been largely successful, with at least one deep learning model achieving 96.2% accuracy (Yazdi et al., 2020). Another study looked at airline data without weather data, and found that between random forest, logistic regression, KNN, decision trees, and Naive Bayes, their random forest models had the best outcomes by over 4%, managing to get 66% accuracy, with logistic regression following behind it with 62% accuracy (Huo et al., 2021). A quick google search results in thousands of papers, theses, dissertations, and kaggle contests, all surrounding how best to predict flight delays, indicating that we've chosen a very important topic to focus on.
# MAGIC <br>
# MAGIC <br>
# MAGIC Based on these previous studies, the limited scope of the project (limited to only the weather and subsetted airline information from 2015-2019) and the limited timeline of the project, we felt that a 60% F1 score would be a good indicator of success as we wanted to at least have better than 50% to be practically useful. We chose to use F1 scores as our main evaluation as it's a decent blend between both precision and recall. When a flight is delayed without warning (low recall), passengers will be unhappy and less likely to book with the airline again, and when a flight is predicted as a delay but ends up being on time (low precision), passengers may miss their flights and revenue may be lost from making alternative arrangements- along with airlines taking the brunt of flying an empty plane.
# MAGIC <br>
# MAGIC <br>
# MAGIC To become the premier "low delay airline", the specific question we're answering is:
# MAGIC #### Two hours before the planned departure time, can we use weather and airline data to predict with an F1 score of over 60% whether a flight will be canceled or delayed by more than 15 minutes?
# MAGIC <br>
# MAGIC To accomplish this, we use four datasets, three which were provided (airlines, weather and stations), and one which we sourced separately. The airline data originally came from the US Department of Transportation's passenger flight's on-time performance data. In our case, it was subsetted down to only US domestic flights from the year 2015-2019. It was already set up with the feature "dep_del_15", which indicates whether a flight was more than 15 minutes late, but not if the flight was canceled. The weather and station data was from the National Oceanic and Atmospheric Administration repository, with data from stations all around the world. The OpenFlights dataset was sourced from openflights.org, and provided us with the additional airport features that we needed to easily join our data. These four datasets were joined together to create one massive dataset with all flight and weather features on US domestic flights from 2015-2019. From there, we created additional features from the data such as percent and mean delays across different aggregations, holiday indicators, prior delays, delay potential, and others.
# MAGIC <br>
# MAGIC <br>
# MAGIC Because we’re working with time series data, we can't just split the data without thought, since time may have a factor in whether flights are delayed. Instead, we used cross validation to split into our train and test sets. We have significantly more on-time flights than delayed flights, which makes our data set unbalanced. In at least one previous study, upsampling the delayed data was shown to overfit (Yazdi et al., 2020) so we undersampled our on-time flights to balance the two classes. After preparing and transforming the data, we selected four models to predict delays: logistic regression, support vector machine, gradient boost tree, and random forest. We expected our random forest model to give us the best results based on previous studies (Huo et al., 2021), but our logistic regression model had the best results with a mean F1 score of 0.92. With these scores, we feel confident that we’ll be able to reduce and better predict delays for our flights.
# MAGIC <br>
# MAGIC <br>
# MAGIC 
# MAGIC ##### References:
# MAGIC Huo, Jiage & Keung, K.L. & Lee, C. & Ng, Kam K.H. & Li, K.C.. The Prediction of Flight Delay: Big Data-driven Machine Learning Approach. (2021). 
# MAGIC 10.1109/IEEM45057.2020.9309919. 
# MAGIC <br>
# MAGIC <br>
# MAGIC Thiagarajan B, et al. A machine learning approach for prediction of on-time performance of flights. In 2017 IEEE/AIAA 36th Digital Avionics Systems Conference (DASC). New York: IEEE. 2017.
# MAGIC <br>
# MAGIC <br>
# MAGIC Yazdi, M.F., Kamel, S.R., Chabok, S.J.M. et al. Flight delay prediction based on deep learning and Levenberg-Marquart algorithm. J Big Data 7, 106 (2020). https://doi.org/10.1186/s40537-020-00380-z

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Section 2 - EDA & Discussion of Challenges
# MAGIC 
# MAGIC To start, we'll set up our environment by load all of the packages and variables that we'll need for reading and storing the data:

# COMMAND ----------

#imports
from pyspark.sql.functions import col, substring, split, when, lit, max as pyspark_max, countDistinct, count, mean,\
sum as pyspark_sum, expr, unix_timestamp, to_utc_timestamp, to_timestamp, concat, length, row_number
from pyspark.sql.window import Window
from pyspark.sql import SQLContext
from pyspark.sql.types import IntegerType, StringType, BooleanType, DateType
import pandas as pd
from gcmap import GCMapper, Gradient
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar
import seaborn as sns
import graphframes as gf
from pyspark.ml.image import ImageSchema

#for reading and writing
blob_container = "w261team07container" # The name of your container created in https://portal.azure.com
storage_account = "w261team07storage" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261team07" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "w261team07-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Core Datasets and Cleaning
# MAGIC For this predictive modeling task, three primary datasets have been provided: a flights table, a weather station table, and a weather table. In order to get a baseline understanding of these datasets, the first quarter data from 2015 will be loaded for all three tables. The flights table has additionally been preprocessed to only contain flights originating from the two busiest US airports: Chicago (ORD) and Atlanta (ATL). While we did do a lot of our initial work with the dataset using the three months of data, rather than the full data, we wrote our work into functions to that we could reuse them easily with the full data. The full EDA code may be found in the separate attached EDA notebook, with cleaned and summarized results read into this notebook for explanatory purposes.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Airline Flight Data
# MAGIC First, we will take a look at the raw provided airlines data:

# COMMAND ----------

# Load the full airline dataset
df_airlines_full = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*")
display(df_airlines_full.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC **Airline Data Cleaning Summary**
# MAGIC 
# MAGIC The airlines dataset is robust, however, some features are unnecessary for our purpose and many rows are duplicated. An overview of the cleaning steps for the airline table are as follows:
# MAGIC 
# MAGIC 1. Feature removal
# MAGIC   - Alternative names for airline carriers, airports, cities, and states removed, leaving OP_UNIQUE_CARRIER, ORIGIN/DEST_AIRPORT_ID, ORIGIN/DEST, ORIGIN/DEST_STATE_ABR, ORIGIN/DEST_CITY_NAME 
# MAGIC   - Removed information about dept, 2nd, 3rd, 4th, and 5th diverted flights, because we're only looking at whether a flight is or is not 15 minutes delayed. Any diversion is going to knock a flight more than 15 minutes off track, and the 4th and 5th diverted flights are all NULL in the 3mo dataset.
# MAGIC   - Removed additional gate departure time columns as that departure time is built into the departure delay.
# MAGIC 
# MAGIC 2. Feature combinations
# MAGIC   - To make up for removing the information about the specific diverted flights, we added the diverted flight arrival delays into the arrival delay. Diverted delays previously had their own column and now they're combined. We know where the delay is from based on the diverted column showing 1 or 0.
# MAGIC   - Cancellations were added as delays into the outcome variable dep_is_delayed, with a separate indicator variable
# MAGIC 
# MAGIC 3. Null Values
# MAGIC   - The NULL values are generally where canceled flights do not have a departure time. At this stage, we are not filling nulls.
# MAGIC 
# MAGIC 4. Duplicate values
# MAGIC   - Duplicate values are dropped. These were not present in the three month dataset, but did exist in the full dataset.
# MAGIC 
# MAGIC 5. Additional datasets and timezones
# MAGIC   - The dataset uses local timezone and does not have the same airport code type as our weather data, so a publically available Open Airlines dataset which has both timezones and the ICAO airport codes is joined
# MAGIC   - Finally, the timezones are used to generate the "time_at_prediction_utc" feature, which is the timestamp used when predicting delays (2 hours before the planned departure).
# MAGIC 6. Indexing
# MAGIC   - An index is created for more efficient joining, sorting, and processing based on the flight date, departure time, and airplane tail number
# MAGIC 
# MAGIC After cleaning, the airlines data is structured as follows:

# COMMAND ----------

# Load the cleaned airline dataset
df_airlines_clean_full = spark.read.parquet(f"{blob_url}/airlines_clean_full_v1")
display(df_airlines_clean_full.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Weather Station Location Data
# MAGIC Next, we explored the weather station table to better understand the available features. The first 10 records are printed to show the original weather station data format:

# COMMAND ----------

# Load the weather stations data
df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")
display(df_stations.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC **Station Data Cleaning Summary**
# MAGIC 
# MAGIC The station data is not time-dependent, so there is no smaller subset to test on. This dataset is primarily used to link the flights data to the airlines data. In order to make joining more efficient, the following cleaning steps were taken:
# MAGIC 
# MAGIC 1. Feature removal
# MAGIC   - Removed all featuers except for the weather station ID and the ICAO code that will be used to join to the Airlines/OpenAirlines dataset.
# MAGIC 
# MAGIC 2. Filtering
# MAGIC   - Filtered out all stations that had a 'distance_to_neighbor' of greater than 0, removing all weather stations not located the recorded ICAO airport location.

# COMMAND ----------

# display
df_stations_clean = spark.read.parquet(f"{blob_url}/stations_clean_full")
display(df_stations_clean.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Weather Observation Data
# MAGIC Next, we'll take a peek at our NOAA [weather dataset](https://www.ncdc.noaa.gov/orders/qclcd/). The first 10 rows of the raw dataset are shown to demonstrate the structure of the data:

# COMMAND ----------

# Load the full weather dataset
df_weather_full = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*")
display(df_weather_full.limit(10))

# COMMAND ----------

# MAGIC %md 
# MAGIC **Weather Data Cleaning Summary**
# MAGIC 
# MAGIC This weather dataset has not yet been tailored for our specific use-case. As such, we are dealing with massively more data that we will require. Additionally, there are fields that are composites of multiple measures (e.g. `WND` is comprised of a list of strings, integers, and numeric IDs). The full Interface Specification Document (ISD) may be found on the NOAA site, [here](https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf). The cleaning summary is outline below:
# MAGIC 
# MAGIC 1. Data subsetting
# MAGIC   - The weather dataset contains several hundred million measurements across the globe, however, we are only interested in a small subset of these measurements
# MAGIC   - Dropped all measurements from locations that are not present as flight origination locations
# MAGIC 2. Feature selection and removal
# MAGIC   - Many of the features are nested as comma-separated strings. Using the ISD, these features are mapped and unpacked as follows:
# MAGIC     - `STATION`: The unique identifier for the station
# MAGIC     - `DATE`, `LATITUDE`, `LONGITUDE`, `NAME`: These fields are self-explanatory 
# MAGIC     - `REPORT_TYPE`: Here we are dealing with reports of type FM-12 (observation from fixed land station) and FM-15 (aviation routine weather report) 
# MAGIC     - `WND`: Set of mandatory observations about the wind in the format `ANGLE,ANGLE_QC,TYPE,SPEED,SPEED_QC`
# MAGIC       - `ANGLE`: Wind direction in angular degrees (North = 0)
# MAGIC       - `ANGLE_QC`: Quality check for the angle (Pass = 1)
# MAGIC       - `TYPE`: Wind characteristic (Calm = C, Normal = N, etc)
# MAGIC       - `SPEED`: Wind speed in meters per second
# MAGIC       - `SPEED_QC`: Quality check for the speed (Pass = 1)
# MAGIC     - `CIG`: Set of mandatory observations about the sky condition in the format `CLOUD_AGL,CLOUD_AGL_QC,METHOD,CAVOK`
# MAGIC       - `CLOUD_AGL`: Height above ground level of the lowest cloud in meters (Max = 22000, Missing = 99999)
# MAGIC       - `CLOUD_AGL_QC`: Quality check for the cloud height (Pass = 1)
# MAGIC       - `METHOD`: Method of ceiling measurement (Aircraft = A, Balloon = B, etc)
# MAGIC       - `CAVOK`: Ceiling and visibility OK code (No = N, Yes = Y, Missing = 9)
# MAGIC     - `VIS`: Set of mandatory observations about horizontal object visibility in the format `DIST,DIST_QC,VAR,VAR_QC`
# MAGIC       - `DIST`: Horizontal distance visibility in meters (Max = 160000, Missing = +9999)
# MAGIC       - `DIST_QC`: Quality check for the distance (Pass = 1)
# MAGIC       - `VAR`: Is the visibility variable? (No = N, Variable = V, Missing = 9)
# MAGIC       - `VAR_QC`: Quality check for the variability (Pass = 1)
# MAGIC     - `TMP`: Set of mandatory observations about air temperature in the format `C,QC`
# MAGIC       - `C`: Air temperature in degress Celsius (Scale = 10x, Missing = +9999)
# MAGIC       - `QC`: Quality check for the air temperaure (Pass = 1)
# MAGIC     - `DEW`: Set of mandatory observations about horizontal object visibility in the format `C,QC`
# MAGIC       - `C`: Dew point temperature in degrees Celsius (Scale = 10x, Missing =)
# MAGIC       - `QC`: Quality check for the dew point temperature (Pass = 1)
# MAGIC     - `SLP`: Set of mandatory observations about sea level atmospheric pressure (SLP) in the format `P,QC`
# MAGIC       - `P`: Sea-level pressure in hectopascals (Scale = 10x, Missing = 99999)
# MAGIC       - `QC`: Quality check for the SLP (Pass = 1)
# MAGIC   - Dropped additional optional measurements related to precipitation, snow, dew point, and weather occurrences that are too sparsely recorded.
# MAGIC 3. Replacing missing values
# MAGIC   - As listed above, when recorded measurements are missing they are filled with various codes (9, 9999, +9999, etc)
# MAGIC   - To avoid misinterpreting these as actual measurements, we replace these will nulls in our cleaning step
# MAGIC 
# MAGIC Below, we demonstrate the cleaned and unpacked dataset

# COMMAND ----------

# read in cleaned data
df_weather_full_clean = spark.read.parquet(f"{blob_url}/weather_clean_full_v1")
display(df_weather_full_clean.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC To better understand the benefit of dropping irrelevant weather records at this stage, let's take a look at how many weather measurements come from airports in our flights dataset:

# COMMAND ----------

print(f"Number of flight records (airlines data): {df_airlines_clean_full.count()}")
print(f"Number of unique origin airlines (airlines data): {df_airlines_clean_full.select(col('origin_ICAO')).dropDuplicates().count()}")
print(f"Number of unique weather stations (raw weather data): {df_weather_full.select(col('STATION')).dropDuplicates().count()}")
print(f"Number of weather records (raw weather data): {df_weather_full.count()}")
print(f"Number of weather records from origin airports (cleaned weather data): {df_weather_full_clean.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC By subsetting to only relevant weather measurement locations, we werer able to drop 96.2% of the weather data. Without doing so, joining the datasets would be an enormously computationally expensive task.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 2.2 Joining the Datasets
# MAGIC 
# MAGIC We joined the provided airline data with the origin and destination ICAO codes in the cleaning function. Now, we join each of the individual core datasets together as outlined in the diagram below. In order to effectively join our weather and flight datasets, we had to consider our goal: predicting flight delays *2 hours in advance*. The station data is joined to the cleaned airline data, then the weather at the origin airport is joined *at or before the prediction time*.

# COMMAND ----------

displayHTML('''<img src="files/shared_uploads/mbollig@berkeley.edu/join_diagram.png">''')

# COMMAND ----------

# MAGIC %md #### Stations to Airlines
# MAGIC Due to the ICAO codes joined to the airlines from the OpenAirlines data sources, this join is straightforward. The stations data may be joined by this code (`neighbor_call` to `ICAO`). This is done twice: once for the origin stations and once for the destination stations. Below is the join function, run in the separate EDA notebook for both the subset dataset and the full datasets.

# COMMAND ----------

def join_stations_to_airlines(airlines, stations):
  # subset and rename
  stations = stations.select(col("neighbor_call"),
                             col("station_id").alias(f"origin_station_id"),
                             col("station_id").alias(f"dest_station_id"))
  
  # join
  joined = airlines.alias('a').join(stations.alias('so'),
                                    col("a.origin_ICAO") == col("so.neighbor_call"), 'left')\
                              .join(stations.alias('sd'),
                                    col("a.dest_ICAO") == col("sd.neighbor_call"), 'left')\
                              .select('a.*', 'so.origin_station_id', 'sd.dest_station_id')
  
  return joined

# COMMAND ----------

# MAGIC %md
# MAGIC #### Weather to Airport Join
# MAGIC 
# MAGIC The weather to airport join takes the station and airport dataset, the weather data, and the airport a prefix (either `origin_` or `dest_`). It first prepares the weather dataset by droping duplicates and prepending the column names with the specified the prefix. Next, the weather is joined to the airlines using the station ids and the prediction time, with a 2-hour buffer for the weather time. This time range join is because the weather data is rarely recorded at exact same time as a flight prediction time. This does cause duplicate flight measurements to be generated, when there are more than one weather measurement in the 2-hour window, so we need to address this. A new feature indicating the difference between the prediction time and weather recording time is generated (`weather_offset_minutes`). Finally, we are able to drop duplicate flight records by ordering our datset by index ID (chronological flights) and weather offset (increasing) and dropping on these keys. This results in flight records now having the closest weather measurement recorded within 2 hours of the prediction time. The function is shown below, and run in the separate EDA notebook:

# COMMAND ----------

def join_weather_to_airlines(airlines, weather, prefix):
  # subset and rename
  weather = weather.dropDuplicates(subset=['STATION', 'DATE'])\
                   .select([col(c).alias(prefix+str.lower(c)) for c in weather.columns if c not in ['LATITUDE', 'LONGITUDE', 'NAME']])\
  
  # join weather by station id and prediction time with 2 hour buffer, sort and drop duplicates
  joined = airlines.alias('a').join(weather.alias('w'),
                                    (col(f"a.{prefix}station_id") == col(f"w.{prefix}station")) &\
                                    (col(f"w.{prefix}date").between(col("a.time_at_prediction_utc") + expr('INTERVAL -2 HOURS'),
                                                                    col("a.time_at_prediction_utc"))),
                                    'left')\
                              .select('a.*',
                                      'w.*',
                                      ((unix_timestamp(col(f"{prefix}date")) - unix_timestamp(col("a.time_at_prediction_utc")))/60).alias(f"{prefix}weather_offset_minutes"))\
                              .drop(f"{prefix}station", f"{prefix}date")\
                              .fillna(value=0, subset=[f"{prefix}weather_offset_minutes"])\
                              .orderBy(col("index_id").asc(),col(f"{prefix}weather_offset_minutes").desc())\
                              .dropDuplicates(['index_id'])
  
  return joined

# COMMAND ----------

# MAGIC %md
# MAGIC While we were able to join the airlines and *stations* together in one go, the weather is significantly more computationally expensive. We opted to split the weather join into a separate join for each year, in order to reduce the risk of the full join being interrupted and needing to start over. Each year iwas written to parquet for safekeeping, with the next step being to union these datasets together for a comprehensive set. 
# MAGIC 
# MAGIC We recorded the time each join took to run, though were unable to note how many nodes the computation had access to due to fluctuations in availability during this time period.
# MAGIC - 2015 join: 1.69 hours
# MAGIC - 2016 join: 1.58 hours
# MAGIC - 2017 join: 1.47 hours
# MAGIC - 2018 join: 1.72 hours
# MAGIC - 2019 join: 1.59 hours
# MAGIC 
# MAGIC The joined dataset is structured as follows:

# COMMAND ----------

# read in joined data
full_join_2019 = spark.read.parquet(f"{blob_url}/full_join_2015_2019_v0")
display(full_join_2019.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Exploration and Visualizations
# MAGIC Now that we have a fully joined dataset, we plot some visualizations of the existing features in order to better understand the data and develop hypotheses for our predictive models. First, we'll take a look at the percentage of flights delayed by month of the year.

# COMMAND ----------

# Load datasets
full = spark.read.parquet(f"{blob_url}/full_join_2015_2019_v0/*")
subset = spark.read.parquet(f"{blob_url}/airlines_weather_origin_v0/*")

# prepare for plotting
plot_subset_data = subset.toPandas()
plot_cropped_subset_data = subset.filter(col("delay_minutes") >= 15).filter(col("delay_minutes") < 400).toPandas()

# COMMAND ----------

# aggregate and summarize by month
monthly_agg = full.select(['dest_ICAO', 'year', 'month', 'dep_is_delayed', 'delay_minutes' ]).dropna(how='any').groupBy('year', 'month')\
        .agg(count('dest_ICAO').alias('n_flights'),\
             pyspark_sum(when(col('dep_is_delayed') == 'true', 1).otherwise(0)).alias('n_delayed'),\
             (pyspark_sum(when(col('dep_is_delayed') == 'true', 1).otherwise(0))/count('dep_is_delayed') * 100).alias('pct_origin_dow_delayed'),\
             mean('delay_minutes').alias('mean_origin_dow_delay'))\
        .withColumn("sort_order", col('month').cast(IntegerType()))\
        .orderBy(['year', 'sort_order'], ascending=1)

# convert to pandas for plotting
plot_monthly_agg = monthly_agg.toPandas()

# plot
plt.figure(figsize=(13.5,10))
ax = sns.lineplot(data=monthly_agg, y="pct_origin_dow_delayed", x="month", hue="year", palette="Blues_d",
                 ci=None)
plt.xlabel('Month')
plt.ylabel('Percent Flights Delayed (%)')
plt.title('Percentage Departure Delays by Month')
plt.show()

# COMMAND ----------

# MAGIC %md Based on the plot above, we can see there is clear seasonality in the data. A higher percentage of flights are delayed during the summer months, with fewer delays present in September-November. When we design our models, we will need to include month of the year as a feature to account for this. Let's next take a look at if there are noticeable differences based on the origin airport. We'll plot histograms of flight delay times with our 3 month subset data of just ORD and ATL:

# COMMAND ----------

plt.figure(figsize=(13.5,6))
ax = sns.displot(data=plot_eda_data, x="delay_minutes", hue="origin_airport_code", col="origin_airport_code", linewidth = 0.5, kde=True)
plt.xlabel('Departure Delay (minutes)')
plt.subplots_adjust(top=0.85)
plt.suptitle('Departure delays by Airport')
plt.show()

# COMMAND ----------

# MAGIC %md This provides some context for the spread of the delayed flights. It would appear that the vast majority of flights have little to no delay from the planned departure time. Let's remove flights that are not delayed (<15 minutes), and flights that are delayed by more than 400 minutes.

# COMMAND ----------

plt.figure(figsize=(13.5,6))
ax = sns.displot(data=plot_cropped_eda_data, x="delay_minutes", hue="origin_airport_code", col="origin_airport_code", bins=50, linewidth = 0.5, kde=True)
plt.subplots_adjust(top=0.85)
plt.suptitle('Departure delays by Airport (15-400 minutes)')
plt.show()

# COMMAND ----------

# MAGIC %md We see that the distributions are similar in that they both have long tails, with ATL having slightly higher kurtosis. Let's see if this same distribution shape is present on the larger dataset. We'll filter to the top 10 busiest airports in order to avoid clutter in the plot: 

# COMMAND ----------

print("Top 10 airports by number of flights:")
top_10 = full.groupBy(['origin_ICAO']).agg(count('index_id').alias(f"n_flights"))\
        .orderBy('n_flights', ascending=0).limit(10).cache()
display(top_10)

# COMMAND ----------

# get busiest airport codes
top_10_ICAO = [row[0] for row in top_10.select(col('origin_ICAO')).collect()]

# filter by business, and window delays
filtered_full = full.filter(col('origin_ICAO').isin(top_10_ICAO))\
                    .filter(col("delay_minutes") >= 15)\
                    .filter(col("delay_minutes") < 400).toPandas()

plt.figure(figsize=(13.5,13.5))
ax = sns.FacetGrid(filtered_full,
                   col="origin_ICAO",
                   col_wrap=5,
                   hue="dep_is_delayed",
                   sharey=False)
ax.map(sns.histplot, "delay_minutes", linewidth=0.5, alpha=.5, bins=30)
plt.subplots_adjust(top=0.85)
plt.suptitle('Departure delays by Airport (Top 10 airports, 15-400 minute delays)')
plt.show()

# COMMAND ----------

# MAGIC %md It would appear that the distributions do vary between airports, though they all have a similar shape. We opt to keep the airport code as a feature in our models, as it is not too cardinal and does appear to contain some explanatory power. Next, let's take a look at the carriers with respect to the airports. We hypothesize that carriers will be an important feature distinct from the airport, since they have their own specific hurdles to tackle (staffing, equipment, scheduling systems, etc). We'll look at a stripplot to see a side-by-side spread of each as well as histograms to get a better idea of the density of the distributions. We'll also be using the 3 month dataset, as the full 5-year is too dense to plot without aggregating.

# COMMAND ----------

plt.figure(figsize=(13.5,10))
ax = sns.stripplot(data=plot_eda_data, y="carrier", x="delay_minutes",
                   hue="origin_airport_code", hue_order=["ORD", "ATL"], dodge=True,
                   size = 4,  linewidth = 0.5,  jitter=0.25)
plt.axvline(x=0, color='k', linestyle='--', linewidth = 0.5)
plt.axvline(x=15, color='r', linestyle='--', linewidth = 0.5)
plt.xlabel('Departure Delay (minutes)')
plt.ylabel('Carrier')
plt.title('Departure delays by Airport-Carrier')
plt.show()

# COMMAND ----------

plt.figure(figsize=(13.5,13.5))
ax = sns.FacetGrid(plot_cropped_eda_data,
                   col="carrier",
                   col_wrap=5,
                   hue="origin_airport_code",
                   sharey=False)
ax.map(sns.histplot, "delay_minutes", linewidth=0.5, alpha=.5, bins=30)
plt.subplots_adjust(top=0.85)
plt.suptitle('Departure delays by Airport-Carrier (15-400 minutes)')
plt.show()

# COMMAND ----------

# MAGIC %md We do see distinct differences between the airline carrier AND airports. This aligns with what we were expecting and indicates that we ought to include the carrier in our predictive models. Next, let's do a little further investigation into how the timing of flights may impact delay likelihood. We know that there is seasonality by month, but let's look at if the day of the week may also have an impact. The thought here being that individuals are more likely to travel on weekends, and busier airports may have a higher propensity for flight delays. We'll start by aggregating thee full data by weekday and hour, then calculating summary statistics. Finally, these are plotted as to show weekdays: 

# COMMAND ----------

full_weekly_agg = full.withColumn("hour_of_day", substring('planned_dep_time',1,2))\
        .groupBy('origin_ICAO', 'day_of_week', 'hour_of_day')\ # aggregate by weekday and hour
        .agg(count('dest_ICAO').alias('n_flights'),\ # number of flights
             pyspark_sum(when(col('dep_is_delayed') == 'true', 1).otherwise(0)).alias('n_delayed'),\ # number flights delayed
             (pyspark_sum(when(col('dep_is_delayed') == 'true', 1).otherwise(0))/count('dep_is_delayed') * 100).alias('pct_origin_dow_delayed'),\ # percent of flights delayed
             mean('delay_minutes').alias('mean_origin_dow_delay'))\ # 
        .withColumn("sort_order", concat(col('hour_of_day'), col('day_of_week')).cast(IntegerType()))\
        .orderBy('sort_order', ascending=1)
plot_full_weekly_agg = full_weekly_agg.toPandas()
display(full_weekly_agg.limit(10))

# COMMAND ----------

plt.figure(figsize=(13.5,6))
ax = sns.barplot(data=plot_full_weekly_agg, y="pct_origin_dow_delayed", x="day_of_week",palette="ch:.25",
                 linewidth = 0.5)
plt.xlabel('Departure Day of Week')
plt.ylabel('Percent Flights Delayed (%)')
plt.title('Percentage Departure Delays by Weekday')
plt.show()

# COMMAND ----------

# MAGIC %md There do appear to be differences by weekday, though they do not align with our expectation that weekends would have more delays. We do think there is value in including this feature as well. Let's look at weekday and hour-of-day to see if more flights are delayed in the morning or afternoon.

# COMMAND ----------

plt.figure(figsize=(13.5,6))
ax = sns.barplot(data=plot_full_weekly_agg, y="pct_origin_dow_delayed", x="day_of_week",
                 hue='hour_of_day', palette="ch:.25",
                 ci=None)
ax.legend_.remove()
plt.xlabel('Departure Day of Week')
plt.ylabel('Percent Flights Delayed (%)')
plt.title('Percentage Departure Delays by Weekday and Hour')
plt.show()


# COMMAND ----------

# MAGIC %md Here there is a clear trend for every day of the week. It appears that flights later in the day (local time) have higher delay likelihood. This feature has the strongest trend yet, so we opt to include it in our model features as well. Finally, since we are looking at flight data we would be remiss if we did not create a cool visualization for the flight paths. We used the `GCMapper` module to visualize our flights, with some custom cropping and color-coding written in.

# COMMAND ----------

def plot_map(routes, crop_to_usa=True, dataset_name=''):
  # extract values
  orig_lon = [float(row['origin_longitude']) for row in routes]
  orig_lat = [float(row['origin_latitude']) for row in routes]
  dest_lon = [float(row['dest_longitude']) for row in routes]
  dest_lat = [float(row['dest_latitude']) for row in routes]
  n_flights = [float(row['n_flights_for_route']) for row in routes]
  
  # normalize flights
  n_min = min(n_flights)
  n_max = max(n_flights)
  n_flights_normalized = [(x-n_min)/(n_max-n_min) for x in n_flights]

  # create gradient
  g = Gradient(((0, 0, 0, 0), (0.5, 0, 204, 85), (1, 205, 255, 204)))

  # initialize GCMapper and set data
  gcm = GCMapper(cols=g, height=8000, width=16000)
  gcm.set_data(orig_lon,
               orig_lat,
               dest_lon,
               dest_lat,
               n_flights_normalized)
  # draw image
  img = gcm.draw()

  # plot
  plt.figure(figsize=(13.5, 10))
  plt.axis('off')
  plt.title(f"Flight routes - {dataset_name}")
  if crop_to_usa:
    plt.imshow(img.crop((600, 800, 5500, 3500)))
  else:
    plt.imshow(img)
  plt.show()

# COMMAND ----------

plot_map(agg_route.collect(),
         dataset_name='Q1 2015, ORD and ATL')

# COMMAND ----------

plot_map(aggregate_keys(df_airlines_full_clean,
                        ['origin_ICAO',
                        'dest_ICAO',
                        'origin_longitude',
                        'origin_latitude',
                        'dest_longitude',
                        'dest_latitude'],
                        'for_route',
                        count_only=True).filter(col("origin_ICAO").isNotNull() & col("dest_ICAO").isNotNull()).collect(),
         dataset_name='2015-2019, US Domestic')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3 - Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Aggregate Features
# MAGIC We suspect that there are features related to the flight path and origin/destination locations that wouldd have high predictive value. Anecdotally, there are certain airports where more delays occur, or certain flight routes that always seem to be delayed. In order to quantify this, we wrote a simple aggregation function that takes a list of keys and provides summary statistics about them:
# MAGIC - Total number of flights
# MAGIC - Total number of delayed flights
# MAGIC - Percentage of flights delayed
# MAGIC - Mean delay time (minutes)
# MAGIC 
# MAGIC This function is copied below, and again run in our EDA/Feature Engineering notebook.

# COMMAND ----------

def aggregate_keys(df, keys, name, count_only=False):
  if count_only:
    return df.groupBy(keys).agg(count('index_id').alias(f"n_flights_{name}"))
  
  else:
    return df.groupBy(keys).agg(count('index_id').alias(f"n_flights_{name}"),\
                                count(when(col("dep_is_delayed"), col("dep_is_delayed"))).alias(f"n_delayed_{name}"),\
                                (count(when(col("dep_is_delayed"), col("dep_is_delayed")))/count('dep_is_delayed') * 100).alias(f"pct_delayed_{name}"),\
                                mean('delay_minutes').alias(f"mean_delay_{name}"))

# COMMAND ----------

# MAGIC %md Let's take a look at some aggregates of our 3 month subset dataset. We'll aggregate by airport, route, and state to see if features about the flight routes or origin/destinations are substantially different:

# COMMAND ----------

# origin aggregates
agg_origin = aggregate_keys(joined_eda,
                            ['origin_ICAO'],
                            'from_origin')
display(agg_origin)

# dest aggregates
agg_dest = aggregate_keys(joined_eda,
                          ['dest_ICAO'],
                          'to_dest')
display(agg_dest)

# route (origin + destination) aggregates
agg_route = aggregate_keys(joined_eda,
                           ['origin_ICAO',
                           'dest_ICAO',
                           'origin_longitude',
                           'origin_latitude',
                           'dest_longitude',
                           'dest_latitude'],
                           'for_route')
display(agg_route)

# state aggregates
agg_origin_state = aggregate_keys(joined_eda,
                           ['origin_state'],
                           'from_state')
display(agg_origin_state)
agg_dest_state = aggregate_keys(joined_eda,
                         ['dest_state'],
                         'to_state')
display(agg_dest_state)


# COMMAND ----------

# MAGIC %md These aggregated statistics provide valuable context for how likely different flights are to be delayed, based on their route, origin, and destination.  For example, in the 3 month dataset we see that the percentage of flights from ATL is 19.4% whereas from ORD is 33.9%. These features will be important for flight prediction, so we will join them onto our flights dataset. The join function below is copied from the EDA/Feature Engineering notebook:

# COMMAND ----------

def join_aggs(df, origin, dest, route, o_state, d_state):
  joined = df.alias('d').join(origin.alias('o'),
                              (col("d.origin_ICAO") == col("o.origin_ICAO")),
                              'left')\
                        .join(dest.alias('de'),
                              (col("d.dest_ICAO") == col("de.dest_ICAO")),
                              'left')\
                        .join(route.alias('r'),
                              (col("d.origin_ICAO") == col("r.origin_ICAO")) &\
                              (col("d.dest_ICAO") == col("r.dest_ICAO")),
                              'left')\
                        .join(o_state.alias('os'),
                              (col("d.origin_state") == col("os.origin_state")),
                              'left')\
                        .join(d_state.alias('ds'),
                              (col("d.dest_state") == col("ds.dest_state")),
                              'left')\
                        .select('d.*',
                                'o.pct_delayed_from_origin',
                                'o.mean_delay_from_origin',
                                'de.pct_delayed_to_dest',
                                'de.mean_delay_to_dest',
                                'r.pct_delayed_for_route',
                                'r.mean_delay_for_route',
                                'os.pct_delayed_from_state',
                                'os.mean_delay_from_state',
                                'ds.pct_delayed_to_state',
                                'ds.mean_delay_to_state'
                                )\
                        .orderBy(col("index_id").asc())
  return joined

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at our full dataset with the additional aggregate statistics joined on:

# COMMAND ----------

full_join_with_aggs = spark.read.parquet(f"{blob_url}/full_join_with_aggs_v0/*")
display(full_join_with_aggs.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Window Calculations
# MAGIC 
# MAGIC 1. Previous Flight Delay (`prev_fl_del`)
# MAGIC   - Prior flight performance can have an impact on flight departure times. We tracked flights by their tail numbers and arrival times in order to determine if the previous flight was delayed. If the previous flight was delayed, we set the indicator to 1 and if it was not delayed, we set the indicator to 0. 
# MAGIC  
# MAGIC 2. Potential for Delay (`poten_for_del`)
# MAGIC   - Previous flight arrival times can also have an impact on flight departure times. After landing, the plane needs to be refueled, cleaned, and maintenanced. The cabin crew and pilots may need to be changed. The more time in between the flight’s arrival time and next departure time, the less likely the flight departure will be delayed. We calculated the time in between flights by tracking the tail number and actual arrival time and created an indicator where flights with more than 2 hours in between flights were indicated with a 1 and less than 2 hours were indicated with a 0. Flights that were canceled, diverted, or did not have a previous flight were null and were indicated with a -1. 
# MAGIC  
# MAGIC 3. Indicators for Average Delay 2-4 Hours Prior to Planned Departure 
# MAGIC   - At times, there may be certain issues, such as security, weather, maintenance, etc., that can affect flight performance at the airport or carrier level. We created a few indicator variables to capture the average delay minutes 2-4 hours prior to planned departure times. If the average delay 2-4 hours prior is less than 15 minutes, it is assigned a 0, and if it is greater than 15 minutes, it is assigned a 1. There are null values if there are no flights in the 2-4 hour measurement window. We assign nulls a -1. 
# MAGIC 
# MAGIC     - Origin Airport Average Delay 2-4 Hours Prior (`oa_del_ind`)
# MAGIC       - This feature is created based on calculating the average delay minutes 2-4 before planned departure at the origin airport. 
# MAGIC 
# MAGIC     - Destination Airport Average Delay 2-4 Hours Prior (`da_del_ind`)
# MAGIC       - For this feature, we are determining average arrival delay at the departure 2-4 hours prior to the departure time at the origin airport. The concept is similar to the previous feature. If there is an issue that is affecting the destination airport, then flights to that airport may be delayed. 
# MAGIC 
# MAGIC     - Carrier Average Delay 2-4 Hours Prior by Origin Airport (`carrier_del_ind`)
# MAGIC       - If a specific airline is low on maintenance or cleaning staff on a certain day, it may impact departure times. We created this feature by calculating the average delay minutes by carrier at the origin airport. 
# MAGIC 
# MAGIC     - Five Delay Categories Average Delay 2-4 Hours Prior by Origin Airport (`security_window_del_ind`, `nas_window_del_ind`, `carrier_window_del_ind`, `weather_window_del_ind`, `late_ac_window_del_ind`)
# MAGIC       - There were five features, security, NAS, carrier, weather, and late aircraft delay, that provided the total delay minutes for the specific category. We created five features to capture the average delay minutes of each category by origin airport   
# MAGIC  

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Holiday and Holiday-adjacent
# MAGIC Airports typically see the most traffic during the holiday seasons. We captured this information by setting flights that depart on a US holiday to a  “holiday” category. We also set the two days prior and after a holiday to “holiday_adjacent” category since many people travel to a location before the actual holiday, spend time with their family or friends, and then fly back home after the holiday.

# COMMAND ----------

displayHTML('''<img src="files/shared_uploads/mbollig@berkeley.edu/calendar.png">''')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Mean imputation and Null Removal
# MAGIC At this point, we have finished adding all features that we intend to add. In order to ensure the data is usable with all of our models we intend to employ, we will need to address missing values. There are two approaches to take, based on whether or not the feature is numeric or categorical:
# MAGIC - **Categorical**: For nulls in categorical feature columns, we will replace the null with a string indicating it was missing (e.g. "NULL")
# MAGIC - **Numeric**: For nulls in numeric feature columns, we will first generate an indicator variable that tracks which rows are being filled, then replace the nulls with the mean of the values that are not missing from that column (mean imputation)
# MAGIC 
# MAGIC The function below, copied from the EDA/Feature Engineering notebook, fills nulls as described, with a short example to illustrate:

# COMMAND ----------

def fill_nulls(data, features_with_numeric_nulls, features_with_str_nulls):
  # indicate numeric nulls
  data = data.select(["*"] + [col(x).isNull().cast(StringType()).alias(f"{x}_null") for x in features_with_numeric_nulls]).cache()
  
  for col_name in features_with_numeric_nulls:   
    # calculate mean for column after dropping nulls
    col_mean = data.select(col(col_name)).na.drop().select(mean(col(col_name)).alias('mean')).collect()[0]['mean']

    # replace nulls with mean
    data = data.withColumn(col_name, when(col(col_name).isNull(), col_mean).otherwise(col(col_name)))

  for col_name in features_with_str_nulls:
    # replace nulls with str
    data = data.withColumn(col_name, when(col(col_name).isNull(), 'NULL').otherwise(col(col_name)))
  
  return data

# COMMAND ----------

# create example dataframe
example = spark.createDataFrame(pd.DataFrame({'index': [1, 2, 3, 4, 5, 6],
                                              'numerics': [4, 5, 6, None, 6, None],
                                              'categoricals': ['word', None, 'here', 'Apple', None, 'example']}))

print("Original data:")
display(example)
print("Filled data:")
display(fill_nulls(example, ['numerics'], ['categoricals']))

# COMMAND ----------

# MAGIC %md This concept is applied to the full dataset for features to be included in the models

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Feature Selection
# MAGIC Below is a complete list of features that we chose to keep in order to test in our models. In addition to the features we engineered, we chose to keep some features related to date and time (Year, Quarter, Month, Day of the Month, Day of the Week, Depature Hour, Arrival Hour) since our EDA showed that travel is seasonal and is also dependent on the weekday and time of day. We also kept a few location features (Airport, Carrier, State, City). We saw certain carriers and airports have more delays compared to others, which may be a result of unique logistics and procedures. We also thought distance features (Flight Distance, Planned Duration, Distance Group, Altitude) were also important to keep since the data showed longer flights have longer delays. 
# MAGIC 
# MAGIC Although the weather numeric features had low correlation with depature delay (Origin Cloud Angle, Wind Speed, Visibility Distance, Temperature, Dew Point Temperature, Sea-Level Pressure), we did not want to exclude them based on that information alone. From some of our research, we know weather can impact departure delays so we chose to keep them to see how they would perform in the models. 

# COMMAND ----------

model_features = spark.read.parquet(f"{blob_url}/model_features_v6/*")
print(model_features.columns)
display(model_features.limit(10))

# COMMAND ----------

# MAGIC %md ## Section 4 - Algorithm Exploration

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Model 1 Baseline: Logistic Regression
# MAGIC 
# MAGIC For the baseline model, we chose to implement a logistic regression model since it is easy to implement, efficient to train and it scales well. It can help us determine feature importance and the relationship between features by measuring the coefficient size and direction of association. Additionally, model coefficients are easy to explain to stakeholders. This was a natural choice for our first binary classification algorithm.
# MAGIC 
# MAGIC Logistic Regression does come with some overhead though. The features would need to be scaled before applying the model - unlike in decision tree models, where one can get away without feature scaling. Logistic Regression uses the Sigmoid function, where it returns a probability between 0 and 1. Thus, it is both a curse and a blessing - where the researcher is left with the decision to choose the decision "threshold". In our baseline case - we choose this threshold to be 0.5. If the probability is greater than 0.5, then the prediction class will be 1 (departure is delayed) and if the probability is less than 0.5, the prediction class will be 0 (departure is not delayed). We did do some fine tuning of our thresholds in later runs too - it is important to be mindful of the threshold probability in a type of problem like "airline delay classification" - because the threshold decision dictates how much confidence we want to put in the threshold when deciding a delay - a decision that impacts both customer experience and airline operating expenses.
# MAGIC 
# MAGIC There are some limitations to Logistic Regression too. Although regression models are effective in determining a baseline relationship and are efficient to train, one of the major limitations is that it assumes linearity between the dependent and independent variables. This makes it difficult to determine complex relationships that are typically found in real world situations. Logistic regression also requires low or no multicollinearity between the variables. Also, if a dataset has high dimensions, it can lead to overfitting. Thus, we have to be extra careful when selecting our hypothesis to not include features that are highly correlated with each other. For this reason, in our feature selection stage - (for example) we choose one of "flight distance" or "planned duration" as we expect them to be highly correlated. Overfitting is less of a concern because fortunately we have large amounts of data relative to the number of features we are training on. 

# COMMAND ----------

displayHTML('''<img src="files/shared_uploads/siranpour@berkeley.edu/Screen_Shot_2021_08_05_at_10_47_15_PM-1.png">''')


# COMMAND ----------

# MAGIC %md 
# MAGIC #### Model 2: Random Forest
# MAGIC 
# MAGIC For our second model, we chose to implement the random forest classifier over decision trees due to the problem of overfitting. Decision trees are a good choice when one wants to model a relatively straight forward dataset with a defined hypothesis tree. Decision trees don't consider multiple independent weak learners that may improve the classification process via a voting mechanism amongst the learners. The nature of this airline classification problem dictates that we require ensemble learning to be able gain collective classification knowledge - thus, we skip decision trees and train on Random Forests directly. 
# MAGIC 
# MAGIC Random Forest is an ensemble of decision trees, and utilizes the bagging technique. Each decision tree is fit to a subsample taken from the entire dataset. The success of a random forest model highly depends on using uncorrelated decision trees. If we use the same or very similar decision trees, overall result will not be very different than the result of a single decision tree. Bootstrapping plays a key role in creating uncorrelated decision trees. The trees are grown without any influence from other trees in the model and the result is determined by taking a majority vote from all the results of all the trees. Random Forest has the power to handle higher dimensionality and can include features with higher correlation that may need to be excluded from other models, such as logistic regression. It can handle many variables and identify the most significant ones via "feature importance" scoring. Other advantages of random forest is that the features do not need to be encoded or scaled and it has an effective method for handling null values in data too. It can maintain accuracy even when big portions of data are missing.  Based on our research on prior work on this topic, we expected this model to perform the best. 

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Model 3: Gradient Boosted Tree
# MAGIC 
# MAGIC We chose another tree-based method as our third model, Gradient Boosted Trees. It is also a set of decision trees, but differs from random forest in that it utilizes the boosting technique (whereas RF uses bagging). The trees are grown sequentially where each tree is grown by using information from the previous tree in order to minimize the errors (based on the residuals) - resulting in a strong learner (as opposed to parallel weak learners in RF). Decision trees in GBDT are not fit to the entire dataset. This is the next natural progression to random forests as it helps us correct our mistakes from our second model choice. Thus, we expect this to perform better than random forests. 
# MAGIC 
# MAGIC However, Gradient Boost Trees may take longer to train with large datasets and may be more difficult to tune. They are expected to train longer because the next set of trees can only train once we have understood the mistakes from the previous set of trees. The number of trees is important since boosting emphasizes small errors and noise. Too many trees can cause overfitting. This is not the case, however, with Random Forests. We can choose to increase the number of trees to train on without overfitting for Random Forests (of course, we suffer the computational costs of doing this - but too many trees in RF will not overfit and accuracy won't increase beyond a point). Also important to note - GBDT does not use or need bootstrapping. Since each decision tree is fit to the residuals from the previous one, we do not need to worry about having correlated trees.

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Model 4: Support Vector Machine
# MAGIC 
# MAGIC The fourth model we chose was a linear support vector machine (SVM), which creates the margin maximizing linearly separable hyperplane (as dictated by the linear kernel) that separates data into 2 classes. It differs from logistic regression in that it finds the optimial distance between the data and support vectors and minimizes the hinge loss function (Hinge loss = [0, 1 - yf(x)] ) as opposed to the logistic loss function. We used the LinearSVC class in PySpark because we were doing support vector classification - as opposed to regression. It is important to note that LinearSVC in Pyspark optimizes the Hinge Loss using the OWLQN optimizer. It only supports L2 regularization currently.
# MAGIC 
# MAGIC Further looking into SVC vs Logistic Regression, SVC tries to find an optimal hyperplane rather than focusing on maximizing the probability of the data.  It is less prone to overfitting compared to logistic regression, however, may take longer to train. We expected SVM to perform at par or better than logistic regression. 

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Other Important Considerations
# MAGIC 
# MAGIC #### Rebalancing Dataset for Training
# MAGIC The dataset is highly imbalanced, with approximately 80% of flights departing on-time. When we run our models on the imbalanced data, F1 scores are low due to low recall. Our model predicts almost all test data to be not delayed, resulting in high precision (low false positive count) - however, since false negatives are very high - our recall scores plummet as seen in the image below. Because of this, we rebalanced the data by implementing undersampling for the majority data class (ie, not-delayed flights). We chose to implement undersampling over SMOTE - because we fortunately have 4 years worth of flight data - which amounts to many millions of rows. In the event of lack of data, we would have chosen to synthetically generate more data for the minority class using SMOTE. Finally, we observe from the 2nd image that both test precision and test recall scores improve significantly once we implement the rebalancing.

# COMMAND ----------

displayHTML('''<img src="files/shared_uploads/mbollig@berkeley.edu/undersampling.png">''')

# COMMAND ----------

# MAGIC %md 
# MAGIC It is important to note that the undersampling is done on the training data to let the algorithms train on both classes equally. However, for the test data, we leave the data untouched and imbalanced because we are predicting flight delays in a real world setting. We did implement undersampling of the majority class on the test data too, just to see how the F1 scores will change. We will discuss these results in our later sections.
# MAGIC 
# MAGIC #### Time Series Split Cross Validation
# MAGIC We have a time series dataset, so we are unable to randomly split the data into train and test sets. This is because there is temporal meaning in the ordering of the rows in the data and that must be preserved during training. Thus to accommodate for this, we choose to train the data in "window iterations". For this project, we are using the cross validation on time series method. 

# COMMAND ----------

displayHTML('''<img src="files/shared_uploads/mbollig@berkeley.edu/cross_validation.png">''')

# COMMAND ----------

# MAGIC %md 
# MAGIC We have written a custom function to do this for us. This can be found in the model notebook. We ensure that the training data is always larger than the test data. We choose to implement this in chunks of several months. For example, for our runs we chose the following splits of data.
# MAGIC 
# MAGIC For the first iteration, we select the following dates for train and test:
# MAGIC 1. Train - (datetime.datetime(2015, 1, 1, 0, 0), datetime.datetime(2015, 8, 29, 0, 0))
# MAGIC 2. Test - (datetime.datetime(2015, 8, 30, 0, 0), datetime.datetime(2015, 12, 28, 0, 0))
# MAGIC 
# MAGIC For the second iteration, we select the following dates for train and test:
# MAGIC 1. Train - (datetime.datetime(2015, 1, 1, 0, 0), datetime.datetime(2016, 4, 27, 0, 0))
# MAGIC 2. Test - (datetime.datetime(2016, 4, 28, 0, 0), datetime.datetime(2016, 8, 26, 0, 0))
# MAGIC 
# MAGIC We continue this pattern till we have exhausted all 4 years of data. We can think of the above as an expanding window strategy, where the training data starts from the beginning of the dataset in each iteration of the model and expands by a few months in every subsequent iteration. The test data begins where the training data date range ends and the test data ends at a fixed interval (4 months say) from its own starting point. The same can be visualized in the diagram above. 
# MAGIC 
# MAGIC Lastly, we had also implemented a sliding window strategy for train-test splits, such that - instead of starting at the beginning of the dataset in every iteration (2015, 1, 1, 0, 0) - we can slide our starting point by a fixed number of months in every iteration. We didn't choose to train with this strategy though, because we were concerned about the falling number of data points that resulted from the rebalancing that we discussed earlier (which may lead to lack of training data).
# MAGIC 
# MAGIC #### Iteration Based Learning
# MAGIC 
# MAGIC It is important to note that we split the data into the above iterations and then run our model (say logistic regression) from scratch for each of these iterations, thereby learning a new model for each iteration date range. Once we have all our models learned, we compute standard classification scores (precision, recall, accuracy, F1, AUROC, AUPR) for each model and we then aggregate these scores by finding mean, min, max, median values for each classification score across all our models (for logistic regression). This gives us a robust estimate on what the true classification performance looks like since we are getting scores across different time ranges, from multiple independent models. 
# MAGIC 
# MAGIC Finally, the above procedure described is repeated for our other classifiers - SVM, GBT, RF and their respective models also report aggregate classification statistics. This allows us to do an apples to apples comparison of these standard classification scores between different classifers.

# COMMAND ----------

# MAGIC %md ## Section 5 - Algorithm Implementation

# COMMAND ----------

# MAGIC %md
# MAGIC #### Hypothesis Selection
# MAGIC 
# MAGIC After we finished feature engineering, we opted to subset the final features described in section 3.5. We formed several hypotheses based on these selected features, which we finally decided to train on. Some of these hypotheses were formed to test the effectiveness of the inclusion of certain collective variables that we believed would be important when predicting flight delays. For example, we know that bad weather is a huge determining factor when it comes to delayed flights, thus it is important to test the hypothesis of the inclusion of temperature and weather variables in our model as an independent hypothesis.
# MAGIC 
# MAGIC We ultimately ended up training 3 hypotheses:
# MAGIC 
# MAGIC ###### 1. Core Variables (Hypothesis 2 in Code)
# MAGIC This hypothesis includes core variables that we include in our more advanced hypotheses as well. We believe that these are simple but valuable predictors to include in any model.
# MAGIC 
# MAGIC 1. **Numeric** - 'pct_delayed_from_origin', 'pct_delayed_to_dest', 'pct_delayed_for_route', 'pct_delayed_from_state', 'pct_delayed_to_state', 'flight_distance', 'origin_tmp_c'
# MAGIC 2. **Categorical** - 'month', 'day_of_month', 'day_of_week', 'dep_hour', 'arr_hour', 'origin_ICAO', 'dest_ICAO', 'carrier', 'distance_group', 'holiday', 'poten_for_del'
# MAGIC 
# MAGIC ###### 2. Core + Geographic Environment Variables (Hypothesis 3 in Code)
# MAGIC This hypothesis includes most variables in the Core Variables hypothesis, along with weather/altitude/pressure (environment) variables that are expected to impact flight delay. We performed this training because we were explicitly interested in the impact of these environment variables on flight delays.
# MAGIC 
# MAGIC 
# MAGIC 1. **Numeric** - 'origin_altitude', 'origin_wnd_speed', 'origin_cig_cloud_agl', 'origin_vis_dist', 'origin_tmp_c', 'origin_dew_c', 'origin_slp_p','dest_altitude', 'planned_duration'
# MAGIC 2. **Categorical** - 'month', 'day_of_month', 'day_of_week', 'dep_hour', 'arr_hour', 'origin_ICAO', 'dest_ICAO', 'carrier', 'distance_group', 'holiday', 'poten_for_del', 'canceled', 'origin_cig_cavok', 'origin_wnd_type', 'origin_vis_var', 'origin_city', 'dest_city'
# MAGIC 
# MAGIC There are a few subtle differences between this selection of features vs our core variables hypothesis. First, we notice that we have selected planned_duration instead of flight_distance. These 2 are highly correlated and thus we only select one of these to avoid multicollinearity in our models. We chose planned_duration in this hypothesis because we just wanted to give some variation in the variables we select across our hypotheses. Second, we notice that we have included both origin/destination city along with their origin/destination ICAO codes. This is to accommodate the fact that there may be cities with multiple airports. Notice that we have dropped the mean/percentage computed variables in this hypothesis as we're only interested in effects of environmental conditions on our scores.
# MAGIC 
# MAGIC ###### 3. Core + Computed Variables (Hypothesis 4 in Code)
# MAGIC This hypothesis includes all core variables along with the features that were computed during the feature engineering phase. Notice that both percentage and mean variables are included in this model. Additionally, the indicator variables also contain weather related delays encoded in them. We believe this hypothesis is going to perform the best since it includes most of the meaningful variables spanning multiple areas such as weather, airport, carrier/late-aircraft, standard temporal (month, holiday, arrival hours, departure hours) and computed mean/percentage values. 
# MAGIC 
# MAGIC 1. **Numeric** - 'flight_distance', 'origin_tmp_c', 'pct_delayed_from_origin', 'mean_delay_from_origin', 'pct_delayed_to_dest', 'mean_delay_to_dest', 'pct_delayed_for_route', 'mean_delay_for_route', 'pct_delayed_from_state', 'mean_delay_from_state', 'pct_delayed_to_state', 'mean_delay_to_state'
# MAGIC 2. **Categorical** - 'month', 'day_of_month', 'day_of_week', 'dep_hour', 'arr_hour', 'origin_ICAO', 'dest_ICAO', 'carrier', 'holiday', 'weather_window_del_ind', 'carrier_window_del_ind', 'security_window_del_ind', 'late_ac_window_del_ind', 'nas_window_del_ind', 'oa_avg_del_ind', 'da_avg_del_ind', 'carrier_avg_del_ind', 'poten_for_del', 'prev_fl_del'
# MAGIC 
# MAGIC If we do see better scores in this hypothesis compared to the others, we will continue to fine tune this as our ultimate selected hypothesis.

# COMMAND ----------

# MAGIC %md
# MAGIC #### "Core Variables" Hypothesis Results
# MAGIC 
# MAGIC We ran all 4 years of data using the iterative methodology described in section 4. Below we present the results of each of the models corresponding to this hypothesis. We see that all our models perform very well, however, logitstic regression seems to outperform all other models with this variable set. It is definitely encouraging to see good results on both precision and recall - 2 metrics that we were closely tracking. These are surfaced in the form of high F1 scores. All results below are running with "default" hyperparameters which can be found in the code notebook for each model.
# MAGIC 
# MAGIC It is important to read the results below in the form of mean, min, max and median values. For example, the first row for the logit image is corresponding to the mean scores across all iteration runs (across all years) for logistic regression with this "core variables" hypothesis. For the sake of explainability, say we trained logistic regression for 2 iterations - the first iteration (trained on 2015 data and tested against 4 months of 2016) gives a F1 score of 0.8 and the second iteration (trained on 2015+2016 data and tested against 4 months of 2017) gives a F1 score of 0.9. Then row 1 in the table below will be the mean of 0.8 and 0.9, which is 0.85. The second row will give us the min (0.8), the third row will give us the max (0.9) and the last row will give us the median (also 0.85)
# MAGIC 
# MAGIC We end up getting a maximum test F1 score of about 0.921 from our logit model. SVM is comes in second with a max F1 score of 0.845. The GBT model seemed to perform better than the RF model as expected. What was interesting about these results was that tree based models didn't do as well as we were expecting them to. This motivated us to try out different hypotheses and explore hyperparmeter tuning for tree based models to see if we could significantly improve F1 scores.
# MAGIC 
# MAGIC It is important to note that the test scores below are being reported on "unbalanced" test data. To perform effective learning on the train data, we had to rebalance the training data such that the positive and negative classes had equal number of examples. However, on the test data - we **do not** rebalance because it is representing a real world dataset where flights are expected to depart on time for the most part. To further our model run exploration, we do ultimately run a prediction against a "balanced" test dataset for our advanced hypotheses. This will be discussed in subsequent sections. 

# COMMAND ----------

displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/hyp2_default_logit.png">''')
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/hyp2_default_svm.png">''')
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/hyp2_default_rf.png">''')
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/hyp2_default_gbt.png">''')

# COMMAND ----------

# MAGIC %md
# MAGIC #### "Core + Geographic Environment Variables" Hypothesis Results

# COMMAND ----------

# MAGIC %md
# MAGIC Similar to our core model, this model also seemed to perform well on logistic regression, compared to the tree based models. Amongst the tree models, it seems that random forests got a good boost in their F1 scores and closed the gap that was previously present between RF and GBT models. Both classifiers now show maximum F1 scores of about 0.8, which is still worse than logistic regression. This shows that the RF model selected trees that were uncorrelated to each other and was able to learn effectively from multiple weak learners. What is also interesting is that the RF models seem to be performing slightly better than the GBT models (if we compare the mean scores of 0.744 vs 0.721). Note that all models were run with the "default" hyperparameters which can be found in the models notebook.
# MAGIC 
# MAGIC What was interesting about this set of results was that SVM reported a 0 F1 score against one of the iteration runs. This was unexpected and we suspect this is because of some error in the processing pipeline. We were unable to investigate the root cause as the same SVM pipeline seemed stable for other hypotheses. Further investigation is warranted to understand why this reported unusual results with default hyperparameter values. For now, we are reporting it. 
# MAGIC 
# MAGIC The results here also confirm that the weather and environment related variables do not give a very significant boost in classification scores all up, although they do help certain classifiers more than others. For example, between the previous hypothesis (core variables) and the current one (including environmental factors), Random Forests were able to gain significant boost in F1 scores (0.72 vs 0.80), whereas other algorithms didn't benefit much (SVM and logit whose F1 scores still hover around 0.83 and 0.92 respectively).

# COMMAND ----------

displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/hyp3_default_logit.png">''')
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/hyp3_default_svm.png">''')
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/hyp3_default_rf.png">''')
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/hyp3_default_gbt.png">''')

# COMMAND ----------

# MAGIC %md
# MAGIC #### "Core + Computed Variables" Hypothesis Results
# MAGIC 
# MAGIC This is the "best" model that the team chose since it had the highest scores for almost all models across the iterations. Note that all models were again run with default hyperparameters which can be found in the models notebook. In this hypothesis, almost all models benefited significantly. For example, SVM scores went up from 0.84 to 0.887 (max F1 scores compared to the previous hypothesis). Similarly, the tree based learners also benefited significantly. For example, GBT's F1 scores jumped from around 0.80 to around 0.90! Similarly, F1 scores for RF increased from 0.80 to 0.87. Logistic regression didn't benefit too much since it was already at around 0.92. This shows that once models have reached a high enough value in their classification metrics, any incremental gains are harder to come by!
# MAGIC 
# MAGIC The tree classifiers closed the gap and performed almost at par with logistic regression (0.90 max F1 score for GBT vs 0.92 max F1 score for Logistic Regression). This can be seen with their maximum F1 scores (row 3) against the test data. GBT performed slightly better than RF in this run - this is evidenced by slightly higher mean, min, max and median F1 scores betwen the 2 classifiers. Note that the test data on which prediction is being performed is unbalanced since it represents "real world" unbalanced data scenarios that we are predicting against.

# COMMAND ----------

displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/Hyp4_default_logreg.png">''')
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/Hyp4_default_SVM.png">''')
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/Hyp4_default_random_forest.png">''')
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/Hyp4_default_GBT.png">''')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Hyper Parameter Tuning on Hypothesis 4 ("Core + Computed Variables")
# MAGIC To further perform hyper parameter tuning on our selected hypothesis, we had 2 variants of hyper-parameters. For details on the hyperparameters used, please refer to the notebooks in the "ModelRuns" folder. At a high level:
# MAGIC 1. **Custom1 hyper parameters** were trying to enforce stricter regularization (higher values for regParam, minInstancesPerNode, minInfoGain) 
# MAGIC 2. **Custom2 hyper parameters** were trying to enforce more balanced values for regularization, while allowing higher flexibility for numTrees, maxIter, maxDepth. This was done to allow the trees to learn without constraints.
# MAGIC 
# MAGIC The results are presented below. We see that the scores do not change very much from what we had before. Again GBT seems to perform slightly better against RF as expected (0.87 vs 0.90 max F1 scores). Logistic regression still performs the best amongst these with a score of about 0.924. While more hyper parameter tuning is warranted for a full analysis on the effect of hyper parameters on model results, it does seem slightly intruiging that the existing set of tuning parameters didn't do much in changing the scores drastically. We may need to try more extreme values in our future model runs that may help move the needle on the results (for better or for worse!).

# COMMAND ----------

# custom 1 hyperparams
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/hyp4_custom1_logit.png">''')
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/hyp4_custom1_svm.png">''')
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/hyp4_custom1_rf.png">''')
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/hyp4_custom1_gbt.png">''')

# custom 2 hyperparams
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/Hyp4_custom2_logreg.png">''')
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/Hyp4_custom2_SVM.png">''')
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/Hyp4_custom2_random_forest.png">''')
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/Hyp4_custom2_GBT.png">''')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Rebalancing Test Data On Hypothesis 4 ("Core + Computed Variables")
# MAGIC 
# MAGIC As we had mentioned before, we had to perform rebalancing of the training data on our examples in order to provide similar ratio of examples for positive and negative classes. We did this against the train data for all the models we have seen so far. However, on the test data, we were not performing rebalancing since we don't expect to predict against a data set that is rebalanced. However, for the sake of completeness, we performed a model run for hypothesis 4 with default hyper parameter values with balanced test data as well. 
# MAGIC 
# MAGIC We see the results presented below. We observe that the F1 scores drop across the board but are still good for being used as classifiers. At this point, both GBT and Logistic Regression seem to give a maximum test F1 score of around 0.75. We will still choose to work with Logistic Regression at this time given that it is less complex and faster to train compared to GBT models. 

# COMMAND ----------

displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/Hyp4_rebalanced_logreg.png">''')
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/Hyp4_rebalanced_SVM.png">''')
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/Hyp4_rebalanced_rf.png">''')
displayHTML('''<img src="files/shared_uploads/abhisha@berkeley.edu/Hyp4_rebalanced_GBT.png">''')


# COMMAND ----------

# MAGIC %md ## Section 6 - Conclusions

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In this project, we were able to successfully classify flight delays with high confidence using a metric of our choosing (F1 score). The motivation for choosing this metric has already been described above - keeping both the passengers and the airline as the main stakeholders in mind. We were able to achieve high scores across all our classifier models and we decided to choose the logistic regression model as our final model for a few reasons:
# MAGIC 1. It consistently classified the data well across a range of hyper parameters
# MAGIC 2. It was quicker to train against other models - a key consideration when we need to be updating the model on a real time basis with incoming flight data.
# MAGIC 3. It is highly scalable and generalizes well
# MAGIC 
# MAGIC The only downside to logistic regression is hyperparameter tuning where we are required to explicitly choose thresholds and we are required to pre-process the data (null handling, normalization of features).
# MAGIC 
# MAGIC We performed limited hyperparameter tuning and trained a few different hypotheses across our models which resulted in the tree based models performing better and almost reaching at par with our logistic regression classifier. In our core variables hypothesis, tree based models were objectively worse than logistic regression. By the time we reached our more advanced hypothesis (Hypothesis 4), we were able to get the performance of tree learners to be at par with our selected classifier. More hyperparamter tuning is required for additional gains to our tree learners, but one can question whether the man hours required for this tuning is worth it for the incremental gains. If we anticipate sparse features in our feature space going forward, tree based models may not be the ideal choice anyway.
# MAGIC 
# MAGIC We also feature engineered some novel features that allowed us to predict with high classification metrics across the board. Some examples were holiday indicators, features around previous flight delay, features indicating delays 2-4 hours prior to departure and of course, the aggregate features for mean and percentage delay by different dimensions (routes, ICAO, state).
# MAGIC 
# MAGIC There are some improvements that can be made in the future to our current modeling state. One improvement that comes to mind is the the cross validation strategy for our train test splits. For example, we can utilize the rolling window feature we discussed in section 4 to train on a moving window of time. Additionally, we could implement block based time series cross validation to sample different "random blocks of time" in our dataset and split them into train-test. Some of these strategies may generalize well, allowing us to train effectively. From a feature selection standpoint - we can form additional hypotheses by combining other features that show up in the "feature importance" list of trees against newly engineered features. If we encounter large number of features that we want to consolidate while still capturing variation in the data, PCA can be employed to perform appropriate dimensionality reduction. We can also perform further hyper parameter tuning to improve scores, especially for tree based models - which we do eventually expect will outperform logistic regression. Lastly, we can try more advanced classification algorithms - including deep neural networks - that will help us achieve even higher scores (hopefully something closer to the SOTA - 96% classification - as discussed in the introduction).
# MAGIC 
# MAGIC Overall, we think our model performs sufficiently well in a variety of scenarios and we look forward to predicting delays accurately to improve customer experience and operational costs for MASE airlines!

# COMMAND ----------

# MAGIC %md
# MAGIC Below is a snippet of our F1 scores for train and test datasets for all the different hypotheses we employed in our training procedure. We notice that most of these F1 scores remain stable across all model runs - which shows that the classifiers were able to perform consistently well no matter what data (date ranges) they were training on.
# MAGIC 
# MAGIC We do note that there was one case where the SVM F1 score falls to 0, which has been described in section 5.

# COMMAND ----------

displayHTML('''<img src="files/shared_uploads/siranpour@berkeley.edu/Screen_Shot_2021_08_07_at_9_42_00_PM.png">''')
displayHTML('''<img src="files/shared_uploads/siranpour@berkeley.edu/Hyp3.png">''')
displayHTML('''<img src="files/shared_uploads/siranpour@berkeley.edu/Screen_Shot_2021_08_07_at_9_31_48_PM.png">''')
displayHTML('''<img src="files/shared_uploads/siranpour@berkeley.edu/hyp4_rebal.png">''')


# COMMAND ----------

# MAGIC %md ## Section 7 - Application of Course Concepts
# MAGIC 
# MAGIC #### One Hot Encoding
# MAGIC We applied various transformations to our data, one of which is one hot encoding (creates a binary column for each category and returns a sparse matrix). Some algorithms such as logistic regression and support vector machine, are unable to work with categorical data directly. Features that do not have a numeric order (or meaning) are converted to numerical form by first assigning each category to an integer value. Then they are one hot encoded to avoid algorithms treating the numbers as an attribute of significance. Each of the one hot encoded features has a value of 1 where the observations in the dataset has the categorical value, and 0 for all other observations. This is a good method for encoding categorical features because it does not assume an inherent order in the categories and is thus suited for both nominal and ordinal features.
# MAGIC 
# MAGIC Its important to note that one hot encoding should be applied to categorical columns with low to medium cardinality. However, if a column has very many unique values - a one hot encoded vector will result in very sparse vectors for that feature, which takes up both memory and increases compute time, while also possibly hurting algorithm training. If there are columns with extremely high cardinality, we should first group them into smaller buckets and then perform one hot encoding against these buckets. 
# MAGIC 
# MAGIC #### Gradient Descent and Importance of Normalization of Numeric Features
# MAGIC 
# MAGIC Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. The idea is to take repeated steps in the opposite direction of the gradient (or approximate gradient) of the function at the current point, because this is the direction of steepest descent. The important notes about gradient descent are that:
# MAGIC 1. It should be applied only on convex functions (to get the guaranteed global minima)
# MAGIC 2. The loss function on which we apply this procedure should be differentiable at all points of the curve. This influences the choice in ML algorithm that is taken into consideration. Each of the algorithms we considered had differentiable loss functions.
# MAGIC 3. It should be applied against normalized features for faster training times. When dealing with features that are of different dimension scale (say number of rooms, vs sq.ft. - in a house price prediction problem) - gradient descent may take much longer to converge if the learning rate (alpha) is the same for these 2 features. This is because the step size (parameter update step) for one dimension may be very large whereas for the other dimension - it may be very small. An efficient way to deal with this problem is to normalize data to make all features fall within a similar range. We do this in our project using the StandardScaler class for each of the numeric features - when transforming features in our pipeline.
# MAGIC 
# MAGIC Even though one can perform ML algorithm parameter learning using the Newton Rhapsody method (or in general computing the hessian and jacobian matrices analytically), it is often not practical to do so and Gradient Descent provides a quick and practical alternative to do this for any general algorithm. 
# MAGIC 
# MAGIC #### Use of Spark Pipelines and Stage Based Execution
# MAGIC 
# MAGIC There are a lot of steps that go into an ML workflow (ingestion, cleaning, preprocessing, modeling, etc.). This can become an issue when we are trying to scale. We utilized Spark's ML Lib to wrap our transformers and estimators in pipelines to conduct our modeling. 
# MAGIC 
# MAGIC Our pipelines include different transformation stages, such as label indexer, string indexer, one hot encoder, and standard scaler. The label indexer is used to transform the prediction label (departure_is_delayed). The string indexer and one hot encoders are used to one hot encode the categorical features in each of our hypotheses. The standard scaler is used to normalize numeric inputs to zero mean and unit variance - which is important for Gradient Descent (as discussed above). After performing these standard transformations, at the end of the pipeline we can simply add our estimators (which represent our different classifiers). The transfomers and esitmators are executed when an "action" is called. This is by design due to Spark's lazy evaluation scheme. We also rely on Spark at this point to create optimized DAGs that will help streamline the execution of these "pipeline stages". Intelligent caching at different stages of the pipeline were also implemented to optimize performance.
# MAGIC 
# MAGIC The way we architected our code shows that we have made our pipelines highly re-usable. For example, for logit and SVM, we used common transformation pipelines. The only difference between these 2 implementations was the estimator stage at the end of the pipeline. Similary, the pre-processing for our tree models was also done via a common pipeline. This makes modeling simple and almost plug and play.
# MAGIC 
# MAGIC #### Use of Columnar Storage (Parquet) for Faster Processing
# MAGIC Since our datasets are very large and often require aggregation queries (group by, sum etc.), we decided to store our data in columnar format (parquet) for better performance. Parquet has a number of benefits:
# MAGIC 1. Parquet automatically stores schema metadata at the end of the file for faster processing
# MAGIC 2. It helps reduce storage requirements and provides higher IO throughput during ingestion and read - this was much needed by the team as they ingested multiple iterations of parquet files during the feature engineering phase.
# MAGIC 3. It is faster when processing aggregate queries - which helped us greatly when creating our mean/percentage features
# MAGIC 4. It allows for encoding nested data types and for efficiently storing sparsely populated data - we had a ton of sparsely populated columns in our dataset, so we relied on this feature heavily.
