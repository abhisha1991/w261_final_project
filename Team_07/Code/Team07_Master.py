# Databricks notebook source
# MAGIC %md
# MAGIC # Feedback - Midpoint Presentation
# MAGIC - Clear objective
# MAGIC - Add a nice visual for joining of datasets together
# MAGIC - He wants code in our slides (never heard of this as a recommendation)
# MAGIC - Comment our code, describe intent of each join and action
# MAGIC - TIME SERIES DATA: need to split data for test/validation conscientiously (my thoughts: random sample the same dates of the year for all datasets)
# MAGIC   - Vini posted this on the channel: https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4#:~:text=Cross%20Validation%20on%20Time%20Series,for%20the%20forecasted%20data%20points
# MAGIC - Slide Numbers
# MAGIC - No pie charts (those weren't ours at least)
# MAGIC - Geoplots and more visuals for EDA
# MAGIC - Seasonality and time plots over years
# MAGIC - Clarity of data used (which datasets and what size)
# MAGIC - Metrics and comparison baseline values (Why are we doing the project? What is our focus? Objectively, how well are we doing?)
# MAGIC - More creative in the feature engineering
# MAGIC - Graph features and components
# MAGIC - Persistent storage
# MAGIC - RDDs
# MAGIC - Use more than just LogReg (pagerank/RF)
# MAGIC - Name dataset on each slide for the EDA
# MAGIC - Block diagrams about pipelines (workflow/preprocessing/experimental framework)
# MAGIC - "75% missing values in 3m data does not mean that the feature is useless" (think about features added in later years. Look at last 3 months of 2019?)
# MAGIC - Normalize our features before modeling
# MAGIC - Joint variable: Airport + wind angle (since crosswinds are critical and are airport-specific)
# MAGIC - Encode variable types correctly (for example, year is categorical - not numeric)

# COMMAND ----------

# MAGIC %md notes:
# MAGIC - Decrease specificity because it looks better for the airline to overperform (makes customer happier when the delay doesn't happen)
# MAGIC - ^ do that by oversampling delayed flights
# MAGIC - Aggregate weather features

# COMMAND ----------

# MAGIC %md 
# MAGIC # Delayed Flight Predictions
# MAGIC W261 Machine Learning At Scale | Summer 2021, Section 002
# MAGIC 
# MAGIC Team 7 | Michael Bollig, Emily Brantner, Sarah Iranpour, Abhi Sharma

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1 - Question Formulation

# COMMAND ----------

# MAGIC %md
# MAGIC FILL OUT

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2 - EDA & Discussion of Challenges
# MAGIC Before building our predictive system, we need to develop a deep understanding of the data being worked with. In this section we focus on cleaning, joining, and exploring the provided datasets. The workflow we implement follows this general structure:
# MAGIC <br>
# MAGIC <br>
# MAGIC 1. Loading and cleaning each provided dataset individually
# MAGIC 2. Joining together all of the cleaned data onto a single cohesive table
# MAGIC 3. Verification of data for the provided features
# MAGIC 4. Exploration of the provided features
# MAGIC 
# MAGIC First, we'll load in the required modules and configure our cloud storage:

# COMMAND ----------

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
# MAGIC ### Section 2.1 - Core Datasets
# MAGIC For this predictive modeling task, three primary datasets have been provided to be used: a flights table, a weather station table, and a weather table. In order to get a baseline understanding of these datasets, the first quarter data from 2015 will be loaded for all three tables. The flights table has additionally been preprocessed to only contain flights originating from the two busiest US airports: Chicago (ORD) and Atlanta (ATL). The data is first loaded into Spark dataframes from the raw parquet files:

# COMMAND ----------

# Load 2015 Q1 for Flights
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")

# Load the 2015 Q1 for Weather
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-04-01T00:00:00.000")

# Load the weather stations data
df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")

# Describe dataset sizes
print(f"The Q1 2015 dataset for ORD and ATL airports contains {df_airlines.count()} records, with {len(df_airlines.columns)} columns.")
print(f"The Q1 2015 NOAA dataset for weather recordings contains {df_weather.count()} records, with {len(df_weather.columns)} columns.")
print(f"The NOAA weather station dataset contains {df_stations.count()} records, with {len(df_stations.columns)} columns.")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Airline Flight Data
# MAGIC We begin with the flights table in order to better understand the available features. To start, we examine the first 10 records are alongside the provided schema.

# COMMAND ----------

display(df_airlines.limit(10))
df_airlines.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Airline Data Cleaning Summary
# MAGIC The airlines dataset is robust, with some features that were redundant, could be combined or needed to be added. We changed the names of the variables for ease of use, and our more technical changes are detailed below.
# MAGIC 
# MAGIC ##### Feature removal
# MAGIC - additional names for airline carriers, airports, cities, and states removed, leaving OP_UNIQUE_CARRIER, ORIGIN/DEST_AIRPORT_ID, ORIGIN/DEST, ORIGIN/DEST_STATE_ABR, ORIGIN/DEST_CITY_NAME 
# MAGIC - removed information about dept, 2nd, 3rd, 4th, and 5th diverted flights. Because we're only looking at whether a flight is or is not 15 minutes delayed, and not HOW delayed it is, any diversion is going to knock a flight more than 15 minutes off track, and the 4th and 5th diverted flights are all NULL in the 3mo dataset.
# MAGIC - removed aditional gate departure time columns as that departure time is built into the departure delay.
# MAGIC 
# MAGIC ##### Feature combinations
# MAGIC - to make up for removing the information about the specific diverted flights, we added the diverted flight arrival delays into the arrival delay. Diverted delays previously had their own column and now they're combined. We know what the delay is from based on the diverted column showing 1 or 0.
# MAGIC - cancellations were added as 1s into the outcome variable dep_is_delayed
# MAGIC 
# MAGIC ##### Null Values
# MAGIC - the NULL values are generally where canceled flights do not have a departure time, or flights that aren't diverted don't have a flight diversion time (and other cases along those lines). For now, we're leaving them as NULL, although we may have to change them to non-nulls depending on the model we use.
# MAGIC 
# MAGIC ##### Duplicate values
# MAGIC 
# MAGIC ##### Additional datasets and timezones
# MAGIC - because the original dataset uses local timezone and does not have the same airport code type as our weather data, we joined the open airlines dataset which has both timezones and the ICAO airport codes. As a bonus, we also have altitude information about the airport in this dataset
# MAGIC - because we wouldn't be able to see the future, our "time_at_prediction_utc" feature is the time at which we'd be predicting the delay.

# COMMAND ----------

# DBTITLE 1,Airline data cleaning function
def clean_airline_data(df):
  # drop duplicate entries and create temporary view of airlines data
  df.drop_duplicates().createOrReplaceTempView('df_tmp')
  
  # load OpenAirlines data for ICAO code and Timezone
  df_open_airlines = spark.read.csv('dbfs:/FileStore/shared_uploads/mbollig@berkeley.edu/airports.dat', header=False, inferSchema= False)\
                          .toDF("airport_id", "name", "city", "country", "IATA", "ICAO", "latitude", "longitude", "altitude", "utc_offset", "DST", "timezone", "type", "source")
    
  # create temporary view of OpenAirlines data
  df_open_airlines.createOrReplaceTempView('open_airlines')
  
  # structure query to change data types, add open_airlines, removed and combine base features, and change timezones
  query = '''
          
  SELECT 
     ROW_NUMBER() OVER (ORDER BY FL_DATE, DEP_TIME, TAIL_NUM) as index_id
     , string(oao.ICAO) as origin_ICAO
     , cast(oao.utc_offset as INTEGER) as origin_utc_offset
     , oao.timezone as origin_timezone
     , oao.latitude as origin_latitude
     , oao.longitude as origin_longitude
     , oao.altitude as origin_altitude
     , string(oad.ICAO) as dest_ICAO
     , cast(oad.utc_offset as INTEGER) as dest_utc_offset
     , oad.timezone as dest_timezone
     , oad.latitude as dest_latitude
     , oad.longitude as dest_longitude
     , oad.altitude as dest_altitude
     , string(YEAR) as year
     , string(QUARTER) as quarter
     , string(MONTH) as month
     , string(DAY_OF_MONTH) as day_of_month
     , string(DAY_OF_WEEK) as day_of_week
     , FL_DATE as dt                                  -- date of flight departure
     , string(OP_UNIQUE_CARRIER) as carrier
     , string(TAIL_NUM) as tail_num
     , string(OP_CARRIER_FL_NUM) as flight_num
     , ORIGIN_STATE_ABR as origin_state
     , ORIGIN_CITY_NAME as origin_city
     , string(ORIGIN_AIRPORT_ID) as origin_airport_id         -- airport codes can change, this stays constant
     , string(ORIGIN) as origin_airport_code
     , DEST_STATE_ABR as dest_state
     , DEST_CITY_NAME as dest_city
     , string(DEST_AIRPORT_ID) as dest_airport_id
     , string(DEST) as dest_airport_code
     , CASE
         WHEN LENGTH(CRS_DEP_TIME) = 1 THEN CONCAT('000', CRS_DEP_TIME)
         WHEN LENGTH(CRS_DEP_TIME) = 2 THEN CONCAT('00', CRS_DEP_TIME)
         WHEN LENGTH(CRS_DEP_TIME) = 3 THEN CONCAT('0', CRS_DEP_TIME)
         WHEN CRS_DEP_TIME = 2400 THEN 2359       -- 2400 is not a valid HHMM, moving back 1 minute
         ELSE CRS_DEP_TIME
       END as planned_dep_time
     , DEP_TIME as actual_dep_time
     , IFNULL(DEP_DELAY, 0) as delay_minutes                  -- negatives mean early
     , boolean(CASE
         WHEN CANCELLED = 1 THEN 1
         ELSE DEP_DEL15
       END) as dep_is_delayed                 -- OUTCOME VARIABLE, includes cancelled flights as 1 as well
     , string(DEP_DELAY_GROUP) as dep_delay_group
     , string(DEP_TIME_BLK) as dep_hour
     , ARR_TIME as actual_arr_time
     , CASE
         WHEN DIV_ARR_DELAY IS NOT NULL THEN DIV_ARR_DELAY    -- added diverted delays to arrival delays
         ELSE ARR_DELAY
       END as arr_delay_minutes                    -- negatives mean early (now includes diverted flights)
     , string(ARR_DELAY_GROUP) as arr_delay_group
     , string(ARR_TIME_BLK) as arr_hour
     , boolean(CANCELLED) as canceled
     , string(CANCELLATION_CODE) as cancel_code            -- B = weather, A = airline, C = National Air System, D = security
     , boolean(DIVERTED) as is_diverted
     , CRS_ELAPSED_TIME as planned_duration
     , ACTUAL_ELAPSED_TIME as actual_duration
     , FLIGHTS as num_flights
     , DISTANCE as flight_distance
     , string(DISTANCE_GROUP) as distance_group
     , CARRIER_DELAY as carrier_delay
     , WEATHER_DELAY as weather_delay
     , NAS_DELAY as nas_delay
     , SECURITY_DELAY as security_delay
     , LATE_AIRCRAFT_DELAY as late_aircraft_delay
     -- not including gate_departure info for flights that returned to the gate since wrapped up in delay times for previous flights
     , boolean(DIV_REACHED_DEST) as div_reached_dest
     -- not including most diverted information as not super relevant outside of knowing that it was diverted and adding the diverted arrival delay to the arrival_delay column
  FROM {0} a
  LEFT JOIN {1} oao
    ON oao.IATA = ORIGIN
  LEFT JOIN {1} oad
    ON oad.IATA = DEST

'''.format('df_tmp', 'open_airlines')

  
  # change data types, add open_airlines, remove and combine base features, and change timezones
  df_with_tz = spark.sql(query)\
      .withColumn('planned_departure_utc', to_utc_timestamp(to_timestamp(concat(col('dt'), col('planned_dep_time')), 'yyyy-MM-ddHHmm'), col('origin_timezone')))\
      .withColumn('time_at_prediction_utc', to_utc_timestamp(to_timestamp(concat(col('dt'), col('planned_dep_time')), 'yyyy-MM-ddHHmm'), col('origin_timezone')) + expr('INTERVAL -2 HOURS'))\
  
  return df_with_tz

# COMMAND ----------

df_airlines_clean = clean_airline_data(df_airlines)
display(df_airlines_clean.limit(10))

# write
df_airlines_clean.write.mode('overwrite').parquet(f"{blob_url}/airlines_eda_v1")

# COMMAND ----------

# MAGIC %md FILL OUT: comment on cleaned results

# COMMAND ----------

# MAGIC %md
# MAGIC #### Weather Station Location Data
# MAGIC Next, the weather station table will be explored in order to better understand the available features. To start, the first 10 records are printed alongside the provided schema.

# COMMAND ----------

display(df_stations.limit(10))
df_stations.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC FILL OUT: Explain cleaning steps

# COMMAND ----------

# clean station data
df_stations_clean = df_stations.filter(col("distance_to_neighbor") == 0).select('station_id','neighbor_call')

# write full dataset
df_stations_clean.write.mode('overwrite').parquet(f"{blob_url}/stations_clean_full")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Weather Observation Data
# MAGIC Next, we'll take a peek at our NOAA [weather dataset](https://www.ncdc.noaa.gov/orders/qclcd/)
# MAGIC 
# MAGIC FILL OUT: Subsetting to ORD/ATL

# COMMAND ----------

df_weather_subset = df_weather.filter((col("STATION") == '72219013874') | (col("STATION") == '72530094846'))
display(df_weather_subset.limit(10))

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC This weather set has not yet been tailored for our specific use-case. As such, we are dealing with massively more data that we will require. Additionally, there are fields that are composites of multiple measures (e.g. `WND` is comprised of a list of strings, integers, and numeric IDs). The full Interface Specification Document (ISD) may be found on the NOAA site, [here](https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf). For our purposes we will give a quick overview of some important fields:
# MAGIC <br>
# MAGIC 
# MAGIC - `STATION`: The unique identifier for the station
# MAGIC - `DATE`, `LATITUDE`, `LONGITUDE`, `NAME`: These fields are self-explanatory 
# MAGIC - `REPORT_TYPE`: Here we are dealing with reports of type FM-12 (observation from fixed land station) and FM-15 (aviation routine weather report) 
# MAGIC - `WND`: Set of mandatory observations about the wind in the format `ANGLE,ANGLE_QC,TYPE,SPEED,SPEED_QC`
# MAGIC   - `ANGLE`: Wind direction in angular degrees (North = 0)
# MAGIC   - `ANGLE_QC`: Quality check for the angle (Pass = 1)
# MAGIC   - `TYPE`: Wind characteristic (Calm = C, Normal = N, etc)
# MAGIC   - `SPEED`: Wind speed in meters per second
# MAGIC   - `SPEED_QC`: Quality check for the speed (Pass = 1)
# MAGIC - `CIG`: Set of mandatory observations about the sky condition in the format `CLOUD_AGL,CLOUD_AGL_QC,METHOD,CAVOK`
# MAGIC   - `CLOUD_AGL`: Height above ground level of the lowest cloud in meters (Max = 22000, Missing = 99999)
# MAGIC   - `CLOUD_AGL_QC`: Quality check for the cloud height (Pass = 1)
# MAGIC   - `METHOD`: Method of ceiling measurement (Aircraft = A, Balloon = B, etc)
# MAGIC   - `CAVOK`: Ceiling and visibility OK code (No = N, Yes = Y, Missing = 9)
# MAGIC - `VIS`: Set of mandatory observations about horizontal object visibility in the format `DIST,DIST_QC,VAR,VAR_QC`
# MAGIC   - `DIST`: Horizontal distance visibility in meters (Max = 160000, Missing = +9999)
# MAGIC   - `DIST_QC`: Quality check for the distance (Pass = 1)
# MAGIC   - `VAR`: Is the visibility variable? (No = N, Variable = V, Missing = 9)
# MAGIC   - `VAR_QC`: Quality check for the variability (Pass = 1)
# MAGIC - `TMP`: Set of mandatory observations about air temperature in the format `C,QC`
# MAGIC   - `C`: Air temperature in degress Celsius (Scale = 10x, Missing = +9999)
# MAGIC   - `QC`: Quality check for the air temperaure (Pass = 1)
# MAGIC - `DEW`: Set of mandatory observations about horizontal object visibility in the format `C,QC`
# MAGIC   - `C`: Dew point temperature in degrees Celsius (Scale = 10x, Missing =)
# MAGIC   - `QC`: Quality check for the dew point temperature (Pass = 1)
# MAGIC - `SLP`: Set of mandatory observations about sea level atmospheric pressure (SLP) in the format `P,QC`
# MAGIC   - `P`: Sea-level pressure in hectopascals (Scale = 10x, Missing = 99999)
# MAGIC   - `QC`: Quality check for the SLP (Pass = 1)
# MAGIC - Additional optional measurements related to precipitation, snow, dew point, and weather occurrences are similarly formatted.
# MAGIC - Some *optional* measurements are too sparsely populated for us to use in our model.
# MAGIC <br>
# MAGIC 
# MAGIC In order to make this data more usable, we will need to unpack some of these nested features. In particular, we are interested in those measurements that are likely to have an impact on flight delays (visibility, cloud cover, windspeed, etc). Note that in our printed schema, these nested fields are registered as strings, so we cannot access the values directly. We will need to transform.

# COMMAND ----------

# MAGIC %md
# MAGIC FILL OUT: Replacing missing value codes with None

# COMMAND ----------

# DBTITLE 1,Weather Cleaning Function
def replace(column, value):
  return when(column != value, column).otherwise(lit(None))

def unpack_and_clean_weather(df):
  
  # split columns
  df = df.select(df.STATION,
                 df.DATE,
                 df.LATITUDE,
                 df.LONGITUDE,
                 df.NAME,
                 split(df.WND, ',').alias("split_wnd"),
                 split(df.CIG, ',').alias("split_cig"),
                 split(df.VIS, ',').alias("split_vis"),
                 split(df.TMP, ',').alias("split_tmp"),
                 split(df.DEW, ',').alias("split_dew"),
                 split(df.SLP, ',').alias("split_slp"))

  
  # add columns and replace missing values with null
  df_weather_clean = df.select(df.STATION,
                               df.DATE,
                               df.LATITUDE,
                               df.LONGITUDE,
                               df.NAME,
                               df.split_wnd.getItem(0).alias('WND_ANGLE'),
                               df.split_wnd.getItem(1).alias('WND_ANGLE_QC'),
                               df.split_wnd.getItem(2).alias('WND_TYPE'),
                               replace(df.split_wnd.getItem(3), "9999").cast(IntegerType()).alias('WND_SPEED'),
                               df.split_wnd.getItem(4).alias('WND_SPEED_QC'),
                               replace(df.split_cig.getItem(0), "99999").cast(IntegerType()).alias('CIG_CLOUD_AGL'),
                               df.split_cig.getItem(1).alias('CIG_CLOUD_AGL_QC'),
                               df.split_cig.getItem(2).alias('CIG_METHOD'),
                               replace(df.split_cig.getItem(3), "9").alias('CIG_CAVOK'),
                               replace(df.split_vis.getItem(0), "999999").cast(IntegerType()).alias('VIS_DIST'),
                               df.split_vis.getItem(1).alias('VIS_DIST_QC'),
                               df.split_vis.getItem(2).alias('VIS_VAR'),
                               df.split_vis.getItem(3).alias('VIS_VAR_QC'),
                               replace(df.split_tmp.getItem(0), "+9999").cast(IntegerType()).alias('TMP_C'),
                               df.split_tmp.getItem(1).alias('TMP_QC'),
                               replace(df.split_dew.getItem(0), "+9999").cast(IntegerType()).alias('DEW_C'),
                               df.split_dew.getItem(1).alias('DEW_QC'),
                               replace(df.split_slp.getItem(0), "99999").cast(IntegerType()).alias('SLP_P'),
                               df.split_slp.getItem(1).alias('SLP_QC'),
                              )

  return df_weather_clean

# COMMAND ----------

# subset for relevant airports and clean
df_weather_clean = unpack_and_clean_weather(df_weather)
display(df_weather_clean)

# write
df_weather_clean.write.mode('overwrite').parquet(f"{blob_url}/weather_eda_v1")

# COMMAND ----------

# MAGIC %md FILL OUT: comment on cleaned results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 2.2 - Joining of Datasets

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Joining the data
# MAGIC <br>
# MAGIC FILL OUT: Diagram of join
# MAGIC <br>
# MAGIC 
# MAGIC We can join the station data to the airline data via the neighbor_call field. In most cases, the second three letters will match up to the airport code. In a couple of cases, they don't and we've handled it manually. We are missing one weather station in our dataset, so we substituted another station in the same city (San Juan, Puerto Rico) with the assumption that weather should be similar within the same city. By also filtering down so the `distance_to_neighbor` is 0, we can find the relevant weather stations for each airport.
# MAGIC <br>
# MAGIC <br>
# MAGIC In order to effectively join our weather and flight datasets, we'll need to consider our goal: flight delays. Intuitively, we know that weather...
# MAGIC 
# MAGIC 
# MAGIC FILL OUT: Explanation of weather->airport join

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

# # read if necessary
# df_airlines_clean = spark.read.parquet(f"{blob_url}/airlines_eda_v0/*")
# df_stations_clean = spark.read.parquet(f"{blob_url}/stations_clean_full/*")

# join stations IDs to airlines
df_airlines_stations = join_stations_to_airlines(df_airlines_clean, df_stations_clean)

# write intermediate table
df_airlines_stations.write.mode('overwrite').parquet(f"{blob_url}/airlines_stations_v0")

# COMMAND ----------

# MAGIC %md
# MAGIC FILL OUT: Explanation of weather->airport join

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

# read if necessary
df_weather_clean = spark.read.parquet(f"{blob_url}/weather_eda_v1/*")
df_airlines_stations = spark.read.parquet(f"{blob_url}/airlines_stations_v0/*")

# join origin weather
df_airlines_weather_origin = join_weather_to_airlines(df_airlines_stations, df_weather_clean, "origin_")

# write intermediate table
df_airlines_weather_origin.write.mode('overwrite').parquet(f"{blob_url}/airlines_weather_origin_v0")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 2.3 - Applying EDA to Full Datasets

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cleaning and preparing full datasets
# MAGIC 
# MAGIC FILL OUT: Narrative

# COMMAND ----------

# read full airline dataset
df_airlines_full = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*")

# clean full dataset
df_airlines_full_clean = clean_airline_data(df_airlines_full)

# write intermediate table
df_airlines_full_clean.write.mode('overwrite').parquet(f"{blob_url}/airlines_clean_full_v1")

# COMMAND ----------

# read full weather dataset
df_weather_full = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*")

# clean full dataset
df_weather_full_clean = unpack_and_clean_weather(df_weather_full)

# write
df_weather_full_clean.write.mode('overwrite').parquet(f"{blob_url}/weather_clean_full")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Joining Full Datasets

# COMMAND ----------

df_airlines_stations_full = spark.read.parquet(f"{blob_url}/full_airlines_stations_v0/*")
print(f"Number of flight records (airlines data): {df_airlines_stations_full.count()}")
print(f"Number of unique origin airlines (airlines data): {df_airlines_stations_full.select(col('origin_station_id')).dropDuplicates().count()}")
print(f"Number of unique destination airlines (airlines data): {df_airlines_stations_full.select(col('dest_station_id')).dropDuplicates().count()}")
print(f"Number of weather records (weather data): {df_weather_full_clean.count()}")
print(f"Number of unique weather stations (weather data): {df_weather_full_clean.select(col('STATION')).dropDuplicates().count()}")

# COMMAND ----------

origin_stations = [row[0] for row in df_airlines_stations_full.select(col('origin_station_id')).dropDuplicates().collect()]
print(f"Number of weather records from origin stations (weather data): {df_weather_full_clean.filter(col('STATION').isin(origin_stations)).count()}")

# COMMAND ----------

# read datasets
df_airlines_stations_full = spark.read.parquet(f"{blob_url}/full_airlines_stations_v0/*").cache()
df_weather_full_clean = spark.read.parquet(f"{blob_url}/weather_clean_full/*")

# join 2015 weather and airports
origin_stations = [row[0] for row in df_airlines_stations_full.select(col('origin_station_id')).dropDuplicates().collect()]
origin_weather = df_weather_full_clean.filter(col('STATION').isin(origin_stations)).cache()

full_join_2015 = join_weather_to_airlines(df_airlines_stations_full.filter(col('time_at_prediction_utc') < "2016-01-01T00:00:00.000"),
                                          origin_weather.filter(col('DATE') < "2016-01-01T00:00:00.000"),
                                          "origin_")


# write
full_join_2015.write.mode('overwrite').parquet(f"{blob_url}/full_join_2015_v0")

# COMMAND ----------

# join 2016 weather and airports
full_join_2016 = join_weather_to_airlines(df_airlines_stations_full.filter((col('time_at_prediction_utc') >= "2016-01-01T00:00:00.000") &\
                                                               (col('time_at_prediction_utc') < "2017-01-01T00:00:00.000")),
                                          origin_weather.filter((col('DATE') >= "2016-01-01T00:00:00.000") &\
                                                               (col('DATE') < "2017-01-01T00:00:00.000")),
                                          "origin_")


# write
full_join_2016.write.mode('overwrite').parquet(f"{blob_url}/full_join_2016_v0")

# COMMAND ----------

# join 2017 weather and airports
full_join_2017 = join_weather_to_airlines(df_airlines_stations_full.filter((col('time_at_prediction_utc') >= "2017-01-01T00:00:00.000") &\
                                                               (col('time_at_prediction_utc') < "2018-01-01T00:00:00.000")),
                                          origin_weather.filter((col('DATE') >= "2017-01-01T00:00:00.000") &\
                                                               (col('DATE') < "2018-01-01T00:00:00.000")),
                                          "origin_")


# write
full_join_2017.write.mode('overwrite').parquet(f"{blob_url}/full_join_2017_v0")

# COMMAND ----------

# join 2018 weather and airports
full_join_2018 = join_weather_to_airlines(df_airlines_stations_full.filter((col('time_at_prediction_utc') >= "2018-01-01T00:00:00.000") &\
                                                               (col('time_at_prediction_utc') < "2019-01-01T00:00:00.000")),
                                          origin_weather.filter((col('DATE') >= "2018-01-01T00:00:00.000") &\
                                                               (col('DATE') < "2019-01-01T00:00:00.000")),
                                          "origin_")


# write
full_join_2018.write.mode('overwrite').parquet(f"{blob_url}/full_join_2018_v0")

# COMMAND ----------

# join 2019 weather and airports
full_join_2019 = join_weather_to_airlines(df_airlines_stations_full.filter((col('time_at_prediction_utc') >= "2019-01-01T00:00:00.000")),
                                          origin_weather.filter((col('DATE') >= "2019-01-01T00:00:00.000")),
                                          "origin_")


# write
full_join_2019.write.mode('overwrite').parquet(f"{blob_url}/full_join_2019_v0")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 2.4 - Verify Features in Datasets

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 2.5 - Exploration and Visualizations

# COMMAND ----------

# Load Q1 2015 subset if necessary
joined_eda = spark.read.parquet(f"{blob_url}/df_airlines_weather_origin_and_dest_v1/*")

# prepare for plotting
plot_eda_data = joined_eda.toPandas()
plot_subset_data = joined_eda.limit(1000).toPandas()
plot_cropped_eda_data = joined_eda.filter(col("delay_minutes") >= 15).filter(col("delay_minutes") < 400).toPandas()

# COMMAND ----------

plt.figure(figsize=(13.5,6))
ax = sns.lineplot(data=plot_eda_data, x="dt", y="delay_minutes", hue="origin_airport_code", linewidth = 0.5)
plt.subplots_adjust(top=0.85)
plt.xlabel('Date')
plt.ylabel('Departure Delay (minutes)')
plt.title('Departure delays by Airport')
plt.show()

# COMMAND ----------

plt.figure(figsize=(13.5,3))
ax = sns.stripplot(data=plot_eda_data, y="origin_airport_code", x="delay_minutes",
                   size = 4,  linewidth = 0.5,  jitter=True)
plt.axvline(x=0, color='k', linestyle='--', linewidth = 0.5)
plt.axvline(x=15, color='r', linestyle='--', linewidth = 0.5)
plt.xlabel('Departure Delay (minutes)')
plt.ylabel('Airport Code (IATA)')
plt.title('Departure delays by Airport')
plt.show()

# COMMAND ----------

plt.figure(figsize=(13.5,6))
ax = sns.displot(data=plot_eda_data, x="delay_minutes", hue="origin_airport_code", col="origin_airport_code", linewidth = 0.5, kde=True)
plt.xlabel('Departure Delay (minutes)')
plt.subplots_adjust(top=0.85)
plt.suptitle('Departure delays by Airport')
plt.show()

# COMMAND ----------

plt.figure(figsize=(13.5,6))
ax = sns.displot(data=plot_cropped_eda_data, x="delay_minutes", hue="origin_airport_code", col="origin_airport_code", bins=50, linewidth = 0.5, kde=True)
plt.subplots_adjust(top=0.85)
plt.suptitle('Departure delays by Airport (15-400 minutes)')
plt.show()

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



# COMMAND ----------

hourly_agg = joined_eda.withColumn("hour_of_day", substring('planned_dep_time',1,2))\
        .groupBy('origin_ICAO', 'hour_of_day')\
        .agg(count('dest_ICAO').alias('n_flights'),\
             pyspark_sum('dep_is_delayed').alias('n_delayed'),\
             (pyspark_sum('dep_is_delayed')/count('dep_is_delayed') * 100).alias('pct_origin_dow_delayed'),\
             mean('delay_minutes').alias('mean_origin_dow_delay'))\
        .withColumn("sort_order", concat(col('hour_of_day')).cast(IntegerType()))\
        .orderBy('sort_order', ascending=1)
plot_hourly_agg = hourly_agg.toPandas()

# COMMAND ----------

weekly_agg = joined_eda.withColumn("hour_of_day", substring('planned_dep_time',1,2))\
        .groupBy('origin_ICAO', 'day_of_week', 'hour_of_day')\
        .agg(count('dest_ICAO').alias('n_flights'),\
             pyspark_sum('dep_is_delayed').alias('n_delayed'),\
             (pyspark_sum('dep_is_delayed')/count('dep_is_delayed') * 100).alias('pct_origin_dow_delayed'),\
             mean('delay_minutes').alias('mean_origin_dow_delay'))\
        .withColumn("sort_order", concat(col('hour_of_day'), col('day_of_week')).cast(IntegerType()))\
        .orderBy('sort_order', ascending=1)
plot_weekly_agg = weekly_agg.toPandas()

# COMMAND ----------

plt.figure(figsize=(13.5,6))
ax = sns.barplot(data=plot_weekly_agg, y="pct_origin_dow_delayed", x="day_of_week",palette="ch:.25",
                 ci=None, linewidth = 0.5)
plt.xlabel('Departure Day of Week')
plt.ylabel('Percent Flights Delayed (%)')
plt.title('Percentage Departure Delays by Weekday')
plt.show()

# COMMAND ----------

plt.figure(figsize=(13.5,6))
ax = sns.barplot(data=plot_hourly_agg, y="pct_origin_dow_delayed", x="hour_of_day", palette="ch:.25",
                 ci=None, linewidth = 0.5)
plt.xlabel('Departure Hour (local time)')
plt.ylabel('Percent Flights Delayed (%)')
plt.title('Percentage Departure Delays by Hour')
plt.show()

# COMMAND ----------

plt.figure(figsize=(13.5,6))
ax = sns.barplot(data=plot_weekly_agg, y="pct_origin_dow_delayed", x="day_of_week",
                 hue='hour_of_day', palette="ch:.25",
                 ci=None, linewidth = 0.5)
ax.legend_.remove()
plt.xlabel('Departure Day of Week')
plt.ylabel('Percent Flights Delayed (%)')
plt.title('Percentage Departure Delays by Weekday and Hour')
plt.show()


# COMMAND ----------



# COMMAND ----------

ax = sns.jointplot(data=joined_eda.filter(col("delay_minutes") >= 15).filter(col("delay_minutes") < 400)\
      .withColumn("planned_dep_time", col("planned_dep_time").cast(IntegerType())).toPandas(),
                   y="planned_dep_time", x="delay_minutes", hue="origin_airport_code",
                   height = 10, linewidth = 0.5)
ax.set_axis_labels('Departure Delay (minutes)', 'Departure Local Time (HHMM)')
ax.fig.suptitle('Departure delays by Time of Day and Airport')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3 - Feature Engineering

# COMMAND ----------

# Load Q1 2015 subset if necessary
joined_eda = spark.read.parquet(f"{blob_url}/df_airlines_weather_origin_and_dest_v1/*")

# Load full dataset if necessary
# joined_eda = spark.read.parquet(f"{blob_url}/<>/*")

# COMMAND ----------

df_full_airline_stations = spark.read.parquet(f"{blob_url}/full_airlines_stations_v0/*")
df_full_airline_stations.count()

# COMMAND ----------

df_full_clean = spark.read.parquet(f"{blob_url}/airlines_weather_origin_full_v0/*")
df_full_clean.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 3.1 - Aggregate Features

# COMMAND ----------

# MAGIC %md
# MAGIC #### Explore aggregations

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

# MAGIC %md FILL OUT: Narrative

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
# MAGIC #### Join features

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

# join
eda_with_aggs = join_aggs(joined_eda, agg_origin, agg_dest, agg_route, agg_origin_state, agg_dest_state)

# write
eda_with_aggs.write.mode('overwrite').parquet(f"{blob_url}/eda_with_aggs_v1")

# COMMAND ----------

eda_with_aggs = spark.read.parquet(f"{blob_url}/eda_with_aggs_v1/*")
display(eda_with_aggs)

# COMMAND ----------

full_join_2015  = spark.read.parquet(f"{blob_url}/full_join_2015_v0/*").cache()

# origin aggregates
agg_origin = aggregate_keys(full_join_2015,
                            ['origin_ICAO'],
                            'from_origin')
display(agg_origin)

# dest aggregates
agg_dest = aggregate_keys(full_join_2015,
                          ['dest_ICAO'],
                          'to_dest')
display(agg_dest)

# route (origin + destination) aggregates
agg_route = aggregate_keys(full_join_2015,
                           ['origin_ICAO',
                           'dest_ICAO',
                           'origin_longitude',
                           'origin_latitude',
                           'dest_longitude',
                           'dest_latitude'],
                           'for_route')
display(agg_route)

# state aggregates
agg_origin_state = aggregate_keys(full_join_2015,
                           ['origin_state'],
                           'from_state')
display(agg_origin_state)
agg_dest_state = aggregate_keys(full_join_2015,
                         ['dest_state'],
                         'to_state')

# join
full_2015_with_aggs = join_aggs(full_join_2015, agg_origin, agg_dest, agg_route, agg_origin_state, agg_dest_state)

# write
full_2015_with_aggs.write.mode('overwrite').parquet(f"{blob_url}/full_2015_with_aggs_v0")


# COMMAND ----------

full_join  = spark.read.parquet(f"{blob_url}/full_join_2015_v0/*")\
             .union(spark.read.parquet(f"{blob_url}/full_join_2016_v0/*"))\
             .union(spark.read.parquet(f"{blob_url}/full_join_2017_v0/*"))\
             .union(spark.read.parquet(f"{blob_url}/full_join_2018_v0/*"))\
             .union(spark.read.parquet(f"{blob_url}/full_join_2019_v0/*")).cache()

# origin aggregates
agg_origin = aggregate_keys(full_join,
                            ['origin_ICAO'],
                            'from_origin')

# dest aggregates
agg_dest = aggregate_keys(full_join,
                          ['dest_ICAO'],
                          'to_dest')

# route (origin + destination) aggregates
agg_route = aggregate_keys(full_join,
                           ['origin_ICAO',
                           'dest_ICAO',
                           'origin_longitude',
                           'origin_latitude',
                           'dest_longitude',
                           'dest_latitude'],
                           'for_route')

# state aggregates
agg_origin_state = aggregate_keys(full_join,
                           ['origin_state'],
                           'from_state')
agg_dest_state = aggregate_keys(full_join,
                         ['dest_state'],
                         'to_state')

# join
full_join_with_aggs = join_aggs(full_join, agg_origin, agg_dest, agg_route, agg_origin_state, agg_dest_state)

# write
full_join_with_aggs.write.mode('overwrite').parquet(f"{blob_url}/full_join_with_aggs_v0")


# COMMAND ----------



# COMMAND ----------

df_airlines_full = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*")
print(f"The raw airlines dataset contains {df_airlines_full.count()} records, with {len(df_airlines_full.columns)} columns.")

# COMMAND ----------

full_no_duplicates = df_airlines_full.dropDuplicates(['FL_DATE', 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'CRS_DEP_TIME'])
print(f"After dropping duplicates, the raw airlines dataset contains {full_no_duplicates.count()} records, with {len(full_no_duplicates.columns)} columns.")
display(full_no_duplicates.limit(10))

# COMMAND ----------

full_join_with_aggs = spark.read.parquet(f"{blob_url}/full_join_with_aggs_v0/*")
print(f"The fully joined dataset contains {full_join_with_aggs.count()} records, with {len(full_join_with_aggs.columns)} columns.")
display(full_join_with_aggs.limit(10))

# COMMAND ----------

full_join_with_aggs_d = full_join_with_aggs.dropDuplicates(['dt', 'tail_num', 'flight_num', 'planned_dep_time'])
print(f"After dropping duplicates, the fully joined dataset contains {full_join_with_aggs_d.count()} records, with {len(full_join_with_aggs_d.columns)} columns.")
display(full_join_with_aggs_d.limit(10))

# COMMAND ----------

#update data types for models
full_join_with_aggs_no_bool = full_join_with_aggs_d.withColumn('canceled',col('canceled').cast(StringType())) \
                                                    .withColumn('is_diverted',col('is_diverted').cast(StringType())) \
                                                    .withColumn('div_reached_dest',col('div_reached_dest').cast(StringType())) \
                                                    .withColumn('dep_is_delayed',col('dep_is_delayed').cast(StringType())) \
                                                    .drop(col("index_id"))

display(full_join_with_aggs_no_bool.limit(10))

# COMMAND ----------

#add new index
full_join_with_aggs_new_ind = full_join_with_aggs_no_bool.withColumn('index_id', row_number().over(Window.orderBy('dt', 'planned_dep_time', 'tail_num', 'flight_num'))) 

display(full_join_with_aggs_new_ind.limit(10))

# COMMAND ----------

#write to parquet
full_join_with_aggs_new_ind.write.mode('overwrite').parquet(f"{blob_url}/full_2015_with_aggs_v2_no_dupes")

# COMMAND ----------

# TIME BETWEEN FLIGHTS TEST - I NEED TO CHECK IF VARIABLES MATCH IN HERE 
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number 

# Group by tail number then sort by actual arrival time 
win_ind = Window.partitionBy('tail_num').orderBy('actual_arr_utc')
# Get the prior actual arrival time of each flight
# Calculate the hours in between prior actual arrival time and planned departure time 
# Set in between flight hours > 2 to 0 and anything else to 1 - (should negatives have a separate indicator? like 2 or 3 or -1?)
test = df_a1.withColumn('prev_actual_arr_utc', f.lag('actual_arr_utc',1, None).over(win_ind))\
            .withColumn('inbtwn_fl_hrs', (f.unix_timestamp('planned_departure_utc') - f.unix_timestamp('prev_actual_arr_utc'))/60/60)\
            .withColumn('poten_for_del', f.when((f.col('inbtwn_fl_hrs') < 2), 1).otherwise(0))\


display(test)

# COMMAND ----------

# Change time to cyclical - I NEED TO CHECK IF VARIABLES MATCH IN HERE 
df_a1 = df_airlines_clean\
        .withColumn('quarter_cos', f.cos((f.col('quarter').cast(IntegerType()) - 1)*(2*np.pi/4)))\
        .withColumn('quarter_sin', f.sin((f.col('quarter').cast(IntegerType()) - 1)*(2*np.pi/4)))\
        .withColumn('month_cos', f.cos((f.col('month').cast(IntegerType()) - 1)*(2*np.pi/12)))\
        .withColumn('month_sin', f.sin((f.col('month').cast(IntegerType()) - 1)*(2*np.pi/12)))\
        .withColumn('hour_cos', f.cos((hour(col('planned_departure_utc')).cast(IntegerType())-1)*(2*np.pi/24)))\
        .withColumn('hour_sin', f.sin((hour(col('planned_departure_utc')).cast(IntegerType())-1)*(2*np.pi/24)))\

# day of the week? 

display(df_a1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 3.2 - Window Calculations

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC ### Mean imputation

# COMMAND ----------


df = spark.read.parquet(f"{blob_url}/airlines_eda_v0/*").cache()
display(df)
col_name = 'weather_delay'

# calculate mean for column after dropping nulls
col_mean = df.select(col(col_name)).na.drop().select(mean(col(col_name)).alias('mean')).collect()[0]['mean']

# replace nulls with mean
replaced = df.withColumn(col_name, when(col(col_name).isNull(), col_mean).otherwise(col(col_name)))

display(replaced)

# COMMAND ----------

# MAGIC %md ## Section 4 - Algorithm Exploration

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Model 1 Baseline: Logistic Regression
# MAGIC 
# MAGIC For the baseline model, we chose to implement a logistic regression model since it is easy to implement and efficient to train. It can help us determine feature importance and relationship between features by measuring the coefficient size and direction of association. 
# MAGIC 
# MAGIC Although regression models are effective in determining baseline relationship and are efficient to train, one of the major limitations is that it assumes linearity between the dependent and independent variables. This makes it difficult to determine complex relationships that is typically found in real world situations. Logistic regression also requires low or no multicollinearity between the variables. If a dataset has high dimensions, it can lead to overfitting. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Model 2: Random Forest
# MAGIC 
# MAGIC For our second model, we chose to implement the random forest classifier. It is an ensemble of decision trees, which has the power to handle higher dimensionality and can include features with lower correlation that may need to be excluded from other models, such as logistic regression. It can handle many variables and identify the most significant ones. Other advantages of random forest is that the features do not need to be encoded or scaled and it has an effective method for estimating null values. It can maintain accuracy even when big portions of data are missing. 

# COMMAND ----------

# MAGIC %md ## Section 5 - Algorithm Implementation

# COMMAND ----------

# MAGIC %md ## Section 6 - Conclusions

# COMMAND ----------

# MAGIC %md ## Section 7 - Application of Course Concepts
