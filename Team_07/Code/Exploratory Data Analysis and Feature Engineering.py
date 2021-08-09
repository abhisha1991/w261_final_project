# Databricks notebook source
# MAGIC %md
# MAGIC # Exploratory Data Analysis and Feature Engineering
# MAGIC ### Section 0 - Environment Set-up

# COMMAND ----------

from pyspark.sql.functions import col, substring, split, when, lit, max as pyspark_max, min as pyspark_min, countDistinct, count, mean, sum as pyspark_sum, expr, to_utc_timestamp, to_timestamp, concat, length, unix_timestamp
from pyspark.sql import SQLContext
from pyspark.sql.types import IntegerType, StringType, BooleanType, DateType, DoubleType
import pandas as pd
from gcmap import GCMapper, Gradient
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar

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
# MAGIC ### Section 1 - Cleaning individual tables

# COMMAND ----------

# MAGIC %md
# MAGIC We'll do our cleaning and preparation with a 3 month subset dataset in order to test the functions before applying to a larger dataset. We start by loading and examining the subset:

# COMMAND ----------

# Load the full airline dataset
df_airlines_full = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*")

# Load the full weather dataset
df_weather_full = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*")

# Load the weather stations data
df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")

# Load 2015 Q1 for Flights
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")

# Load the 2015 Q1 for Weather
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-04-01T00:00:00.000")


# Describe dataset sizes
print("\033[1mFull Datasets:\033[0m")
print(f"The full dataset for airlines contains {df_airlines_full.count()} records, with {len(df_airlines_full.columns)} columns.")
print(f"The full dataset for weather recordings contains {df_weather_full.count()} records, with {len(df_weather_full.columns)} columns.")
print(f"The NOAA weather station dataset contains {df_stations.count()} records, with {len(df_stations.columns)} columns.")

# Describe dataset sizes
print("\n")
print("\033[1mSubset Datasets:\033[0m")
print(f"The Q1 2015 dataset for ORD and ATL airports contains {df_airlines.count()} records, with {len(df_airlines.columns)} columns.")
print(f"The Q1 2015 NOAA dataset for weather recordings contains {df_weather.count()} records, with {len(df_weather.columns)} columns.")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Airline Flight Data
# MAGIC We begin with the flights table in order to better understand the available features. To start, we examine the first 10 records, along with the provided schema.

# COMMAND ----------

display(df_airlines.limit(10))
df_airlines.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC **Airline Data Cleaning Summary**
# MAGIC 
# MAGIC The airlines dataset is robust, however, some features are unnecessary for our purpose. An overview of the cleaning steps for the airline table are as follows:
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
# MAGIC Below, we lay out the function for cleaning the data:

# COMMAND ----------

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
     , string(CASE
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

  
  # add timezones
  df_with_tz = spark.sql(query)\
      .withColumn('planned_departure_utc',
                  to_utc_timestamp(to_timestamp(concat(col('dt'), col('planned_dep_time')), 'yyyy-MM-ddHHmm'), col('origin_timezone')))\
      .withColumn('time_at_prediction_utc',
                  to_utc_timestamp(to_timestamp(concat(col('dt'), col('planned_dep_time')), 'yyyy-MM-ddHHmm'), col('origin_timezone'))\
                  + expr('INTERVAL -2 HOURS'))\
  
  return df_with_tz

# COMMAND ----------

# MAGIC %md
# MAGIC This function is used to clean the subset 3 month dataset, as well as the full airlines data. We save these intermediate tables for efficiency.

# COMMAND ----------

# clean the 3 month dataset
df_airlines_clean = clean_airline_data(df_airlines)

# write to parquet
df_airlines_clean.write.mode('overwrite').parquet(f"{blob_url}/airlines_eda_v1")

# clean full dataset
df_airlines_full_clean = clean_airline_data(df_airlines_full)

# write intermediate table
df_airlines_full_clean.write.mode('overwrite').parquet(f"{blob_url}/airlines_clean_full_v1")

# COMMAND ----------

# display
display(df_airlines_clean_full.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Weather Station Location Data
# MAGIC Next, we explored the weather station table to better understand the available features. To start, the first 10 records are printed, as well as the provided schema.

# COMMAND ----------

display(df_stations.limit(10))
df_stations.printSchema()

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

# clean station data
df_stations_clean = df_stations.filter(col("distance_to_neighbor") == 0).select('station_id','neighbor_call')

# display
display(df_stations_clean.limit(10))

# write full dataset
df_stations_clean.write.mode('overwrite').parquet(f"{blob_url}/stations_clean_full")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Weather Observation Data
# MAGIC Next, we'll take a peek at our NOAA [weather dataset](https://www.ncdc.noaa.gov/orders/qclcd/)
# MAGIC 
# MAGIC Because we started our data exploration with just the 3 month data to make it faster, we subsetted our weather data to just ATL and ORD since our 3 month airlines dataset is also subsetted to only departures from those airlines. Subsetting this dataset here, and in the full dataset speeds up the join considerably. The full dataset contains all of the weather stations around the world, and, as you'll see later, we subset it to only include weather stations in the US.

# COMMAND ----------

display(df_weather_full.limit(10))

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

def replace(column, value):
  return when(column != value, column).otherwise(lit(None))

def unpack_and_clean_weather(df, origin_stations):
  # drop measurements from irrelevant locations
  df = df.filter(col('STATION').isin(origin_stations))
  
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

# MAGIC %md
# MAGIC We test this on the 3 month subset data:

# COMMAND ----------

# list unique origin airports
origin_ICAO = [row[0] for row in df_airlines_clean.select(col('origin_ICAO')).dropDuplicates().collect()]

# list stations ids of unique origin airports
origin_stations = [row[0] for row in df_stations_clean.filter(col('neighbor_call').isin(origin_ICAO)).select(col('station_id')).dropDuplicates().collect()]

# subset for relevant airports and clean
df_weather_clean = unpack_and_clean_weather(df_weather, origin_stations)

# write
df_weather_clean.write.mode('overwrite').parquet(f"{blob_url}/weather_eda_v1")

# display
display(df_weather_clean.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC We apply it to the full weather data

# COMMAND ----------

# list unique origin airports
origin_ICAO_full = [row[0] for row in df_airlines_clean_full.select(col('origin_ICAO')).dropDuplicates().collect()]

# list stations ids of unique origin airports
origin_stations_full = [row[0] for row in df_stations_clean.filter(col('neighbor_call').isin(origin_ICAO_full)).select(col('station_id')).dropDuplicates().collect()]

# clean full dataset
df_weather_full_clean = unpack_and_clean_weather(df_weather_full, origin_stations_full)

# write
df_weather_full_clean.write.mode('overwrite').parquet(f"{blob_url}/weather_clean_full_v1")

# COMMAND ----------

# size savings
print(f"Number of weather records (raw weather data): {df_weather_full.count()}")
print(f"Number of weather records from origin airports (cleaned weather data): {df_weather_full_clean.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 2 - Joining the data

# COMMAND ----------

# MAGIC %md #### Stations to Airlines
# MAGIC Due to the ICAO codes joined to the airlines from the OpenAirlines data sources, this join is straightforward. The stations data may be joined by this code (`neighbor_call` to `ICAO`). This is done twice: once for the origin stations and once for the destination stations.

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

# MAGIC %md Again, we test the function on the subset 3 month dataset:

# COMMAND ----------

# read if necessary
df_airlines_clean = spark.read.parquet(f"{blob_url}/airlines_eda_v1/*")
df_stations_clean = spark.read.parquet(f"{blob_url}/stations_clean_full/*")

# join
df_airlines_stations = join_stations_to_airlines(df_airlines_clean, df_stations_clean)
display(df_airlines_stations.limit(10))

# write
df_airlines_stations.write.mode('overwrite').parquet(f"{blob_url}/airlines_stations_v0")

# COMMAND ----------

# MAGIC %md 
# MAGIC We apply the join to the full data and write the intermediate table to parquet:

# COMMAND ----------

# read if necessary
df_airlines_full_clean = spark.read.parquet(f"{blob_url}/airlines_clean_full_v1/*")
df_stations_clean = spark.read.parquet(f"{blob_url}/stations_clean_full/*")

# join stations
df_airlines_stations_full = join_stations_to_airlines(df_airlines_full_clean, df_stations_clean)

# write
df_airlines_stations_full.write.mode('overwrite').parquet(f"{blob_url}/full_airlines_stations_v0")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Weather to Airport Join
# MAGIC 
# MAGIC The weather to airport join takes the station and airport dataset, the weather data, and the airport a prefix (either `origin_` or `dest_`). It first prepares the weather dataset by droping duplicates and prepending the column names with the specified the prefix. Next, the weather is joined to the airlines using the station ids and the prediction time, with a 2-hour buffer for the weather time. This time range join is because the weather data is rarely recorded at exact same time as a flight prediction time. This does cause duplicate flight measurements to be generated, when there are more than one weather measurement in the 2-hour window, so we need to address this. A new feature indicating the difference between the prediction time and weather recording time is generated (`weather_offset_minutes`). Finally, we are able to drop duplicate flight records by ordering our datset by index ID (chronological flights) and weather offset (increasing) and dropping on these keys. This results in flight records now having the closest weather measurement recorded within 2 hours of the prediction time. The function is shown below:

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

# MAGIC %md We'll again first apply this join to our subset 3 month datasets:

# COMMAND ----------

# read if necessary
df_weather_clean = spark.read.parquet(f"{blob_url}/weather_eda_v0/*")
df_airlines_stations_clean = spark.read.parquet(f"{blob_url}/airlines_stations_v0/*")

# join origin weather
df_airlines_weather_origin = join_weather_to_airlines(df_airlines_stations_clean, df_weather_clean, "origin_")

# display
display(df_airlines_weather_origin.limit(10))

# write
df_airlines_weather_origin.write.mode('overwrite').parquet(f"{blob_url}/airlines_weather_origin_v0")

# COMMAND ----------

# MAGIC %md 
# MAGIC While we were able to join the airlines and *stations* together in one go, the weather is significantly more computationally expensive. We opt to split the weather join into a separate join for each year, in order to reduce the risk of the full join being interrupted and needing to start over. Each year is written to parquet for safekeeping, with the next step being to union these datasets together for a comprehensive set. 
# MAGIC 
# MAGIC The code below does not show the duration of the commands, however, we recorded the time each join took to run. We were unable to note how many nodes the computation had access to due to fluctuations in availability during this time period.
# MAGIC - 2015 join: 1.69 hours
# MAGIC - 2016 join: 1.58 hours
# MAGIC - 2017 join: 1.47 hours
# MAGIC - 2018 join: 1.72 hours
# MAGIC - 2019 join: 1.59 hours

# COMMAND ----------

# read datasets if necessary
df_airlines_stations_full = spark.read.parquet(f"{blob_url}/full_airlines_stations_v0/*").cache()
df_weather_full_clean = spark.read.parquet(f"{blob_url}/weather_clean_full/*").cache()

# join
full_join_2015 = join_weather_to_airlines(df_airlines_stations_full.filter(col('time_at_prediction_utc') < "2016-01-01T00:00:00.000"),
                                          weather_clean_full_v1.filter(col('DATE') < "2016-01-01T00:00:00.000"),
                                          "origin_")


# write
full_join_2015.write.mode('overwrite').parquet(f"{blob_url}/full_join_2015_v0")

# COMMAND ----------

# join 2016 weather and airports
full_join_2016 = join_weather_to_airlines(df_airlines_stations_full.filter((col('time_at_prediction_utc') >= "2016-01-01T00:00:00.000") &\
                                                               (col('time_at_prediction_utc') < "2017-01-01T00:00:00.000")),
                                          weather_clean_full_v1.filter((col('DATE') >= "2016-01-01T00:00:00.000") &\
                                                                       (col('DATE') < "2017-01-01T00:00:00.000")),
                                          "origin_")


# write
full_join_2016.write.mode('overwrite').parquet(f"{blob_url}/full_join_2016_v0")

# COMMAND ----------

# join 2017 weather and airports
full_join_2017 = join_weather_to_airlines(df_airlines_stations_full.filter((col('time_at_prediction_utc') >= "2017-01-01T00:00:00.000") &\
                                                               (col('time_at_prediction_utc') < "2018-01-01T00:00:00.000")),
                                          weather_clean_full_v1.filter((col('DATE') >= "2017-01-01T00:00:00.000") &\
                                                                       (col('DATE') < "2018-01-01T00:00:00.000")),
                                          "origin_")


# write
full_join_2017.write.mode('overwrite').parquet(f"{blob_url}/full_join_2017_v0")

# COMMAND ----------

# join 2018 weather and airports
full_join_2018 = join_weather_to_airlines(df_airlines_stations_full.filter((col('time_at_prediction_utc') >= "2018-01-01T00:00:00.000") &\
                                                               (col('time_at_prediction_utc') < "2019-01-01T00:00:00.000")),
                                          weather_clean_full_v1.filter((col('DATE') >= "2018-01-01T00:00:00.000") &\
                                                                       (col('DATE') < "2019-01-01T00:00:00.000")),
                                          "origin_")


# write
full_join_2018.write.mode('overwrite').parquet(f"{blob_url}/full_join_2018_v0")

# COMMAND ----------

# join 2019 weather and airports
full_join_2019 = join_weather_to_airlines(df_airlines_stations_full.filter((col('time_at_prediction_utc') >= "2019-01-01T00:00:00.000")),
                                          weather_clean_full_v1.filter((col('DATE') >= "2019-01-01T00:00:00.000")),
                                          "origin_")


# write
full_join_2019.write.mode('overwrite').parquet(f"{blob_url}/full_join_2019_v0")

# COMMAND ----------

# union all datasets
full_join  = spark.read.parquet(f"{blob_url}/full_join_2015_v0/*")\
             .union(spark.read.parquet(f"{blob_url}/full_join_2016_v0/*"))\
             .union(spark.read.parquet(f"{blob_url}/full_join_2017_v0/*"))\
             .union(spark.read.parquet(f"{blob_url}/full_join_2018_v0/*"))\
             .union(spark.read.parquet(f"{blob_url}/full_join_2019_v0/*")).cache()

# write
full_join.write.mode('overwrite').parquet(f"{blob_url}/full_join_2015_2019_v0")

# COMMAND ----------

display(full_join.limit(10))

# COMMAND ----------

# MAGIC %md ## Section 3 - Feature Engineering

# COMMAND ----------

# MAGIC %md #### Aggregate Features

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

# MAGIC %md First we'll generate aggregates for the flight routes and origin/destinations:

# COMMAND ----------

full_join  = spark.read.parquet(f"{blob_url}/full_join_2015_2019_v0/*").cache()

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

# origin state aggregates
agg_origin_state = aggregate_keys(full_join,
                           ['origin_state'],
                           'from_state')

# dest state aggregates
agg_dest_state = aggregate_keys(full_join,
                         ['dest_state'],
                         'to_state')


# COMMAND ----------

# MAGIC %md Next, we'll join these features back onto the dataset:

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
full_join_with_aggs = join_aggs(full_join, agg_origin, agg_dest, agg_route, agg_origin_state, agg_dest_state)

# write
full_join_with_aggs.write.mode('overwrite').parquet(f"{blob_url}/full_join_with_aggs_v0")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Window Calculations
# MAGIC 
# MAGIC 1. Previous Flight Delay (prev_fl_del)
# MAGIC   - Prior flight performance can have an impact on flight departure times. We tracked flights by their tail numbers and arrival times in order to determine if the previous flight was delayed. If the previous flight was delayed, we set the indicator to 1 and if it was not delayed, we set the indicator to 0. 
# MAGIC  
# MAGIC 2. Potential for Delay (poten_for_del)
# MAGIC   - Previous flight arrival times can also have an impact on flight departure times. After landing, the plane needs to be refueled, cleaned, and maintenanced. The cabin crew and pilots may need to be changed. The more time in between the flight’s arrival time and next departure time, the less likely the flight departure will be delayed. We calculated the time in between flights by tracking the tail number and actual arrival time and created an indicator where flights with more than 2 hours in between flights were indicated with a 1 and less than 2 hours were indicated with a 0. Flights that were canceled, diverted, or did not have a previous flight were null and were indicated with a -1. 
# MAGIC  
# MAGIC 3. Indicators for Average Delay 2-4 Hours Prior to Planned Departure 
# MAGIC   - At times, there may be certain issues, such as security, weather, maintenance, etc., that can affect flight performance at the airport or carrier level. We created a few indicator variables to capture the average delay minutes 2-4 hours prior to planned departure times. If the average delay 2-4 hours prior is less than 15 minutes, it is assigned a 0, and if it is greater than 15 minutes, it is assigned a 1. There are null values if there are no flights in the 2-4 hour measurement window. We assign nulls a -1. 
# MAGIC 
# MAGIC     - Origin Airport Average Delay 2-4 Hours Prior (oa_del_ind)
# MAGIC       - This feature is created based on calculating the average delay minutes 2-4 before planned departure at the origin airport. 
# MAGIC 
# MAGIC     - Destination Airport Average Delay 2-4 Hours Prior (da_del_ind)
# MAGIC       - For this feature, we are determining average arrival delay at the departure 2-4 hours prior to the departure time at the origin airport. The concept is similar to the previous feature. If there is an issue that is affecting the destination airport, then flights to that airport may be delayed. 
# MAGIC 
# MAGIC     - Carrier Average Delay 2-4 Hours Prior by Origin Airport (carrier_del_ind)
# MAGIC       - If a specific airline is low on maintenance or cleaning staff on a certain day, it may impact departure times. We created this feature by calculating the average delay minutes by carrier at the origin airport. 
# MAGIC 
# MAGIC     - Five Delay Categories Average Delay 2-4 Hours Prior by Origin Airport (security_window_del_ind, nas_window_del_ind, carrier_window_del_ind, weather_window_del_ind, late_ac_window_del_ind)
# MAGIC       - There were five features, security, NAS, carrier, weather, and late aircraft delay, that provided the total delay minutes for the specific category. We created five features to capture the average delay minutes of each category by origin airport   
# MAGIC  

# COMMAND ----------

import pyspark.sql.functions as f
from pyspark.sql.window import Window

# access data
data = spark.read.parquet(f"{blob_url}/full_join_with_aggs_v0/*").cache()

# ensure data is in appropriate type
data = data.withColumn("dep_is_delayed",data["dep_is_delayed"].cast(IntegerType()))\
          .withColumn("security_delay",data["security_delay"].cast(IntegerType()))\
          .withColumn("is_diverted",data["is_diverted"].cast(IntegerType()))\
          .withColumn("delay_minutes",data["delay_minutes"].cast(IntegerType()))\
          .withColumn("arr_delay_minutes",data["arr_delay_minutes"].cast(IntegerType()))\
          .withColumn("carrier_delay",data["carrier_delay"].cast(IntegerType()))\
          .withColumn("weather_delay",data["weather_delay"].cast(IntegerType()))\
          .withColumn("nas_delay",data["nas_delay"].cast(IntegerType()))

####################################### New feature 1: potential for delay  ######################################
####################################### New feature 2: prev flight delay ind  ####################################
#Potential for delay: if flight arrives > 2 hrs before departure likelihood for delay is smaller
# pad if length is not equal to 4 
data = data.withColumn('actual_arr_time_pad', f.lpad(data['actual_arr_time'], 4, '0'))\
            .withColumn('actual_dep_time_pad', f.lpad(data['actual_dep_time'], 4, '0'))

# Convert actual arrival time and actual departure time to utc 
data = data.withColumn('actual_arr_utc', to_utc_timestamp(to_timestamp(concat(col('dt'), col('actual_arr_time_pad')), 'yyyy-MM-ddHHmm'), col('dest_timezone')))\
            .withColumn('actual_dep_utc', to_utc_timestamp(to_timestamp(concat(col('dt'), col('actual_dep_time_pad')), 'yyyy-MM-ddHHmm'), col('origin_timezone')))

# update rows where actual arrival date is greater than departure date 
data = data.withColumn("actual_arr_utc", f.when((data.actual_arr_utc < data.actual_dep_utc),(f.from_unixtime(f.unix_timestamp('actual_dep_utc') + (data.actual_duration*60)))).otherwise(data.actual_arr_utc))

# Group by tail number then sort by actual arrival time 
win_ind = Window.partitionBy('tail_num').orderBy('actual_arr_utc')
# Get the prior actual arrival time of each flight
# Calculate the hours in between prior actual arrival time and planned departure time 
# Categorize flight gap (>2 hours = 0, between 0-1 = 1, <0 = 2, and null = -1)
data = data.withColumn('prev_actual_arr_utc', f.lag('actual_arr_utc',1, None).over(win_ind))\
             .withColumn('prev_fl_del', f.lag('dep_is_delayed',1, None).over(win_ind))\
            .withColumn('inbtwn_fl_hrs', (f.unix_timestamp('planned_departure_utc') - f.unix_timestamp('prev_actual_arr_utc'))/60/60)\
            .withColumn('poten_for_del', expr("CASE WHEN inbtwn_fl_hrs < 2 THEN '0'" + "WHEN inbtwn_fl_hrs IS NULL THEN '-1'" + "ELSE '1' END"))


###################### New feature 3: Origin aiport avg dep delay 2-4 hrs prior to planned dept   #######################
# if there are serious weather issues or some "global" or "local" issue is occuring, then all flights should be delayed 

# Group by origin airport then sort by planned departure time to get avg dep delay by origin airport 2-4 hours before planned departure
win_ind_airport = Window.partitionBy('origin_airport_code')\
                         .orderBy(f.unix_timestamp('planned_departure_utc'))\
                         .rangeBetween(-14400, -7200)
data = data.withColumn('oa_avg_del2_4hr', f.round(f.avg('delay_minutes').over(win_ind_airport),2))\
            .withColumn('oa_avg_del_ind', expr("CASE WHEN oa_avg_del2_4hr < 15 THEN '0'" + "WHEN oa_avg_del2_4hr IS NULL THEN '-1'" + "ELSE '1' END"))


############### New feature 4: Carrier avg dep delay by origin airport 2-4 hrs prior to planned dept   ################# 
# Group by origin airport then sort by planned departure time to get avg dep delay by carrier 2-4 hours before planned departure at origin airport 
win_ind_carrier = Window.partitionBy([col('origin_airport_code'), col('carrier')])\
                         .orderBy(f.unix_timestamp('planned_departure_utc'))\
                         .rangeBetween(-14400, -7200)
data = data.withColumn('carrier_avg_del2_4hr', f.round(f.avg('delay_minutes').over(win_ind_carrier),2))\
            .withColumn('carrier_avg_del_ind', expr("CASE WHEN carrier_avg_del2_4hr < 15 THEN '0'" + "WHEN carrier_avg_del2_4hr IS NULL THEN '-1'" + "ELSE '1' END"))


############### New feature 5: Destination avg arrival delay by 2-4 hrs prior to planned dept   ################# 
# Group by destination airport then sort by planned departure time to get avg arr delay at dest airport 2-4 hours before planned departure  
win_ind_dest_airport = Window.partitionBy('dest_airport_code')\
                         .orderBy(f.unix_timestamp('planned_departure_utc'))\
                         .rangeBetween(-14400, -7200)
data = data.withColumn('da_avg_del2_4hr', f.round(f.avg('arr_delay_minutes').over(win_ind_dest_airport),2))\
            .withColumn('da_avg_del_ind', expr("CASE WHEN da_avg_del2_4hr < 15 THEN '0'" + "WHEN da_avg_del2_4hr IS NULL THEN '-1'" + "ELSE '1' END"))


############### New feature 6: Avg delay of 5 delay categories 2-4 hrs prior to planned dept   ################# 
# Unable to use delay categories, so determine the avg delay for each category 2-4 hours prior to planned departure at origin ariport   
win_ind_del = Window.partitionBy('origin_airport_code')\
                         .orderBy(f.unix_timestamp('planned_departure_utc'))\
                         .rangeBetween(-14400, -7200)
data = data.withColumn('weather_window_del', f.round(f.avg('weather_delay').over(win_ind_del),2))\
           .withColumn('carrier_window_del', f.round(f.avg('carrier_delay').over(win_ind_del),2))\
           .withColumn('security_window_del', f.round(f.avg('security_delay').over(win_ind_del),2))\
           .withColumn('late_ac_window_del', f.round(f.avg('late_aircraft_delay').over(win_ind_del),2))\
           .withColumn('nas_window_del', f.round(f.avg('nas_delay').over(win_ind_del),2))\
           .withColumn('weather_window_del_ind', expr("CASE WHEN weather_window_del < 15 THEN '0'" + "WHEN weather_window_del IS NULL THEN '-1'" + "ELSE '1' END"))\
           .withColumn('carrier_window_del_ind', expr("CASE WHEN carrier_window_del < 15 THEN '0'" + "WHEN carrier_window_del IS NULL THEN '-1'" + "ELSE '1' END"))\
           .withColumn('security_window_del_ind', expr("CASE WHEN security_window_del < 15 THEN '0'" + "WHEN security_window_del IS NULL THEN '-1'" + "ELSE '1' END"))\
           .withColumn('late_ac_window_del_ind', expr("CASE WHEN late_ac_window_del < 15 THEN '0'" + "WHEN late_ac_window_del IS NULL THEN '-1'" + "ELSE '1' END"))\
           .withColumn('nas_window_del_ind', expr("CASE WHEN nas_window_del < 15 THEN '0'" + "WHEN nas_window_del IS NULL THEN '-1'" + "ELSE '1' END"))

data.write.mode('overwrite').parquet(f"{blob_url}/full_join_with_windows_v0")

# COMMAND ----------

# MAGIC %md ### Holiday and Holiday-Adjacent
# MAGIC Airports typically see the most traffic during the holiday seasons. We captured this information by setting flights that depart on a US holiday to a  “holiday” category. We also set the two days prior and after a holiday to “holiday_adjacent” category since many people travel to a location before the actual holiday, spend time with their family or friends, and then fly back home after the holiday.

# COMMAND ----------

data = spark.read.parquet(f"{blob_url}/full_join_with_windows_v0/*").cache()
data = data.withColumn('holiday', expr("""CASE WHEN dt in ('2015-12-25', '2016-12-25', '2017-12-25', '2018-12-25', '2019-12-25',
                                                         '2015-11-26', '2016-11-24', '2017-11-23', '2018-11-22', '2019-11-28', 
                                                         '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', 
                                                         '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', 
                                                         '2015-07-04', '2016-07-04', '2017-07-04', '2018-07-04', '2019-07-04') THEN 'holiday' """ + 
                                                         """ WHEN dt in ('2015-12-23', '2015-12-24', '2015-12-26', '2015-12-27', '2016-12-23', 
                                                                    '2016-12-24', '2016-12-26', '2016-12-27', '2017-12-23', '2017-12-24', 
                                                                    '2017-12-26', '2017-12-27', '2018-12-23', '2018-12-24', '2018-12-26', 
                                                                    '2018-12-27', '2019-12-23', '2019-12-24', '2019-12-26', '2019-12-27', 
                                                                    '2015-11-24', '2015-11-25', '2015-11-27', '2015-11-28', '2016-11-22', 
                                                                    '2016-11-24', '2016-11-25', '2016-11-26', '2017-11-21', '2017-11-22', 
                                                                    '2017-11-24', '2017-11-25', '2018-11-20', '2018-11-21', '2018-11-23', 
                                                                    '2018-11-24', '2019-11-26', '2019-11-27', '2019-11-29', '2019-11-30', 
                                                                    '2015-01-02', '2015-01-03', '2015-12-30', '2015-12-31', '2016-01-02', 
                                                                    '2016-01-03', '2016-12-30', '2016-12-31', '2017-01-02', '2017-01-03', 
                                                                    '2017-12-30', '2017-12-31', '2018-01-02', '2018-01-03', '2018-12-30', 
                                                                    '2018-12-31', '2019-01-02', '2019-01-03', '2019-12-30', '2019-12-31', 
                                                                    '2015-07-02', '2015-07-03', '2015-07-05', '2015-07-06', '2016-07-02', 
                                                                    '2016-07-03', '2016-07-05', '2016-07-06', '2017-07-02', '2017-07-03', 
                                                                    '2017-07-05', '2017-07-06', '2018-07-02', '2018-07-03', '2018-07-05', 
                                                                    '2018-07-06', '2019-07-02', '2019-07-03', '2019-07-05', '2019-07-06') THEN 'holiday_adjacent' """
                                      "ELSE 'non-holiday' END"))

data.write.mode('overwrite').parquet(f"{blob_url}/full_join_with_holidays_v0")

# COMMAND ----------

# MAGIC %md ### Mean Imputation and Null removal
# MAGIC List of features to fill

# COMMAND ----------

features_with_numeric_nulls = [
  'origin_wnd_speed',
  'origin_cig_cloud_agl',
  'origin_vis_dist',
  'origin_tmp_c',
  'origin_dew_c',
  'origin_slp_p'
]

features_with_str_nulls = [
  'origin_wnd_type',
  'origin_cig_cavok',
  'origin_vis_var'
]

# COMMAND ----------

# MAGIC %md This function will fill categorical nulls with the word "NULL" and numeric nulls with the mean of the feature:

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

full_join_with_holidays = spark.read.parquet(f"{blob_url}/full_join_with_holidays_v0/*").cache()
feature_complete = fill_nulls(full_join_with_holidays,
                             features_with_numeric_nulls,
                             features_with_str_nulls)

feature_complete.write.mode('overwrite').parquet(f"{blob_url}/feature_complete_v6")

# COMMAND ----------

# MAGIC %md ### Feature Selection
# MAGIC Let's subset down to only the features we will use in our predictive models:

# COMMAND ----------

model_features = [
  'dep_is_delayed', # outcome
  'canceled',       # outcome classification
  'planned_departure_utc', # datetime for cross validation
  'origin_state', # origin features
  'origin_city',
  'origin_ICAO',
  'origin_altitude',
  'origin_wnd_type', # origin weather
  'origin_wnd_speed',
  'origin_cig_cloud_agl',
  'origin_cig_cavok',
  'origin_vis_dist',
  'origin_vis_var',
  'origin_tmp_c',
  'origin_dew_c',
  'origin_slp_p',
  'dest_state', # dest features
  'dest_city',
  'dest_ICAO',
  'dest_altitude',
  'carrier', # flight features
  'year',
  'quarter',
  'month',
  'day_of_month',
  'day_of_week',
  'dep_hour',
  'arr_hour',
  'planned_duration',
  'flight_distance',
  'distance_group',
  'pct_delayed_from_origin', # aggregates
  'mean_delay_from_origin',
  'pct_delayed_to_dest',
  'mean_delay_to_dest',
  'pct_delayed_for_route',
  'mean_delay_for_route',
  'pct_delayed_from_state',
  'mean_delay_from_state',
  'pct_delayed_to_state',
  'mean_delay_to_state',
  'oa_avg_del_ind',
  'da_avg_del_ind',
  'carrier_avg_del_ind',
  'poten_for_del',
  'prev_fl_del',
  'nas_window_del_ind',
  'weather_window_del_ind',
  'carrier_window_del_ind',
  'security_window_del_ind',
  'late_ac_window_del_ind',
  'holiday'
] + [f"{x}_null" for x in features_with_numeric_nulls]


# final.write.mode('overwrite').parquet(f"{blob_url}/feature_complete_v6")
model_data = final.select(model_features).dropna(how='any').write.mode('overwrite').parquet(f"{blob_url}/model_features_v6")

# COMMAND ----------

# MAGIC %md For these features, we will print any remaining nulls

# COMMAND ----------

from pyspark.sql.functions import isnan, when, count, col
display(model_data)
display(model_data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in model_data.columns if c != "planned_departure_utc"]))
print(model_data.count())

# COMMAND ----------

# MAGIC %md There appear to be nulls in the `dep_is_delayed` outcome variable. Let's take a look:

# COMMAND ----------

display(model_data.filter(col('dep_is_delayed').isNull()))

# COMMAND ----------

# MAGIC %md There are only 4744 nulls out of over 32M. These appear to be data entry issues. Since it is such a small subset, we will opt to drop these rows and write out our final model data.

# COMMAND ----------

model_data.dropna(how='any').write.mode('overwrite').parquet(f"{blob_url}/model_features_v6")
