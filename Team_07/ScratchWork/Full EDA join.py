# Databricks notebook source
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

# join
df_joined = join_stations_to_airlines_historic(df_airlines_clean, df_stations_clean)
display(df_joined.limit(10))

# write
# df_joined.write.mode('overwrite').parquet(f"{blob_url}/airlines_stations_v0")

# COMMAND ----------

# read if necessary
#df_airlines_clean = spark.read.parquet(f"{blob_url}/airlines_eda_v0/*")
#df_stations_clean = spark.read.parquet(f"{blob_url}/stations_clean_full/*")

# join
df_joined = join_stations_to_airlines(df_airlines_clean, df_stations_clean)
display(df_joined.limit(10))

# write
df_joined.write.mode('overwrite').parquet(f"{blob_url}/airlines_stations_v0")

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
df_weather_clean = spark.read.parquet(f"{blob_url}/weather_eda_v0/*")
df_airlines_stations_clean = spark.read.parquet(f"{blob_url}/airlines_stations_v0/*")

display(df_weather_clean.limit(10))

# COMMAND ----------

# join origin weather
df_airlines_weather_origin = join_weather_to_airlines(df_airlines_stations_clean, df_weather_clean, "origin_")

display(df_airlines_weather_origin)

# write
df_airlines_weather_origin.write.mode('overwrite').parquet(f"{blob_url}/airlines_weather_origin_v0")

# COMMAND ----------

# join dest weather
df_airlines_weather_both = join_weather_to_airlines(df_airlines_weather_origin, df_weather_clean, "dest_")

# write
df_airlines_weather_both.write.mode('overwrite').parquet(f"{blob_url}/df_airlines_weather_both_v0")

# COMMAND ----------

# read if necessary
df_weather_full_clean = spark.read.parquet(f"{blob_url}/weather_clean_full/*")
df_airlines_full_clean = spark.read.parquet(f"{blob_url}/airlines_clean_full_v1/*")
df_stations_clean = spark.read.parquet(f"{blob_url}/stations_clean_full/*")

# join stations
df_airlines_stations_full = join_stations_to_airlines(df_airlines_full_clean, df_stations_clean)

# write
df_airlines_stations_full.write.mode('overwrite').parquet(f"{blob_url}/full_airlines_stations_v0")

# COMMAND ----------

# join origin weather
df_airlines_weather_origin_full = join_weather_to_airlines(df_airlines_stations_full, df_weather_full_clean, "origin_")

# write
df_airlines_weather_origin_full.write.mode('overwrite').parquet(f"{blob_url}/airlines_weather_origin_full_v0")

# COMMAND ----------



# COMMAND ----------

import pyspark.sql.functions as f
from pyspark.sql.window import Window

# data = full.withColumn("delay_minutes", data["delay_minutes"].cast(IntegerType()))

data = data.withColumn("dep_is_delayed",data["dep_is_delayed"].cast(IntegerType()))\
          .withColumn("security_delay",data["security_delay"].cast(IntegerType()))\
          .withColumn("is_diverted",data["is_diverted"].cast(IntegerType()))\
          .withColumn("delay_minutes",data["delay_minutes"].cast(IntegerType()))\
          .withColumn("arr_delay_minutes",data["arr_delay_minutes"].cast(IntegerType()))\
          .withColumn("canceled",data["canceled"].cast(IntegerType()))\
          .withColumn("planned_duration",data["planned_duration"].cast(IntegerType()))\
          .withColumn("actual_duration",data["actual_duration"].cast(IntegerType()))\
          .withColumn("num_flights",data["num_flights"].cast(IntegerType()))\
          .withColumn("flight_distance",data["flight_distance"].cast(IntegerType()))\
          .withColumn("carrier_delay",data["carrier_delay"].cast(IntegerType()))\
          .withColumn("weather_delay",data["weather_delay"].cast(IntegerType()))\
          .withColumn("nas_delay",data["nas_delay"].cast(IntegerType()))\
          .withColumn("late_aircraft_delay",data["late_aircraft_delay"].cast(IntegerType()))\
          .withColumn("div_reached_dest",data["div_reached_dest"].cast(IntegerType()))\
          .withColumn("origin_weather_offset_minutes",data["origin_weather_offset_minutes"].cast(IntegerType()))\
          .withColumn("pct_delayed_from_origin",data["pct_delayed_from_origin"].cast(IntegerType()))\
          .withColumn("mean_delay_from_origin",data["mean_delay_from_origin"].cast(IntegerType()))\
          .withColumn("pct_delayed_to_dest",data["pct_delayed_to_dest"].cast(IntegerType()))\
          .withColumn("mean_delay_to_dest",data["mean_delay_to_dest"].cast(IntegerType()))\
          .withColumn("pct_delayed_for_route",data["pct_delayed_for_route"].cast(IntegerType()))\
          .withColumn("mean_delay_for_route",data["mean_delay_for_route"].cast(IntegerType()))\
          .withColumn("pct_delayed_from_state",data["pct_delayed_from_state"].cast(IntegerType()))\
          .withColumn("mean_delay_from_state",data["mean_delay_from_state"].cast(IntegerType()))\
          .withColumn("pct_delayed_to_state",data["pct_delayed_to_state"].cast(IntegerType()))\
          .withColumn("mean_delay_to_state",data["mean_delay_to_state"].cast(IntegerType()))

####################################### New feature 1: time between flights ######################################
####################################### New feature 2: potential for delay  ######################################
####################################### New feature 3: prev flight delay ind  ####################################
#if flight arrives > 2 hrs before departure likelihood for delay is smaller
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
# Set in between flight hours > 2 to 0, between 0-1 as 1, less than 0 as 2, and null as -1 
data = data.withColumn('prev_actual_arr_utc', f.lag('actual_arr_utc',1, None).over(win_ind))\
             .withColumn('prev_fl_del', f.lag('dep_is_delayed',1, None).over(win_ind))\
            .withColumn('inbtwn_fl_hrs', (f.unix_timestamp('planned_departure_utc') - f.unix_timestamp('prev_actual_arr_utc'))/60/60)\
            .withColumn('poten_for_del', expr("CASE WHEN inbtwn_fl_hrs > 2 THEN '0'" + "ELSE '1' END"))
######################### New feature 4: OG aiport avg dep delay 2-4 hrs prior to planned dept   #########################
# if there are serious weather issues or some "global" or "local" issue is occuring, then all flights should be delayed 
# Group by origin airport then sort by planned departure time 
win_ind_airport = Window.partitionBy('origin_airport_code')\
                         .orderBy(f.unix_timestamp('planned_departure_utc'))\
                         .rangeBetween(-14400, -7200)
data = data.withColumn('oa_avg_del2_4hr', f.round(f.avg('delay_minutes').over(win_ind_airport),2))\
            .withColumn('oa_avg_del_ind', expr("CASE WHEN oa_avg_del2_4hr < 15 THEN '0'" + "ELSE '1' END"))
###################### New feature 5: OG aiport avg dep delay by carrier 2-4 hrs prior to planned dept   ###################### 
win_ind_carrier = Window.partitionBy([col('origin_airport_code'), col('carrier')])\
                         .orderBy(f.unix_timestamp('planned_departure_utc'))\
                         .rangeBetween(-14400, -7200)
data = data.withColumn('carrier_avg_del2_4hr', f.round(f.avg('delay_minutes').over(win_ind_carrier),2))\
            .withColumn('carrier_avg_del_ind', expr("CASE WHEN carrier_avg_del2_4hr < 15 THEN '0'" + "ELSE '1' END"))


display(data.cache())
# df_joined.write.mode('overwrite').parquet(f"{blob_url}/airlines_stations_v0")

# COMMAND ----------

from pyspark.sql.functions import isnan, when, count, col, lpad, lag, to_utc_timestamp, to_timestamp

data = spark.read.parquet(f"{blob_url}/airlines_eda_v0/*")

# convert actual arrival time and actual departure time to utc 
data = data.withColumn('actual_dep_utc', to_utc_timestamp(to_timestamp(concat(col('dt'), lpad(col('actual_dep_time'), 4, '0')), 'yyyy-MM-ddHHmm'), col('origin_timezone')).alias('actual_dep_utc'))

# calculate actual arrival time
data = data.withColumn('actual_arr_utc', (unix_timestamp(col('actual_dep_utc')) + col("actual_duration") * 60).cast('timestamp').alias('actual_arr_utc'))


# Group by tail number then sort by actual arrival time 
win_ind = Window.partitionBy('tail_num').orderBy('actual_arr_utc')

# Get the prior actual arrival time of each flight
# Calculate the hours in between prior actual arrival time and planned departure time 
# Set in between flight hours > 2 to 0, between 0-1 as 1, less than 0 as 2, and null as -1 
data = data.withColumn('prev_actual_arr_utc', lag('actual_arr_utc',1, None).over(win_ind))\
           .withColumn('prev_fl_del', lag('dep_is_delayed',1, None).over(win_ind))\
           .withColumn('inbtwn_fl_hrs', (unix_timestamp('planned_departure_utc') - unix_timestamp('prev_actual_arr_utc'))/60/60)\
           .withColumn('poten_for_del', expr("CASE WHEN prev_actual_arr_utc IS NULL THEN '0' WHEN inbtwn_fl_hrs > 2 THEN '0' ELSE '1' END"))
######################### New feature 4: OG aiport avg dep delay 2-4 hrs prior to planned dept   #########################
# if there are serious weather issues or some "global" or "local" issue is occuring, then all flights should be delayed 
# Group by origin airport then sort by planned departure time 
win_ind_airport = Window.partitionBy('origin_airport_code')\
                         .orderBy(f.unix_timestamp('planned_departure_utc'))\
                         .rangeBetween(-14400, -7200)
data = data.withColumn('oa_avg_del2_4hr', f.round(f.avg('delay_minutes').over(win_ind_airport),2))\
            .withColumn('oa_avg_del_ind', expr("CASE WHEN oa_avg_del2_4hr < 15 THEN '0'" + "ELSE '1' END"))
###################### New feature 5: OG aiport avg dep delay by carrier 2-4 hrs prior to planned dept   ###################### 
win_ind_carrier = Window.partitionBy([col('origin_airport_code'), col('carrier')])\
                         .orderBy(f.unix_timestamp('planned_departure_utc'))\
                         .rangeBetween(-14400, -7200)
data = data.withColumn('carrier_avg_del2_4hr', f.round(f.avg('delay_minutes').over(win_ind_carrier),2))\
            .withColumn('carrier_avg_del_ind', expr("CASE WHEN carrier_avg_del2_4hr < 15 THEN '0'" + "ELSE '1' END"))

display(data)

# COMMAND ----------

from pyspark.sql.functions import isnan, when, count, col, lpad, to_utc_timestamp, to_timestamp

df = spark.read.parquet(f"{blob_url}/airlines_eda_v0/*")

# get actual departure and arrival times in UTC
df_w_times = df.alias('d').select('d.*',
                             to_utc_timestamp(to_timestamp(concat(col('dt'), lpad(col('actual_dep_time'), 4, '0')), 'yyyy-MM-ddHHmm'), col('origin_timezone')).alias('actual_dep_utc'),
                             (unix_timestamp(to_utc_timestamp(to_timestamp(concat(col('dt'), lpad(col('actual_dep_time'), 4, '0')), 'yyyy-MM-ddHHmm'), col('origin_timezone')).alias('actual_dep_utc')) + col("actual_duration") * 60).cast('timestamp').alias('actual_arr_utc'))



# COMMAND ----------

data.write.mode('overwrite').parquet(f"{blob_url}/feature_complete_v1")

# COMMAND ----------



# COMMAND ----------

from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml import Pipeline


features_to_normalize = [
  'origin_altitude',
  'origin_wnd_speed',
  'origin_cig_cloud_agl',
  'origin_vis_dist',
  'origin_tmp_c',
  'origin_dew_c',
  'origin_slp_p',
  'dest_altitude',
  'planned_duration',
  'flight_distance',
  'pct_delayed_from_origin', # aggregates
  'mean_delay_from_origin',
  'pct_delayed_to_dest',
  'mean_delay_to_dest',
  'pct_delayed_for_route',
  'mean_delay_for_route',
  'pct_delayed_from_state',
  'mean_delay_from_state',
  'pct_delayed_to_state',
  'mean_delay_to_state'
#   'oa_avg_del2_4hr', # feature engineered
#   'carrier_avg_del2_4hr'
]

# df = spark.read.parquet(f"{blob_url}/airlines_eda_v0/*")
df = spark.read.parquet(f"{blob_url}/feature_complete_v1/*")

# Combine training input columns into a single vector column, "features" is the default column name for sklearn/pyspark feature df
# so we preserve that default name
assembler = VectorAssembler(inputCols=features_to_normalize, outputCol="features").setHandleInvalid("keep")
# Scale features so we can actually use them in logit
# StandardScaler standardizes features by removing the mean and scaling to unit variance.
standardscaler = StandardScaler(withMean=True).setInputCol("features").setOutputCol("scaled_features")


# create pipeline
pipeline = Pipeline(stages=[assembler, standardscaler])


# subset
df = df.select(['index_id'] + features_to_normalize)

# run
pipelineModel = pipeline.fit(df)
df_train = pipelineModel.transform(df)
df_train = df_train.select(['scaled_features']).cache()


def extract(row):
    return tuple(row.scaled_features.toArray().tolist())
test = df_train.rdd.map(extract).toDF(features_to_normalize).cache()
display(test)

# COMMAND ----------

data = spark.read.parquet(f"{blob_url}/feature_complete_v0/*")
display(data)

# COMMAND ----------

features_that_need_null_indicators = [
  'origin_wnd_type', # origin weather
  'origin_wnd_speed',
  'origin_cig_cloud_agl',
  'origin_cig_cavok',
  'origin_vis_dist',
  'origin_vis_var',
  'origin_tmp_c',
  'origin_dew_c',
  'origin_slp_p',
  'oa_avg_del2_4hr', # feature engineered
  'carrier_avg_del2_4hr',
  'da_avg_del2_4hr'
]

data = data.select(["*"] + [col(x).isNull().cast(IntegerType()).alias(f"{x}_null") for x in features_that_need_null_indicators]).cache()

display(data)

# COMMAND ----------

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

display(data)

# COMMAND ----------

features_to_normalize = [
  'origin_altitude', # origin weather
  'origin_wnd_speed',
  'origin_cig_cloud_agl',
  'origin_vis_dist',
  'origin_tmp_c',
  'origin_dew_c',
  'origin_slp_p',
  'dest_altitude',
  'planned_duration',
  'flight_distance',
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
  'oa_avg_del2_4hr', # feature engineered
  'carrier_avg_del2_4hr',
  'da_avg_del2_4hr'
]


features_that_need_null_indicators = [
  'origin_wnd_type', # origin weather
  'origin_wnd_speed',
  'origin_cig_cloud_agl',
  'origin_cig_cavok',
  'origin_vis_dist',
  'origin_vis_var',
  'origin_tmp_c',
  'origin_dew_c',
  'origin_slp_p',
  'oa_avg_del2_4hr', # feature engineered
  'carrier_avg_del2_4hr',
  'da_avg_del2_4hr'
]

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
  'oa_avg_del2_4hr', # feature engineered
  'oa_avg_del_ind',
  'da_avg_del2_4hr',
  'da_avg_del_ind',
  'carrier_avg_del2_4hr',
  'carrier_avg_del_ind',
  'poten_for_del',
  'prev_fl_del',
  'holiday'
] + [f"{x}_null" for x in features_that_need_null_indicators]

# COMMAND ----------

data.write.mode('overwrite').parquet(f"{blob_url}/feature_complete_v1")
data.select(model_features).write.mode('overwrite').parquet(f"{blob_url}/model_features_v1")

# COMMAND ----------


df = spark.read.parquet(f"{blob_url}/feature_complete_v1/*")

features_to_normalize = [
  'origin_altitude', # origin weather
  'origin_wnd_speed',
  'origin_cig_cloud_agl',
  'origin_vis_dist',
  'origin_tmp_c',
  'origin_dew_c',
  'origin_slp_p',
  'dest_altitude',
  'planned_duration',
  'flight_distance',
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
  'oa_avg_del2_4hr', # feature engineered
  'carrier_avg_del2_4hr',
  'da_avg_del2_4hr'
]

batch_1 = [
  'origin_altitude', # origin weather
  'origin_wnd_speed',
  'origin_cig_cloud_agl',
  'origin_vis_dist',
  'origin_tmp_c',
  'origin_dew_c',
  'origin_slp_p',
  'dest_altitude',
  'planned_duration',
  'flight_distance',
  'pct_delayed_from_origin', # aggregates
  'mean_delay_from_origin'
]

batch_2 = [
  'pct_delayed_to_dest',
  'mean_delay_to_dest',
  'pct_delayed_for_route',
  'mean_delay_for_route',
  'pct_delayed_from_state',
  'mean_delay_from_state',
  'pct_delayed_to_state',
  'mean_delay_to_state',
  'oa_avg_del2_4hr', # feature engineered
  'carrier_avg_del2_4hr',
  'da_avg_del2_4hr'
]

# for test in batch_1:
#   df = df.crossJoin(df.select('index_id',
#                  col(test).cast(DoubleType())).na.drop()\
#     .select(mean(test).alias(f"mean_{test}"),
#             stddev(test).alias(f"stdev_{test}")))\
#     .withColumn(f"{test}" , (col(test) - col(f"mean_{test}")) / col(f"stdev_{test}"))\
#     .withColumn(f"{test}", when(col(f"{test}").isNull(), 0).otherwise(col(f"{test}")))


for test in features_to_normalize:
  df = df.withColumn(test, when(col(test).isNull(), 0).otherwise(col(test).cast(DoubleType())))

display(df.cache())
df.write.mode('overwrite').parquet(f"{blob_url}/feature_complete_v2")
df.select(model_features).write.mode('overwrite').parquet(f"{blob_url}/model_features_v2")

# COMMAND ----------

df = spark.read.parquet(f"{blob_url}/full_2015_with_aggs_v2_no_dupes/*")
df.count()

# COMMAND ----------

from pyspark.sql.functions import stddev, mean

df = spark.read.parquet(f"{blob_url}/feature_complete_v1/*")

for test in batch_1:
  df = df.crossJoin(df.select('index_id',
                 col(test).cast(DoubleType())).na.drop()\
    .select(mean(test).alias(f"mean_{test}"),
            stddev(test).alias(f"stdev_{test}")))\
    .withColumn(f"{test}" , (col(test) - col(f"mean_{test}")) / col(f"stdev_{test}"))\
    .withColumn(f"{test}", when(col(f"{test}").isNull(), 0).otherwise(col(f"{test}"))).cache()
  display(df.limit(10))

for test in batch_2:
  df = df.crossJoin(df.select('index_id',
                 col(test).cast(DoubleType())).na.drop()\
    .select(mean(test).alias(f"mean_{test}"),
            stddev(test).alias(f"stdev_{test}")))\
    .withColumn(f"{test}" , (col(test) - col(f"mean_{test}")) / col(f"stdev_{test}"))\
    .withColumn(f"{test}", when(col(f"{test}").isNull(), 0).otherwise(col(f"{test}"))).cache()
  display(df.limit(10))


display(df.cache())
df.write.mode('overwrite').parquet(f"{blob_url}/feature_complete_v3")
df.select(model_features).write.mode('overwrite').parquet(f"{blob_url}/model_features_v3")

# COMMAND ----------

data = spark.read.parquet(f"{blob_url}/full_2015_with_aggs_v2_no_dupes/*")

features_that_need_null_indicators = [
  'origin_wnd_type', # origin weather
  'origin_wnd_speed',
  'origin_cig_cloud_agl',
  'origin_cig_cavok',
  'origin_vis_dist',
  'origin_vis_var',
  'origin_tmp_c',
  'origin_dew_c',
  'origin_slp_p',
#   'oa_avg_del2_4hr', # feature engineered
#   'carrier_avg_del2_4hr',
#   'da_avg_del2_4hr'
]

data = data.select(["*"] + [col(x).isNull().cast(StringType()).alias(f"{x}_null") for x in features_that_need_null_indicators]).cache()

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
                                      "ELSE 'non-holiday' END"))\
        .withColumn('origin_altitude', col('origin_altitude').cast(DoubleType()))\
        .withColumn('dest_altitude', col('dest_altitude').cast(DoubleType()))

display(data)

data.write.mode('overwrite').parquet(f"{blob_url}/feature_complete_v3")
data.select(model_features).write.mode('overwrite').parquet(f"{blob_url}/model_features_v3")

# COMMAND ----------

features_that_need_null_indicators = [
  'origin_wnd_type', # origin weather
  'origin_wnd_speed',
  'origin_cig_cloud_agl',
  'origin_cig_cavok',
  'origin_vis_dist',
  'origin_vis_var',
  'origin_tmp_c',
  'origin_dew_c',
  'origin_slp_p',
#   'oa_avg_del2_4hr', # feature engineered
#   'carrier_avg_del2_4hr',
#   'da_avg_del2_4hr'
]

features_with_numeric_nulls = [
  'origin_wnd_speed', # origin weather
  'origin_cig_cloud_agl',
  'origin_vis_dist',
  'origin_tmp_c',
  'origin_dew_c',
  'origin_slp_p',
#   'oa_avg_del2_4hr', # feature engineered
#   'carrier_avg_del2_4hr',
#   'da_avg_del2_4hr'
]

features_with_str_nulls = [
  'origin_wnd_type', # origin weather
  'origin_cig_cavok',
  'origin_vis_var'
]

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
#   'oa_avg_del2_4hr', # feature engineered
#   'oa_avg_del_ind',
#   'da_avg_del2_4hr',
#   'da_avg_del_ind',
#   'carrier_avg_del2_4hr',
#   'carrier_avg_del_ind',
#   'poten_for_del',
#   'prev_fl_del',
  'holiday'
] + [f"{x}_null" for x in features_that_need_null_indicators]


# COMMAND ----------

for col_name in features_with_numeric_nulls:
  # calculate mean for column after dropping nulls
  col_mean = data.select(col(col_name)).na.drop().select(mean(col(col_name)).alias('mean')).collect()[0]['mean']

  # replace nulls with mean
  data = data.withColumn(col_name, when(col(col_name).isNull(), col_mean).otherwise(col(col_name)))
  
for col_name in features_with_str_nulls:
  # replace nulls with str
  data = data.withColumn(col_name, when(col(col_name).isNull(), 'NULL').otherwise(col(col_name)))

display(data)
data.write.mode('overwrite').parquet(f"{blob_url}/feature_complete_v4")
data.select(model_features).write.mode('overwrite').parquet(f"{blob_url}/model_features_v4")

# COMMAND ----------

from pyspark.sql.functions import isnan, when, count, col
model_data = spark.read.parquet(f"{blob_url}/model_features_v4/*")
display(model_data)
display(model_data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in model_data.columns if c != "planned_departure_utc"]))
print(model_data.count())

# COMMAND ----------

display(data.filter(col('dep_is_delayed').isNull()))

# COMMAND ----------

features = spark.read.parquet(f"{blob_url}/feature_complete_v5/*").cache()
display(features)
print(features.count())

# COMMAND ----------

cols = ['index_id',
        'oa_avg_del2_4hr', # feature engineered
        'oa_avg_del_ind',
        'da_avg_del2_4hr',
        'da_avg_del_ind',
        'carrier_avg_del2_4hr',
        'carrier_avg_del_ind',
        'poten_for_del',
        'prev_fl_del']

final = data.alias('a').join(features.select(cols).alias('f'), col("a.index_id") == col("f.index_id"), 'left')

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
  'oa_avg_del2_4hr', # feature engineered
  'oa_avg_del_ind',
  'da_avg_del2_4hr',
  'da_avg_del_ind',
  'carrier_avg_del2_4hr',
  'carrier_avg_del_ind',
  'poten_for_del',
  'prev_fl_del',
  'holiday'
] + [f"{x}_null" for x in features_that_need_null_indicators]


final.write.mode('overwrite').parquet(f"{blob_url}/feature_complete_v6")
final.select(model_features).dropna(how='any').write.mode('overwrite').parquet(f"{blob_url}/model_features_v6")
