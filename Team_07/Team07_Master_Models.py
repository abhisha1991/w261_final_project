# Databricks notebook source
# MAGIC %md
# MAGIC ## Global Settings

# COMMAND ----------

from pyspark.sql.functions import col, substring, split, when, lit, max as pyspark_max, countDistinct, count, mean, sum as pyspark_sum, expr, to_utc_timestamp, to_timestamp, concat, length
from pyspark.sql import SQLContext, Window 
from pyspark.sql.types import IntegerType, StringType, BooleanType, DateType, DoubleType, TimestampType
import pandas as pd
from gcmap import GCMapper, Gradient
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime
from pyspark.sql import functions as f

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
# MAGIC ## Load Dataset

# COMMAND ----------

# Inspect the Mount's Final Project folder
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project/"))

# COMMAND ----------

# data = spark.read.parquet(f"{blob_url}/joined_eda/*")
# data = spark.read.parquet(f"{blob_url}/full_join_2015_v0/*")
# data = spark.read.parquet(f"{blob_url}/full_join_with_aggs_v0/*")
data = spark.read.parquet(f"{blob_url}/model_features_v1/*")


# COMMAND ----------

n = data.count()
print("The number of rows are {}".format(n))

# COMMAND ----------

# MAGIC %md
# MAGIC ## On-The-Fly Feature Engineering

# COMMAND ----------

display(data)

# COMMAND ----------

data.dtypes

# COMMAND ----------

# null check
from pyspark.sql.functions import isnan, when, count, col
if False:
  display(data.select([(100 * count(when(isnan(c) | col(c).isNull(), c))/n).alias(c) for c in data.columns if c != "planned_departure_utc"]))


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Helper Functions

# COMMAND ----------

from pyspark.sql.functions import percent_rank, to_timestamp
from pyspark.sql import Window
from datetime import datetime, timedelta
from pyspark.sql import functions as F
from pyspark.sql.types import *

# write model to storage
def write_model_to_storage(list_dic, model_class_path, mod_name =''):
  if len(list_dic) == 0:
    raise Exception("Cannot insert empty object into storage")
    
  # add timestamp as key so we can differentiate models of the same type by time
  list_dic_new = []
  now = datetime.now()
  for d in list_dic:
    assert("train" in d.keys())
    assert("test" in d.keys())
    d["timestamp"] = now
    list_dic_new.append(d)
  
  schema = StructType([ \
    StructField("timestamp", TimestampType(), True), \
    StructField("train", StringType(), True), \
    StructField("test", StringType(), True), \
    StructField("val", StringType(), True)])
  
  todf = []
  for d in list_dic_new:
    todf.append((d["timestamp"], d["train"], d["test"], None))
    
  df = spark.createDataFrame(data = todf, schema = schema)
  
  # default model name is based on timestamp - to generate unique name
  if mod_name == '':
    mod_name = str(now).replace(' ', '').replace(':', '').replace('-', '').split('.')[0]
  
  df.write.mode('overwrite').parquet(f"{blob_url}/{model_class_path}/{mod_name}")

def read_model_from_storage(model_path):
  return spark.read.parquet(f"{blob_url}/{model_path}/*")
  
def get_numeric_features(df):
  return [t[0] for t in df.dtypes if t[1] == 'int' or t[1] == 'double']

def get_categorical_features(df):
  return [t[0] for t in df.dtypes if t[1] == 'string']

def get_datetime_features(df):
  return  [t[0] for t in df.dtypes if t[1] == 'timestamp']

def get_boolean_features(df):
  return  [t[0] for t in df.dtypes if t[1] == 'boolean']

def assert_no_other_features_exist(df):
  numeric_features = get_numeric_features(df)
  categorical_features = get_categorical_features(df)
  dt_features = get_datetime_features(df)
  boolean_features = get_boolean_features(df)
  other_features = [t[0] for t in df.dtypes if t[0] not in numeric_features + categorical_features + dt_features + boolean_features]
  assert len(other_features) == 0

def pretty_print_list(elements):
  print("#########################")
  for e in elements:
    print(e)
  print("#########################")
  
def get_feature_dtype(df, colName):
  for t in df.dtypes:
    if t[0] == colName:
      return t[1]
  return None
  
def set_feature_dtype(df, colNames, dtype='string'):
  for colName in colNames:
    currentType = get_feature_dtype(df, colName)
    if currentType == None:      
      raise Exception("Colname is not valid: {}".format(colName))
    
    # preserve existing type
    if currentType == dtype:
      continue
      
    
    # implicit conversion from bool/str to int is not allowed, for some reason - this problem only appears with "dep_is_delayed"
    # we get back nulls for each row if we do a straight conversion to int
    # special case to convert dep_is_delayed to int (needed to be in this form for ML models to work)
    if (currentType == 'string' and colName == "dep_is_delayed") and dtype == 'int':
      
      def convert_to_int(value):
        return 1 if value == "true" else 0
        
      udf_convert = F.udf(convert_to_int, IntegerType())
    
      df = df.withColumn(colName + "_tmp", udf_convert(colName))
      df = df.drop(df.dep_is_delayed)
      df = df.withColumnRenamed(colName + "_tmp", colName)
    
    
    elif dtype == 'string':
      df = df.withColumn(colName, col(colName).cast(StringType()))
    elif dtype == 'int':
      df = df.withColumn(colName, col(colName).cast(IntegerType()))
    elif dtype == 'double':
      df = df.withColumn(colName, col(colName).cast(DoubleType()))
    elif dtype == 'boolean':
      df = df.withColumn(colName, col(colName).cast(BooleanType()))
    elif dtype == 'timestamp':
      df = df.withColumn(colName, to_timestamp(colName))
    else:
      raise Exception("Unsupported data type")
  
  return df

def get_df_for_model(df, splits, index, datatype="train"):
  start_date, end_date = get_dates_from_splits(splits, index, dtype = datatype)
  if verbose:
    print("In method: get_df_for_model - getting back data for data type '{}'. Start date is: {} and End date is: {}".format(datatype, start_date, end_date))
  return get_df(df, start_date, end_date, True)
  
# gets df between 2 given dates
def get_df(df, start_date, end_date, raise_empty=True):
  # assumes that we have access to planned_departure_utc 
  all_columns = [t[0] for t in df.dtypes]
  if "planned_departure_utc" not in all_columns:
    raise Exception("We cannot slice the data by time because we are missing planned_departure_utc")
  
  df = df.filter((col('planned_departure_utc') >= start_date) & (col('planned_departure_utc') <= end_date))
  
  if df.count() == 0 and raise_empty:
    raise Exception("Found 0 records, raising an error as this is not expected")
  
  if verbose:
    print("In method: get_df - getting back data with Start date: {} and End date: {}. Returning {} results".format(start_date, end_date, df.count()))
    
  return df

# contract format depends on function get_timeseries_train_test_splits
def get_dates_from_splits(splits, index, dtype="train"):
  if index >= len(splits):
    raise Exception("Index out of bounds")
    
  split = splits[index]
    
  if dtype == "train":
    # 1st 2 dates are training
    return (split[0], split[1])
  if dtype == "test":
    # next pair is test
    return (split[2], split[3])
  if dtype == "val":
    # last pair is val
    return (split[4], split[5])
  
  # by default return all
  return split

# get rolling or non-rolling time series splits of data
def get_timeseries_train_test_splits(df, rolling=False, roll_months=3, start_year=2015, start_month=1, end_year=2016, end_month=6, train_test_ratio=2, test_months=1):
  if start_year < 2015 or start_year > 2019:
    raise Exception("Invalid date range")
  
  if start_month < 1 or start_month > 12:
    raise Exception("Invalid date range")
  
  if end_month < 1 or end_month > 12:
    raise Exception("Invalid date range")
    
  if start_year > end_year:
    raise Exception("Start year cannot be larger than end year")
  
  if train_test_ratio <= 1 or int(train_test_ratio) != train_test_ratio:
    raise Exception("train_test_ratio must be > 1 and must be int")
  
  assert(test_months >=1 and train_test_ratio > 1 and roll_months >=1)
  
  # assert that we have values for the year and month
  assert(data.filter(data.year.isNull()).count() == 0)
  assert(data.filter(data.month.isNull()).count() == 0)
  
  # format months to 2 numbers - needed for date time parsing
  if start_month <= 9:
    start_month = "0" + str(start_month)
    
  if end_month <= 9:
    end_month = "0" + str(end_month)
  
  
  global_start = "{}-{}-01T00:00:00.000+0000".format(start_year, start_month)
  # why 28? consider february
  global_end = "{}-{}-28T00:00:00.000+0000".format(end_year, end_month)
  
  global_start = datetime.strptime(global_start, '%Y-%m-%dT%H:%M:%S.%f+0000')
  global_end = datetime.strptime(global_end, '%Y-%m-%dT%H:%M:%S.%f+0000')
  
  # check for sufficient data
  # train data is ratio x num months used for testing, hence (test_months * train_test_ratio)
  # validation set and test set have same number of months always, hence (2 * test_months)
  if (global_end - global_start).days < 30 * ((2 * test_months) + (test_months * train_test_ratio)):
    raise Exception("Insufficient data to train on. Please increase date range")
  
  df = df.filter((col('year') >= start_year) & (col('month') >= start_month))
  df = df.filter((col('year') <= end_year) & (col('month') <= end_month))
  
  # create result object - a list of tuple objects
  # tuple object is of the form of dates: (train_start, train_end, test_start, test_end, val_start, val_end)
  result = []
  
  # train is between start (T0) and X days after start, say (T1)
  temp_start_train = global_start
  temp_end_train = global_start + timedelta(days=(test_months * train_test_ratio * 30))

  while (global_end-temp_end_train).days > 0:
    # test is between T1 and Y days after T1, say T2
    temp_start_test = temp_end_train + timedelta(days=1) 
    temp_end_test = temp_start_test + timedelta(days=(test_months * 30))

    # validation is between T2 and Y days after T2, say T3
    temp_start_val = temp_end_test + timedelta(days=1)
    temp_end_val = temp_start_val + timedelta(days=(test_months * 30))

    # add these dates to our result
    result.append((temp_start_train, temp_end_train, temp_start_test, temp_end_test, temp_start_val, temp_end_val))

    # reset new date for ending point for train data and repeat till we reach global end date
    temp_end_train = temp_end_val
    
    # if rolling is enabled, we just roll the train start date by the rolling months
    # and adjust the end train date as well
    if rolling:
      temp_start_train = temp_start_train + timedelta(days=30 * roll_months)
      temp_end_train = temp_start_train + timedelta(days=(test_months * train_test_ratio * 30))
  
  if verbose:
    print("There are {} splits formed based on the date ranges given".format(len(result)))
    print("Date ranges are: start: {} and end: {} with rolling set to {} and rolling window months set to {} months".format(global_start, global_end, rolling, roll_months))
    print("Note that the train_test_ratio is {} and test_months is {}, so training data will have {} month(s) size and test/val data will have {} month(s) size".format(train_test_ratio, test_months, train_test_ratio * test_months, test_months))
    print("Here is a sample split that follows the following format: (train_start, train_end, test_start, test_end, val_start, val_end)")
    print(pretty_print_list(result[0]))
  
  return result

def get_best_param_dic_metrics(best_model, displayKeys=False):
  # https://stackoverflow.com/questions/36697304/how-to-extract-model-hyper-parameters-from-spark-ml-in-pyspark
  parameter_dict = best_model.stages[-1].extractParamMap()
  dic = dict()
  for x, y in parameter_dict.items():
    dic[x.name] = y
    if displayKeys:
      print("Parameter available: {}".format(x.name))

  return dic

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Feature Engineering (DataType Transformation, Data Prep)

# COMMAND ----------

def get_std_features(data):
  # Notes
  # div_reached_dest is going to be full of nulls, so dropping - we should consider making the default as "-1" - so it doesn't make us drop rows (dropna)
  numeric_features = get_numeric_features(data)
  categorical_features = get_categorical_features(data)
  dt_features = get_datetime_features(data)
  bool_features = get_boolean_features(data)
  assert_no_other_features_exist(data)
  cols_to_drop = ['index_id', 'origin_utc_offset', 'dest_utc_offset', 'origin_latitude', 
                  'origin_longitude', 'dest_latitude', 'dest_longitude', 'dt', 'planned_dep_time', 
                  'actual_dep_time', 'actual_arr_time', 'div_reached_dest', 
                  'time_at_prediction_utc', 'oa_avg_del2_4hr', 'da_avg_del2_4hr', 'carrier_avg_del2_4hr'] + [x for x in dt_features if x != 'planned_departure_utc']
  
  # there are some special snowflakes we need to handle here
  # dep_is_delayed, origin_altitude and dest_altitude are strings, they should be numeric
  # so we remove them from the categorical and add them to numeric
  numeric_features = numeric_features + ['origin_altitude', 'dest_altitude', 'dep_is_delayed']
  numeric_features = list(set(numeric_features))
  
  try:
    categorical_features.remove('origin_altitude')
    categorical_features.remove('dest_altitude')
    categorical_features.remove('dep_is_delayed')
  except:
    # dont error if these were not in categorical features
    pass
    
  # likewise, there are some indicator variables that are numeric (int), but need to be string (categorical)
  ind_vars = [x for x in numeric_features if x.endswith("_null")]
  for x in ind_vars:
    try:
      numeric_features.remove(x)
    except:
      # dont error if these were not in numeric
      pass
    
  categorical_features = categorical_features + ind_vars
  categorical_features = list(set(categorical_features))
  
  bool_features = [x for x in bool_features if x not in cols_to_drop]
  dt_features = [x for x in dt_features if x not in cols_to_drop]
  categorical_features = [x for x in categorical_features if x not in cols_to_drop]
  numeric_features = [x for x in numeric_features if x not in cols_to_drop]
  all_cols = numeric_features + categorical_features + dt_features + bool_features
  cols_to_consider = [x for x in all_cols if x not in cols_to_drop] 
  
  if verbose:  
    print("There are {} total columns out of which there are {} columns to consider in the model".format(len(all_cols), len(cols_to_consider)))
    print("There are {} categorical features".format(len(categorical_features)))
    print("There are {} numeric features".format(len(numeric_features)))
    print("There are {} date features".format(len(dt_features)))
    print("There are {} bool features".format(len(bool_features)))
    
  return all_cols, cols_to_consider, cols_to_drop, numeric_features, categorical_features, dt_features, bool_features

def add_required_cols(cols):
  # every model must contain the label and the timestamp var
  if 'planned_departure_utc' not in cols:
    cols.append('planned_departure_utc')
  if 'dep_is_delayed' not in cols:
    cols.append('dep_is_delayed')

  return list(set(cols))
  

def get_std_desired_numeric(df, hypothesis=1, custom_cols_to_drop=[]):
  all_cols, cols_to_consider, cols_to_drop, numeric_features, categorical_features, dt_features, bool_features = get_std_features(data)

  if hypothesis == 1:
    # all numeric features in the df
    desired_numeric = [x for x in numeric_features if x in df.columns]
  elif hypothesis == 2:
    desired_numeric = []
  elif hypothesis == 3:
    desired_numeric = []
  elif hypothesis == 4:
    desired_numeric = []
  else:
    raise Exception("Invalid hypothesis number!")

  # drop any columns that are a no-no in the model
  desired_numeric = [x for x in desired_numeric if x not in cols_to_drop + custom_cols_to_drop]

  # confirm no duplicates
  assert(len(desired_numeric) == len(set(desired_numeric)))

  # confirm data actually has these features
  # also confirm that the desired_numeric is part of the "registered" numeric features to choose from
  all_cols = [t[0] for t in df.dtypes]
  for dn in desired_numeric:
    if dn not in all_cols:
      raise Exception("Unknown feature found: {}".format(dn))
    if dn not in numeric_features:
      raise Exception("Feature: {} is not a registered numeric feature".format(dn))
    
  # ensure that the desired numeric columns are indeed converted to numeric
  # for example, this will ensure that origin_altitude is converted to int
  to_convert = get_std_to_convert_numeric(df, desired_numeric)
  df = set_feature_dtype(df, to_convert, dtype='int')

  return df, list(set(desired_numeric + ['dep_is_delayed']))

def get_std_desired_categorical(df, hypothesis=1, custom_cols_to_drop=[]):
  all_cols, cols_to_consider, cols_to_drop, numeric_features, categorical_features, dt_features, bool_features = get_std_features(data)

  if hypothesis == 1:
    # all categorical features in df
    desired_categorical = [x for x in categorical_features if x in df.columns]
  elif hypothesis == 2:
    desired_categorical = []
  elif hypothesis == 3:
    desired_categorical = []
  elif hypothesis == 4:
    desired_categorical = []
  else:
    raise Exception("Invalid hypothesis number!")

  # drop any columns that are a no-no in the model
  desired_categorical = [x for x in desired_categorical if x not in cols_to_drop + custom_cols_to_drop]

  # confirm no duplicates
  assert(len(desired_categorical) == len(set(desired_categorical)))

  # confirm data actually has these features
  # also confirm that the desired_categorical is part of the "registered" categorical features to choose from
  all_cols = [t[0] for t in df.dtypes]
  for dc in desired_categorical:
    if dc not in all_cols:
      raise Exception("Unknown feature found: {}".format(dc))
    if dc not in categorical_features:
      raise Exception("Feature: {} is not a registered categorical feature".format(dc))
  
  # ensure the vars are converted to strings
  df = set_feature_dtype(df, desired_categorical, dtype='string')
  
  return df, desired_categorical

def get_std_desired_numeric_int(df, desired_numeric):  
  return [x for x in desired_numeric if get_feature_dtype(df, x) == 'int']

def get_std_desired_numeric_double(df, desired_numeric):
  return [x for x in desired_numeric if get_feature_dtype(df, x) == 'double']

def get_std_to_convert_numeric(df, desired_numeric):
  desired_numeric_int = get_std_desired_numeric_int(df, desired_numeric)
  desired_numeric_double =  get_std_desired_numeric_double(df, desired_numeric)

  to_convert_numeric = [x for x in desired_numeric if x not in desired_numeric_int + desired_numeric_double]
  if verbose:
    print("These columns need to be converted to numeric type: {}".format(to_convert_numeric))
    
  return to_convert_numeric

def get_proportion_labels(df):
  if verbose:
    print("In method - get_proportion_labels - displaying proportion of labeled class")
    print(display(df.groupby('dep_is_delayed').count()))
  
  positive = df.filter(df.dep_is_delayed == 1).count()
  negative = df.filter(df.dep_is_delayed == 0).count()
  total = negative + positive
  if total == 0:
    raise Exception("No records found!")
  
  if positive == 0:
    raise Exception("No positive records found!")
  
  if negative == 0:
    raise Exception("No negative records found!")
    
  # there is a risk that the positive/negative classes are so imbalanced that they are non existent in the df
  # so we should guard against that case in order to avoid throwing div by 0
  np = -1 if positive == 0 else 1.0 * negative/positive
  pn = -1 if negative == 0 else 1.0 * positive/negative
  
  return 1.0 * positive/total, 1.0 * negative/total, pn, np

def downsample(df, min_major_class_ratio, alpha=0.99):
  if min_major_class_ratio == -1:
    # assign default value to reduce the majority class by half
    min_major_class_ratio = 0.5
    print("In method downsample: Warning - reset min_major_class_ratio to default: {}".format(min_major_class_ratio))
    
  if verbose:
    print("Starting to downsample, negative class has {} rows and positive class has {} rows".format(df.filter(df.dep_is_delayed == 0).count(), df.filter(df.dep_is_delayed == 1).count()))
    
  negative = df.filter(df.dep_is_delayed == 0).sample(False, min_major_class_ratio * alpha, seed=2021)
  positive = df.filter(df.dep_is_delayed == 1)
  
  new_df = positive.union(negative).cache()
  if verbose:
    negative = new_df.filter(new_df.dep_is_delayed ==0).count()
    positive = new_df.filter(new_df.dep_is_delayed ==1).count()
    print("After downsampling, negative class has {} rows and positive class has {} rows".format(negative, positive))
  
  return new_df

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logit Specific Functions

# COMMAND ----------

from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.types import StringType,BooleanType,DateType
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.feature import IndexToString, StringIndexer, OneHotEncoder, VectorAssembler, Bucketizer, StandardScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

def get_train_test_finalset_for_logit(train, test, custom_payload, drop_na = True, set_handle_invalid="keep"):
  
  if custom_payload == None:
    raise Exception("Custom payload cannot be null as it contains feature selection info")
    
  categorical_features = custom_payload["categorical_features"]
  numeric_features = custom_payload["numeric_features"]
  
  # form a string indexer and change name of dep_is_delayed to "label" - used in std naming conventions in models  
  # https://stackoverflow.com/questions/34681534/spark-ml-stringindexer-handling-unseen-labels
  labelIndexer = StringIndexer(inputCol="dep_is_delayed", outputCol="label").setHandleInvalid(set_handle_invalid).fit(train)
  train = labelIndexer.transform(train)
  test = labelIndexer.transform(test)

  # create index for each categorical feature
  categorical_index = [i + "_Index" for i in categorical_features]
  stringIndexer = StringIndexer(inputCols=categorical_features, outputCols=categorical_index).setHandleInvalid(set_handle_invalid).fit(train)
  train = stringIndexer.transform(train)
  test = stringIndexer.transform(test)

  # create indicator feature for each categorical variable and do one hot encoding, encode only train data
  list_encoders = [i + "_Indicator" for i in categorical_features]
  encoder = OneHotEncoder(inputCols=categorical_index, outputCols=list_encoders).setHandleInvalid(set_handle_invalid).fit(train)
  train_one_hot = encoder.transform(train)
  test_one_hot = encoder.transform(test)

  # retain only encoded categorical columns, numeric features and label 
  train_one_hot = train_one_hot.select(["label"] + categorical_index + list_encoders + numeric_features) 
  test_one_hot = test_one_hot.select(["label"] + categorical_index + list_encoders + numeric_features)

  if verbose:
    print("Training Dataset Count Before Dropping NA: " + str(train_one_hot.count()))
    print("Test Dataset Count Before Dropping NA: " + str(test_one_hot.count()))
    # display(training_set)
  
  if drop_na:  
    training_set = train_one_hot.dropna()
    test_set = test_one_hot.dropna()
  else:
    print("Drop NA is set to false, will not drop any rows...")
    training_set = train_one_hot
    test_set = test_one_hot
    
  # convert label to integer type, so we can compute performance metrics easily
  training_set = training_set.withColumn('label', training_set['label'].cast(IntegerType()))  
  test_set = test_set.withColumn('label', test_set['label'].cast(IntegerType()))

  if verbose and drop_na:
    print("Training Dataset Count After Dropping NA: " + str(training_set.count()))
    print("Test Dataset Count After Dropping NA: " + str(test_set.count()))
    # display(training_set)
  
  return training_set, test_set

def get_logit_pipeline(training_set, set_handle_invalid="keep", grid_search_mode=True):
  
  # get features only
  features_only = training_set.columns
  features_only.remove("label")

  # Combine training input columns into a single vector column, "features" is the default column name for sklearn/pyspark feature df
  # so we preserve that default name
  assembler = VectorAssembler(inputCols=features_only,outputCol="features").setHandleInvalid(set_handle_invalid)

  # Scale features so we can actually use them in logit
  # StandardScaler standardizes features by removing the mean and scaling to unit variance.
  standardscaler = StandardScaler().setInputCol("features").setOutputCol("scaled_features")
  
  # use scaled features in logit, with output column as "label"
  lr = LogisticRegression(featuresCol = 'scaled_features', labelCol = 'label', maxIter=10)

  # for ML Lib pipeline, build a pipeline that will assemble the features into a single vector, perform scaling, and do optionally logit
  if grid_search_mode:
    pipeline = Pipeline(stages=[assembler, standardscaler, lr])
  else:
    pipeline = Pipeline(stages=[assembler, standardscaler])
    
  return lr, pipeline


def model_train_logit_grid_search(training_set, test_set, pipeline, lr, ts_split):
  # grid search is broken - fails with the following error mode
  # https://stackoverflow.com/questions/58827795/requirement-failed-nothing-has-been-added-to-this-summarizer
  # this error mode seems specific to the data it is training on - which is non deterministic based on our train-test size
  # so we don't want to take a dependency on this method
  # moreover - its unclear whether the "numFolds" param should be 1 or > 1 
  # if we make it > 1 then we don't preserve the ordering of the time series data, which is important
  result = {}
  
  # form param grid for searching across multiple params to find best model
  paramGrid = ParamGridBuilder() \
    .addGrid(lr.threshold, [0.01, 0.1, 0.2, 0.3]) \
    .addGrid(lr.maxIter, [2, 5, 10]) \
    .addGrid(lr.regParam, [0.1, 0.2]) \
    .build()
  
  # set up cross validator with the pipeline, choose num cross == 1
  # TODO: clarify on what numFolds should be
  crossval = CrossValidator(estimator = pipeline,
                          estimatorParamMaps = paramGrid,
                          evaluator = BinaryClassificationEvaluator(),
                          numFolds = 1)
  
  # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.BinaryClassificationEvaluator.html#pyspark.ml.evaluation.BinaryClassificationEvaluator.metricName
  # https://stats.stackexchange.com/questions/99916/interpretation-of-the-area-under-the-pr-curve
  evaluator_aupr = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")
  evaluator_auroc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
  
  # fit the model
  cvModel = crossval.fit(training_set)
  
  # return best model from all our models we trained on
  best_model = cvModel.bestModel
  best_param_dic = get_best_param_dic_metrics(best_model, False)

  # review performance on training data 
  train_model = cvModel.transform(training_set)
  aupr = evaluator_aupr.evaluate(train_model)
  auroc = evaluator_auroc.evaluate(train_model)
  true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score = compute_classification_metrics(train_model)
  result["train"] = (true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score, aupr, auroc, ts_split, best_param_dic)
  
  # review performance on test data 
  test_model = cvModel.transform(test_set)
  aupr = evaluator_aupr.evaluate(test_model)
  auroc = evaluator_auroc.evaluate(test_model)
  true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score = compute_classification_metrics(test_model)
  result["test"] = (true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score, aupr, auroc, ts_split, best_param_dic)
  
  return result

def model_train_logit(training_set, test_set, pipeline, lr, ts_split, custom_payload):
  result = {}
  if custom_payload == None:
    raise Exception("Custom payload cannot be none as it contains hyper-param information")
    
  pipelineModel = pipeline.fit(training_set)
  df_train = pipelineModel.transform(training_set)
  df_train = df_train.select(['label', 'scaled_features'])
  
  pipelineModel = pipeline.fit(test_set)
  df_test = pipelineModel.transform(test_set)
  df_test = df_test.select(['label', 'scaled_features'])
  
  # hyper param setting
  lr.threshold = custom_payload["threshold"] if "threshold" in custom_payload.keys() else 0.2
  lr.maxIter = custom_payload["maxIter"] if "maxIter" in custom_payload.keys() else 10
  lr.regParam = custom_payload["regParam"] if "regParam" in custom_payload.keys() else 0.5
  
  print("Starting training of Logit model with parameters - threshold: {}, max iterations: {}, regParam: {}"\
        .format(lr.threshold, lr.maxIter, lr.regParam))
  
  lrModel = lr.fit(df_train)
  
  # set up evaluators
  evaluator_aupr = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")
  evaluator_auroc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
  
  # review performance on training data 
  train_model = lrModel.transform(df_train)
  aupr = evaluator_aupr.evaluate(train_model)
  auroc = evaluator_auroc.evaluate(train_model)
  true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score = compute_classification_metrics(train_model)
  result["train"] = (true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score, aupr, auroc, ts_split, lrModel.summary)
  
  # review performance on test data 
  test_model = lrModel.transform(df_test)
  aupr = evaluator_aupr.evaluate(test_model)
  auroc = evaluator_auroc.evaluate(test_model)
  true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score = compute_classification_metrics(test_model)
  result["test"] = (true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score, aupr, auroc, ts_split, lrModel.summary)
  
  return result

# COMMAND ----------

# MAGIC %md
# MAGIC #### Random Forest Specific Functions

# COMMAND ----------

from pyspark.sql.functions import col, isnan, substring, split, when, lit, max as pyspark_max, countDistinct, count, mean, sum as pyspark_sum, expr, to_utc_timestamp, to_timestamp, concat, length
from pyspark.sql import SQLContext
from pyspark.sql.types import IntegerType, StringType, BooleanType, DateType, DoubleType
import pandas as pd
from gcmap import GCMapper, Gradient
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar
from pyspark.sql.types import *
from pyspark.ml.feature import IndexToString, StringIndexer, OneHotEncoder, VectorAssembler, Bucketizer, StandardScaler, VectorIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
import seaborn as sns
from pyspark.ml.feature import QuantileDiscretizer
  
def get_staged_data_for_rf(df, custom_payload):
  stages = []
  if custom_payload == None:
    raise Exception("Custom payload cannot be none as it contains feature selection information")
    
  # create indexer for label class
  labelIndexer = StringIndexer(inputCol="dep_is_delayed", outputCol="label").setHandleInvalid("keep")
  stages += [labelIndexer]
  
  categorical_features = custom_payload["categorical_features"]
  numeric_features = custom_payload["numeric_features"] 
  num_buckets = custom_payload["num_buckets"] if "num_buckets" in custom_payload.keys() else 3
  
  for cat_feat in categorical_features:
    # string indexing categorical features 
    stringIndexer = StringIndexer(inputCol=cat_feat, outputCol=cat_feat + "_Index").setHandleInvalid("keep")
    # one hot encode categorical features
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[cat_feat + "_One_Hot"])
    # add to stages
    stages += [stringIndexer, encoder]
  
  for num_feat in numeric_features:
    # bin numeric features 
    num_bin = QuantileDiscretizer(numBuckets=num_buckets, inputCol=num_feat, outputCol=num_feat + "_Binned").setHandleInvalid("keep")
    stages += [num_bin]
  
  # create vector assembler combining features into 1 vector
  assemblerInputs = [c + "_One_Hot" for c in categorical_features] + [n + "_Binned" for n in numeric_features]
  assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features").setHandleInvalid("keep")
  stages += [assembler]
  
  # notice no need for scaling in the case of RFs
  partialPipeline = Pipeline().setStages(stages)
  pipelineModel = partialPipeline.fit(df)
  preppedDataDF = pipelineModel.transform(df)
  
  features_comb = categorical_features + numeric_features + ["dep_is_delayed", "planned_departure_utc"]
  selectedcols = ["label", "features"] + features_comb
  dataset = preppedDataDF.select(selectedcols)
  
  return dataset

def model_train_rf(train, test, ts_split, custom_payload):
  if custom_payload == None:
    raise Exception("Custom payload cannot be none as it contains hyper-param information")
    
  result = {}
  # convert label to integer type, so we can find performance metrics easily
  train = train.withColumn('label', train['label'].cast(IntegerType()))  
  test = test.withColumn('label', test['label'].cast(IntegerType()))
  
  # create an initial RandomForest model
  rf = RandomForestClassifier(labelCol="label", featuresCol="features")
  
  # hyper param setting
  rf.maxBins = custom_payload["maxBins"] if "maxBins" in custom_payload.keys() else 20
  rf.numTrees = custom_payload["numTrees"] if "numTrees" in custom_payload.keys() else 10
  rf.minInstancesPerNode = custom_payload["minInstancesPerNode"] if "minInstancesPerNode" in custom_payload.keys() else 5
  rf.minInfoGain = custom_payload["minInfoGain"] if "minInfoGain" in custom_payload.keys() else 0.10
  print("Starting training of random forest model with parameters - max bins: {}, num trees: {}, minInstancesPerNode: {}, minInfoGain: {}"\
        .format(rf.maxBins, rf.numTrees, rf.minInstancesPerNode, rf.minInfoGain))
  
  # train model with training data
  rfModel = rf.fit(train)

  # make predictions on test data 
  rf_predictions = rfModel.transform(test)
  
  # set up evaluators
  evaluator_aupr = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")
  evaluator_auroc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
  
  # review performance on training data 
  train_model = rfModel.transform(train)
  aupr = evaluator_aupr.evaluate(train_model)
  auroc = evaluator_auroc.evaluate(train_model)
  true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score = compute_classification_metrics(train_model)
  result["train"] = (true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score, aupr, auroc, ts_split, rfModel.summary)
  
  # review performance on test data 
  test_model = rfModel.transform(test)
  aupr = evaluator_aupr.evaluate(test_model)
  auroc = evaluator_auroc.evaluate(test_model)
  true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score = compute_classification_metrics(test_model)
  result["test"] = (true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score, aupr, auroc, ts_split, rfModel.summary)
  
  return result

# COMMAND ----------

# MAGIC %md
# MAGIC #### Gradient Boosted Trees Specific Functions

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
  
def get_staged_data_for_gbt(df, custom_payload):
  stages = []
  if custom_payload == None:
    raise Exception("Custom payload cannot be none as it contains feature selection information")
    
  # create indexer for label class
  labelIndexer = StringIndexer(inputCol="dep_is_delayed", outputCol="label").setHandleInvalid("keep")
  stages += [labelIndexer]
  
  categorical_features = custom_payload["categorical_features"]
  numeric_features = custom_payload["numeric_features"] 
  num_buckets = custom_payload["num_buckets"] if "num_buckets" in custom_payload.keys() else 3
  
  for cat_feat in categorical_features:
    # string indexing categorical features 
    stringIndexer = StringIndexer(inputCol=cat_feat, outputCol=cat_feat + "_Index").setHandleInvalid("keep")
    # one hot encode categorical features
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[cat_feat + "_One_Hot"])
    # add to stages
    stages += [stringIndexer, encoder]
  
  for num_feat in numeric_features:
    # bin numeric features 
    num_bin = QuantileDiscretizer(numBuckets=num_buckets, inputCol=num_feat, outputCol=num_feat + "_Binned").setHandleInvalid("keep")
    stages += [num_bin]
  
  # create vector assembler combining features into 1 vector
  assemblerInputs = [c + "_One_Hot" for c in categorical_features] + [n + "_Binned" for n in numeric_features]
  assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features").setHandleInvalid("keep")
  stages += [assembler]
  
  # notice no need for scaling in the case of RFs
  partialPipeline = Pipeline().setStages(stages)
  pipelineModel = partialPipeline.fit(df)
  preppedDataDF = pipelineModel.transform(df)
  
  features_comb = categorical_features + numeric_features + ["dep_is_delayed", "planned_departure_utc"]
  selectedcols = ["label", "features"] + features_comb
  dataset = preppedDataDF.select(selectedcols)
  
  return dataset

def model_train_gbt(train, test, ts_split, custom_payload):
  if custom_payload == None:
    raise Exception("Custom payload cannot be none as it contains hyper-param information")
    
  result = {}
  # convert label to integer type, so we can find performance metrics easily
  train = train.withColumn('label', train['label'].cast(IntegerType()))  
  test = test.withColumn('label', test['label'].cast(IntegerType()))
  
  # create an initial GBT model
  gbt = GBTClassifier(labelCol="label", featuresCol="features")
  
  # hyper param setting
  gbt.maxBins = custom_payload["maxBins"] if "maxBins" in custom_payload.keys() else 32
  gbt.maxDepth = custom_payload["maxDepth"] if "maxDepth" in custom_payload.keys() else 10
  gbt.minInstancesPerNode = custom_payload["minInstancesPerNode"] if "minInstancesPerNode" in custom_payload.keys() else 10
  gbt.minInfoGain = custom_payload["minInfoGain"] if "minInfoGain" in custom_payload.keys() else 0.10
  gbt.maxIter = custom_payload["maxIter"] if "maxIter" in custom_payload.keys() else 10
  gbt.stepSize = custom_payload["stepSize"] if "stepSize" in custom_payload.keys() else 0.2
  print("Starting training of GBT model with parameters - max bins: {}, max depth: {}, minInstancesPerNode: {}, minInfoGain: {}, max iterations: {}, step size: {}"\
        .format(gbt.maxBins, gbt.maxDepth, gbt.minInstancesPerNode, gbt.minInfoGain, gbt.maxIter, gbt.stepSize))
  
  # train model with training data
  gbtModel = gbt.fit(train)

  # make predictions on test data
  gbt_predictions = gbtModel.transform(test)
  
  # set up evaluators
  evaluator_aupr = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")
  evaluator_auroc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
  
  # review performance on training data 
  train_model = gbtModel.transform(train)
  aupr = evaluator_aupr.evaluate(train_model)
  auroc = evaluator_auroc.evaluate(train_model)
  true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score = compute_classification_metrics(train_model)
  result["train"] = (true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score, aupr, auroc, ts_split, gbtModel) # no summary object exists in GBT
  
  # review performance on test data 
  test_model = gbtModel.transform(test)
  aupr = evaluator_aupr.evaluate(test_model)
  auroc = evaluator_auroc.evaluate(test_model)
  true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score = compute_classification_metrics(test_model)
  result["test"] = (true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score, aupr, auroc, ts_split, gbtModel) # no summary object exists in GBT
  
  return result

# COMMAND ----------

# MAGIC %md
# MAGIC #### SVM Specific Functions

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Model Training and Evaluation - Apply Pipeline To Data & Train & Collect Metrics

# COMMAND ----------

from statistics import *
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

def compute_classification_metrics(df):
  # assumes df has columns called label and prediction
  true_positive = df[(df.label == 1) & (df.prediction == 1)].count()
  true_negative = df[(df.label == 0) & (df.prediction == 0)].count()
  false_positive = df[(df.label == 0) & (df.prediction == 1)].count()
  false_negative = df[(df.label == 1) & (df.prediction == 0)].count()
  accuracy = ((true_positive + true_negative)/df.count())
  
  if (true_positive + false_negative == 0.0):
    recall = 0.0
    precision = float(true_positive) / (true_positive + false_positive)
    
  elif (true_positive + false_positive == 0.0):
    recall = float(true_positive) / (true_positive + false_negative)
    precision = 0.0
    
  else:
    recall = float(true_positive) / (true_positive + false_negative)
    precision = float(true_positive) / (true_positive + false_positive)

  if(precision + recall == 0):
    f1_score = 0
    
  else:
    f1_score = 2 * ((precision * recall)/(precision + recall))
    
  return true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score

def get_classification_metrics(dic, with_display=True, display_train_metrics=False):
  '''
  assumes every model follows a contract of having a result dictionary
  with key = "train" and key = "test", and optionally, key = "val"
  
  also assumes that the dictionary payload follows the format
  result["key"] = (true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score, aupr, auroc, ts_split, modelsummary)
  '''
  if "train" not in dic.keys() or "test" not in dic.keys():
    raise Exception("Result object does not have the right keys")
  
  contains_val = "val" in dic.keys()
  result = {"train": dict(), "test": dict()}
  if contains_val:
    result["val"] = dict()
  
  if not with_display:
    display_train_metrics = False
  
  true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score, aupr, auroc, ts_split, modelsummary = dic["train"]
  # TODO: format this to human readable form
  ts_split_str = str(ts_split)
  
  tmp = {"true_positive": true_positive, "true_negative": true_negative, "false_positive": false_positive, "false_negative": false_negative,
         "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score, "aupr": aupr, "auroc": auroc, "ts_split": ts_split_str, "summary": modelsummary}
  
  if with_display:
    # enter new line for neatness
    print()
    
  # set the temp dictionary to result
  result["train"] = tmp
  
  str_ts = "Metrics for Split - (Train: {}-{}), (Test: {}-{}), (Val: {}-{})".format(ts_split[0].strftime("%b %d %Y"), \
  ts_split[1].strftime("%b %d %Y"), ts_split[2].strftime("%b %d %Y"), ts_split[3].strftime("%b %d %Y"), ts_split[4].strftime("%b %d %Y"), ts_split[5].strftime("%b %d %Y"))
  
  num = 150
  
  if with_display and display_train_metrics:
    print("#" * num)
    print("Training Data " + str_ts)
    print("Accuracy: {}".format(result["train"]["accuracy"]))
    print("Precision: {}".format(result["train"]["precision"]))
    print("Recall: {}".format(result["train"]["recall"]))
    print("F1 Score: {}".format(result["train"]["f1_score"]))
    print("Area under PR curve: {}".format(result["train"]["aupr"]))
    print("Area under ROC curve: {}".format(result["train"]["auroc"]))
    print("#" * num)
  
  # do the same for test and val
  true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score, aupr, auroc, ts_split, modelsummary = dic["test"]
  # TODO: format this to human readable form
  ts_split_str = str(ts_split)
  
  tmp = {"true_positive": true_positive, "true_negative": true_negative, "false_positive": false_positive, "false_negative": false_negative,
         "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score, "aupr": aupr, "auroc": auroc, "ts_split": ts_split_str, "summary": modelsummary}
  
  # set the temp dictionary to result
  result["test"] = tmp
  
  if with_display:
    print("#" * num)
    print("Test Data " + str_ts)
    print("Accuracy: {}".format(result["test"]["accuracy"]))
    print("Precision: {}".format(result["test"]["precision"]))
    print("Recall: {}".format(result["test"]["recall"]))
    print("F1 Score: {}".format(result["test"]["f1_score"]))
    print("Area under PR curve: {}".format(result["test"]["aupr"]))
    print("Area under ROC curve: {}".format(result["test"]["auroc"]))
    print("#" * num)
    
  if contains_val:
    true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1_score, aupr, auroc, ts_split, modelsummary = dic["val"]
    # TODO: format this to human readable form
    ts_split_str = str(ts_split)
    
    tmp = {"true_positive": true_positive, "true_negative": true_negative, "false_positive": false_positive, "false_negative": false_negative,
         "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score, "aupr": aupr, "auroc": auroc, "ts_split": ts_split_str, "summary": modelsummary}
    
    # set the temp dictionary to result
    result["val"] = tmp
  
    if with_display:
      print("#" * num)
      print("Validation Data " + str_ts)
      print("Accuracy: {}".format(result["val"]["accuracy"]))
      print("Precision: {}".format(result["val"]["precision"]))
      print("Recall: {}".format(result["val"]["recall"]))
      print("F1 Score: {}".format(result["val"]["f1_score"]))
      print("Area under PR curve: {}".format(result["val"]["aupr"]))
      print("Area under ROC curve: {}".format(result["val"]["auroc"]))
      print("#" * num)
  
  return result


def get_classification_metrics_for_storage_ingestion(list_dic):
  # sadly, we cannot store model summary into the dataframe, thus we return everything except that
  # assumes we have output from get_classification_metrics() (in list form) as the input here
  for dic in list_dic:
    dic["train"].pop("summary", None)
    dic["test"].pop("summary", None)
    if "val" in dic.keys():
      dic["val"].pop("summary", None)
  
  return list_dic

def get_aggregated_classification_metrcs(list_dic, dtype="test", with_display=True):
  '''
  gets summary stats (avg, min, percentiles etc.) for the list of models 
  has a dependency on the key naming defined in get_classification_metrics()
  '''
  metric_type = ["accuracy", "precision", "recall", "f1_score", "aupr", "auroc"]
  summary_type = ["mean", "min", "max", "median"]
  
  # for some reason math.min, math.max don't work and throw a "type" error
  # same goes for statistics.mean and statistics.median
  # this shows up sometimes in the RF models get_aggregated_classification_metrcs() calc
  # so we go old school for now :)
  def get_max(list_nums):
    maxi = -1
    for n in list_nums:
      if n > maxi:
        maxi = n
    return maxi
  
  def get_min(list_nums):
    mini = 1000
    for n in list_nums:
      if n < mini:
        mini = n
    return mini
  
  def get_mean(list_nums):
    meanval = 0
    for n in list_nums:
      meanval += n
    return meanval/len(list_nums)
  
  def get_median(list_nums):
    list_nums.sort()
    n = len(list_nums)
    if n % 2 == 0:
      median1 = list_nums[n//2]
      median2 = list_nums[n//2 - 1]
      median = (median1 + median2)/2
    else:
      median = list_nums[n//2]
      
    return median
    
  todf = []
  for s in summary_type:
    metrics = []
    for m in metric_type:
      if s == "mean":
        # print("For summary type: {} and metric type: {}, value is {}".format(s, m, get_mean([dic[dtype][m] for dic in list_dic])))
        metrics.append(get_mean([dic[dtype][m] for dic in list_dic]))
      elif s == "min":
        # print("For summary type: {} and metric type: {}, value is {}".format(s, m, get_min([dic[dtype][m] for dic in list_dic])))
        metrics.append(get_min([dic[dtype][m] for dic in list_dic]))
      elif s == "max":        
        # print("For summary type: {} and metric type: {}, value is {}".format(s, m, get_max([dic[dtype][m] for dic in list_dic])))
        metrics.append(get_max([dic[dtype][m] for dic in list_dic]))
      elif s == "median":
        # print("For summary type: {} and metric type: {}, value is {}".format(s, m, get_median([dic[dtype][m] for dic in list_dic])))
        metrics.append(get_median([dic[dtype][m] for dic in list_dic]))
        
    todf.append(tuple(metrics))   
  
  schema = StructType([ \
    StructField("accuracy", DoubleType(), True), \
    StructField("precision", DoubleType(), True), \
    StructField("recall", DoubleType(), True), \
    StructField("f1_score", DoubleType(), True), \
    StructField("AUPR", DoubleType(), True), \
    StructField("AUROC", DoubleType(), True) \
  ])
  
  df = spark.createDataFrame(data = todf, schema = schema)
  if with_display:
    print("Displaying aggregated metrics - rows are in order: {}".format(summary_type))
    display(df)
    
  return df
    

# COMMAND ----------

# for each train, test split - apply pipeline and perform model training
##### THIS IS THE MAIN METHOD FOR TRAINING AND PLUGGING IN ALL OTHER MODELS #####

def model_train_and_eval(data, splits, max_iter=1, model="logit", collect_metrics = True, rebalance_downsample=True, custom_payload=None):
  '''
  Main method for running models and returning results
  custom_payload is referring to a dictionary that each model can unpack to access model specific values (reg params, special columns, feature slection etc.) 
  '''
  
  # list that holds the metrics results for the specified model
  # the metrics returned per model may be different, 
  # look into each model's specific return format to extract relevant metric
  collect_metrics_result = []
  
  for i in range(len(splits)):
    if i > max_iter-1:
      break
    
    train = get_df_for_model(data, splits, index=i, datatype="train")
    test = get_df_for_model(data, splits, index=i, datatype="test")
    #val = get_df_for_model(data, splits, index=i, datatype="val")
    
    # drop planned_departure_utc and index id before sending off to the model
    # we kept planned_departure_utc up till now as that's needed for data time filtering
    # we kept index_id because its the index and may help spark in retrieving rows quicker
    cols = [t[0] for t in train.dtypes if t[0] != 'planned_departure_utc' or t[0] != 'index_id']
    train = train.select(cols).cache()
    test = test.select(cols).cache()
    
    # need to pass down split dates info to the models as they need this for result object
    split = get_dates_from_splits(splits, index=i, dtype="all")
    
    # finally, downsample the majority class (dep_is_delayed == false) if need be
    # we downsample because we have tons of data fortunately - otherwise, we would have up sampled
    if rebalance_downsample:
      print("Down-sampling the training data to have more balanced classes...")
      pt, nt, pn, np = get_proportion_labels(train)
      train = downsample(train, pn)
      
    print("Starting training iteration: {} for model: '{}' with collect_metrics: {}".format(i+1, model, collect_metrics))
    
    if model == "logit":
      training_set, test_set = get_train_test_finalset_for_logit(train, test, custom_payload)
      lr, pipeline = get_logit_pipeline(training_set, grid_search_mode=False)
      result = model_train_logit(training_set, test_set, pipeline, lr, split, custom_payload)
    
    # do not use gs version of logit - there is a bug - see func defn
    elif model == "logit_gs":
      training_set, test_set = get_train_test_finalset_for_logit(train, test, custom_payload)
      lr, pipeline = get_logit_pipeline(training_set, grid_search_mode=True)
      result = model_train_logit_grid_search(training_set, test_set, pipeline, lr, split)
    
    elif model == "rf":
      result = model_train_rf(train, test, split, custom_payload)
    
    elif model == "gbt":
      result = model_train_gbt(train, test, split, custom_payload)
      
    else:
      raise Exception("Model name not found - given name is {}".format(model))
      
    if collect_metrics:
      collect_metrics_result.append(result)
    
  return collect_metrics_result
      

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Driver Program - Logistic Regression

# COMMAND ----------

###### WARNING: DO NOT MODIFY THE "data" OBJECT ######
###### IT IS SHARED AMONGST OTHER DRIVER PROGRAMS ######

# verbose logging / debug mode
# can be changed per driver program
verbose = True

if verbose:
  print("Total number of rows in original dataset are {}".format(n))

# all_cols, cols_to_consider, cols_to_drop, numeric_features, categorical_features, dt_features, bool_features = get_std_features(data)  


def get_values_from_hypothesis(hypothesis=1, custom_cols_to_drop=[]):
  ##### HYPOTHESIS ######
  data_, desired_numeric_h = get_std_desired_numeric(data, hypothesis= hypothesis, custom_cols_to_drop= custom_cols_to_drop)
  data_, desired_categorical_h = get_std_desired_categorical(data_, hypothesis= hypothesis, custom_cols_to_drop= custom_cols_to_drop)

  # assert label is numeric - this is because its needed for the classification metrics
  assert(get_feature_dtype(data_, 'dep_is_delayed') == 'int')
  cols_to_consider_h = list(set(desired_numeric_h + desired_categorical_h)) 
  
  # we added dep_is_delayed to desired_numeric_h as we wanted to convert it to numeric
  # however, it should not be part of the features list as it is the output var
  # we will later add this col to cols_to_consider so its still part of our dataset
  try:
    desired_numeric_h.remove('dep_is_delayed')
  except:
    pass
  
  # ensure label and planned_departure_utc are present in cols_to_consider
  cols_to_consider_h = add_required_cols(cols_to_consider_h)
  # +2 in assert comes from adding planned_departure_utc and label (dep_is_delayed)
  assert(len(cols_to_consider_h) == len(desired_numeric_h) + len(desired_categorical_h) + 2)
  
  # create custom payload object
  custom_payload = {"categorical_features": desired_categorical_h, "numeric_features": desired_numeric_h}
  
  return desired_categorical_h, desired_numeric_h, cols_to_consider_h, data_.select(cols_to_consider_h), custom_payload

desired_categorical_logit, desired_numeric_logit, cols_to_consider_logit, data_logit, custom_payload_logit = get_values_from_hypothesis(1)


#### COMMON ####  
if verbose:
  print("Finally, there are {} categorical features and {} numeric features".format(len(desired_categorical_logit), len(desired_numeric_logit)))
  print("data_logit has {} rows".format(data_logit.count()))
  display(data_logit)

# get the data split for time series
splits = get_timeseries_train_test_splits(data_logit, train_test_ratio=3, test_months=2, start_year=2015, end_year=2019)
# splits = get_timeseries_train_test_splits(data_logit)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Display Result and Write Models to Storage

# COMMAND ----------

# perform actual training with logit model, get back list of dictionaries (each dic has train, test, val keys)
logit_results = model_train_and_eval(data_logit, splits, max_iter=3, model = "logit", collect_metrics = True, custom_payload = custom_payload_logit)

storage_logit_results = []
for lrdic in logit_results:
  # get back well formed metrics dictionary for each time-iteration of the model
  metrics = get_classification_metrics(lrdic, with_display=True, display_train_metrics=True)
  storage_logit_results.append(metrics)
  
print("Displaying Aggregated Results for Logistic Regression")
get_aggregated_classification_metrcs(storage_logit_results, dtype="test", with_display=True)
  

# COMMAND ----------

print("Writing results to storage")
# get back formatted dictionary list that is compatible to write to storage
storage_logit_results = get_classification_metrics_for_storage_ingestion(storage_logit_results)
write_model_to_storage(storage_logit_results, "logit_v1_test")

display(read_model_from_storage("logit_v1_test"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Driver Program - Random Forests

# COMMAND ----------

###### WARNING: DO NOT MODIFY THE "data" OBJECT ######
###### IT IS SHARED AMONGST OTHER DRIVER PROGRAMS ######

# verbose logging / debug mode
# can be changed per driver program
verbose = True

if verbose:
  print("Total number of rows in original dataset are {}".format(n))

desired_categorical_rf, desired_numeric_rf, cols_to_consider_rf, data_rf, custom_payload_rf = get_values_from_hypothesis(1)

#### COMMON ####  
if verbose:
  print("Finally, there are {} categorical features and {} numeric features".format(len(desired_categorical_rf), len(desired_numeric_rf)))
  print("data_rf has {} rows".format(data_rf.count()))
  display(data_rf)

# get the data split for time series
splits = get_timeseries_train_test_splits(data_rf, train_test_ratio=3, test_months=2, start_year=2015, end_year=2019)
# splits = get_timeseries_train_test_splits(data_rf)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Display Result and Write Models to Storage

# COMMAND ----------

# data needs to be staged before it can be partitioned and trained on
# pass the custom_payload here itself, as this piece does feature selection for us
data_rf = get_staged_data_for_rf(data_rf, custom_payload_rf)

# perform actual training with RF model
rf_results = model_train_and_eval(data_rf, splits, max_iter=3, model = "rf", collect_metrics = True, custom_payload = custom_payload_rf)

storage_rf_results = []
for rfdic in rf_results:
  # get back well formed metrics dictionary for each time-iteration of the model
  metrics = get_classification_metrics(rfdic, with_display=True, display_train_metrics=True)
  storage_rf_results.append(metrics)  

print("Displaying Aggregated Results for Random Forests")
get_aggregated_classification_metrcs(storage_rf_results, dtype="test", with_display=True)

# COMMAND ----------

print("Writing results to storage")
# get back formatted dictionary list that is compatible to write to storage
storage_rf_results = get_classification_metrics_for_storage_ingestion(storage_rf_results)
write_model_to_storage(storage_rf_results, "rf_v1_test")

display(read_model_from_storage("rf_v1_test"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Driver Program - GBT

# COMMAND ----------

###### WARNING: DO NOT MODIFY THE "data" OBJECT ######
###### IT IS SHARED AMONGST OTHER DRIVER PROGRAMS ######

# verbose logging / debug mode
# can be changed per driver program
verbose = True

if verbose:
  print("Total number of rows in original dataset are {}".format(n))

desired_categorical_gbt, desired_numeric_gbt, cols_to_consider_gbt, data_gbt, custom_payload_gbt = get_values_from_hypothesis(1)
    
#### COMMON ####  
if verbose:
  print("Finally, there are {} categorical features and {} numeric features".format(len(desired_categorical_gbt), len(desired_numeric_gbt)))
  print("data_gbt has {} rows".format(data_gbt.count()))
  display(data_gbt)

# get the data split for time series
splits = get_timeseries_train_test_splits(data_gbt, train_test_ratio=3, test_months=2, start_year=2015, end_year=2019)
# splits = get_timeseries_train_test_splits(data_gbt)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Display Result and Write Models to Storage

# COMMAND ----------

# data needs to be staged before it can be partitioned and trained on
# pass the custom_payload here itself, as this piece does feature selection for us
data_gbt = get_staged_data_for_gbt(data_gbt, custom_payload_gbt)

# perform actual training with GBT model
gbt_results = model_train_and_eval(data_gbt, splits, max_iter=3, model = "gbt", collect_metrics = True, custom_payload = custom_payload_gbt)

storage_gbt_results = []
for gbtdic in gbt_results:
  # get back well formed metrics dictionary for each time-iteration of the model
  metrics = get_classification_metrics(gbtdic, with_display=True, display_train_metrics=True)
  storage_gbt_results.append(metrics)  

print("Displaying Aggregated Results for GBT")
get_aggregated_classification_metrcs(storage_gbt_results, dtype="test", with_display=True)

# COMMAND ----------

print("Writing results to storage")
# get back formatted dictionary list that is compatible to write to storage
storage_gbt_results = get_classification_metrics_for_storage_ingestion(storage_gbt_results)
write_model_to_storage(storage_gbt_results, "gbt_v1_test")

display(read_model_from_storage("gbt_v1_test"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Driver Program - SVM

# COMMAND ----------

###### WARNING: DO NOT MODIFY THE "data" OBJECT ######
###### IT IS SHARED AMONGST OTHER DRIVER PROGRAMS ######

# verbose logging / debug mode
# can be changed per driver program
verbose = True

if verbose:
  print("Total number of rows in original data set are {}".format(data.count()))

desired_categorical_svm, desired_numeric_svm, cols_to_consider_svm, data_svm, custom_payload_svm = get_values_from_hypothesis(1)

#### COMMON ####  
if verbose:
  print("Finally, there are {} categorical features and {} numeric features".format(len(desired_categorical_svm), len(desired_numeric_svm)))
  print("data_gbt has {} rows".format(data_svm.count()))
  display(data_svm)

# get the data split for time series
splits = get_timeseries_train_test_splits(data_svm, train_test_ratio=3, test_months=2, start_year=2015, end_year=2019)
# splits = get_timeseries_train_test_splits(data_svm)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Display Result and Write Models to Storage

# COMMAND ----------

# data needs to be staged before it can be partitioned and trained on
# pass the custom_payload here itself, as this piece does feature selection for us
data_svm = get_staged_data_for_svm(data_svm, custom_payload_svm)

# perform actual training with SVM model
svm_results = model_train_and_eval(data_svm, splits, max_iter=2, model = "svm", collect_metrics = True, custom_payload = custom_payload_svm)

storage_svm_results = []
for svmdic in svm_results:
  # get back well formed metrics dictionary for each time-iteration of the model
  metrics = get_classification_metrics(svmdic, with_display=True, display_train_metrics=True)
  storage_svm_results.append(metrics)  

print("Displaying Aggregated Results for SVM")
get_aggregated_classification_metrcs(storage_svm_results, dtype="test", with_display=True)

# COMMAND ----------

print("Writing results to storage")
# get back formatted dictionary list that is compatible to write to storage
storage_svm_results = get_classification_metrics_for_storage_ingestion(storage_svm_results)
write_model_to_storage(storage_svm_results, "svm_v1")

display(read_model_from_storage("svm_v1"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resources and Links

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC - https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/mllib-mlflow-integration.html
# MAGIC - https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.ParamGridBuilder.html
# MAGIC - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
# MAGIC - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# MAGIC - https://spark.apache.org/docs/latest/ml-classification-regression.html#binomial-logistic-regression
# MAGIC - https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa
# MAGIC - https://medium.com/swlh/logistic-regression-with-pyspark-60295d41221
# MAGIC - https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4
# MAGIC - https://medium.com/@haoyunlai/smote-implementation-in-pyspark-76ec4ffa2f1d
# MAGIC - https://docs.databricks.com/applications/machine-learning/train-model/mllib/index.html
# MAGIC - https://github.com/MingChen0919/learning-apache-spark/blob/master/notebooks/06-machine-learning/classification/random-forest-classification.ipynb
# MAGIC - https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier

# COMMAND ----------

display(data)

# COMMAND ----------

data.dtypes

# COMMAND ----------


