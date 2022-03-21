# Databricks notebook source
# MAGIC %md
# MAGIC # PLEASE CLONE THIS NOTEBOOK INTO YOUR PERSONAL FOLDER
# MAGIC # DO NOT RUN CODE IN THE SHARED FOLDER

# COMMAND ----------

from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

# COMMAND ----------

#setting blob container variables
blob_container = "newteam9" # The name of your container created in https://portal.azure.com
storage_account = "newteam9" # The name of your Storage account created in https://portal.azure.com
secret_scope = "te9" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "a" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

# Inspect the Mount's Final Project folder 
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project"))

# COMMAND ----------

# Load 2015 Q1 & Q2 for Flights
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_6m/")
display(df_airlines)

# COMMAND ----------

# Load the 2015 Q1 for Weather
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-07-01T00:00:00.000")
display(df_weather)

# COMMAND ----------

df_weather.columns

# COMMAND ----------

#read in stations dataset from parquet file
df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")
display(df_stations)

# COMMAND ----------

#group stations table by station_id and get the 'max' latitude and longitude. There's multiple records for each station_id (one for each 'neighbor station') so record counts get crazy if we don't do this
df_stations_lkp = df_stations.groupBy('station_id').max('lat','lon').withColumnRenamed('max(lat)','lat').withColumnRenamed('max(lon)','lon')
df_stations_lkp.display()

#upload to blob storage
df_stations_lkp.write.mode('overwrite').parquet(f"{blob_url}/df_stations_lkp")

# COMMAND ----------

#create random sample of tables for testing, change later once testing is done
subset_w = df_weather.sample(.3,42)
subset_a = df_airlines.sample(.3,55)
, START 
#write subset to blob storage
subset_w.write.mode('overwrite').parquet(f"{blob_url}/subset_w")
subset_a.write.mode('overwrite').parquet(f"{blob_url}/subset_a2")

# COMMAND ----------

# DBTITLE 1,Joins: Start here so you don't have to rerun all above.
#read from blob
subset_a = spark.read.parquet(f"{blob_url}/subset_a")
subset_w = spark.read.parquet(f"{blob_url}/subset_w")
df_stations = spark.read.parquet(f"{blob_url}/df_stations_lkp")

# COMMAND ----------

#add hour column to weather table (round down to nearest hour) used for joining to airline table
#2015-01-04T15:00:00.000+0000
from pyspark.sql.functions import *
import pyspark.sql.functions as F
subset_w = subset_w.withColumn("obs_hour", from_unixtime(unix_timestamp(col("DATE"),"yyyy-MM-dd hh:mm:ss"),"hh"))
subset_w = subset_w.withColumn("plain_date", from_unixtime(unix_timestamp(col("DATE"),"yyyy-MM-dd hh:mm:ss"),"yyyy-MM-dd"))

# COMMAND ----------

#don't rerun (or if you do rerun the command to read in from blob)
#Remove "K" prefix for joining
subset_w= subset_w.withColumn("plain_call_sign", when(subset_w.CALL_SIGN.substr(1,1)=='K',subset_w.CALL_SIGN.substr(2,20)).otherwise(subset_w.CALL_SIGN))

subset_w.select('plain_call_sign').distinct().collect()

# COMMAND ----------

### Remove leading and trailing space of plain_call_sign column in weather data
subset_w = subset_w.withColumn('plain_call_sign', trim(subset_w.plain_call_sign))

subset_w.select('plain_call_sign').distinct().collect()

# COMMAND ----------

#subset of weather data count
subset_w.count()

# COMMAND ----------

#Dont rerun unless you change subset_w
subset_w.display(10)

# COMMAND ----------

#subset of weather data count ater filtering for one report type
subset_w.count()

# COMMAND ----------

#listing out column names and types
subset_w.columns

# COMMAND ----------

#get hour (round down) from CRS_DEP_TIME in new column airline data

#first need to convert timestamp format
subset_a = subset_a.withColumn("time_string", F.format_string("%04d", F.col("CRS_DEP_TIME")))\
    .withColumn(
        "time_string",
        F.concat_ws(
            ":",
            F.array(
                [
                    F.substring(
                        "time_string",
                        1,
                        2
                    ),
                    F.substring(
                        "time_string",
                        3,
                        2
                    ),
                    F.lit("00")
                ]
            )
        )
    )\
    .withColumn("time_string", F.concat(F.lit("2018-01-01 "), F.col("time_string")))

#now can parse out the hour
subset_a = subset_a.withColumn("dept_hour", from_unixtime(unix_timestamp(col("time_string"),"yyyy-MM-dd HH:mm:ss"),"hh"))

# COMMAND ----------

#Dont rerun unless you change subset_a
subset_a.display(10)

# COMMAND ----------

#checking distinct values for joining column
subset_a.select('ORIGIN').distinct().collect()

# COMMAND ----------

#num rows in airline data
subset_a.count()

# COMMAND ----------

#num rows in weather data
subset_w.count()

# COMMAND ----------

#num rows in stations
df_stations.count()

# COMMAND ----------

#list of stations table columns
df_stations.columns

# COMMAND ----------

#Join weather and stations (inner join)
join_ws = subset_w.join(df_stations,(subset_w.STATION==df_stations.station_id),'inner')
#when we do a left join (with weather on the left) we have a lot of nulls for our station columns, this means that the station_id isnt matching with the weahter STATION field. If we do an inner join we only get records where we have the station_id in both tables

# COMMAND ----------

join_ws.display(10)

# COMMAND ----------

join_ws.count()

# COMMAND ----------

a = subset_a.alias('a')
w = subset_w.alias('w')

# COMMAND ----------

#viewing unique combos of joining fields from airline data
a.select('ORIGIN','FL_DATE','dept_hour').distinct().sort('ORIGIN','FL_DATE','dept_hour').collect()

# COMMAND ----------

#viewing unique combos of joining fields from weather data
subset_w.select('plain_call_sign','plain_date','obs_hour').distinct().sort('plain_call_sign','plain_date','obs_hour').collect()

# COMMAND ----------

#join new df with airlines
joined_df = subset_a.join(subset_w,(a.ORIGIN == w.plain_call_sign ) & (a.FL_DATE ==subset_w.plain_date) & (a.dept_hour == w.obs_hour),'inner')

# .select('subset_a.*','ws.STATION','ws.DATE','ws.SOURCE','ws.LATITUDE','ws.LONGITUDE','ws.ELEVATION','ws.NAME','ws.REPORT_TYPE','ws.CALL_SIGN','ws.QUALITY_CONTROL','ws.WND','ws.CIG','ws.VIS','ws.TMP','ws.DEW','ws.SLP','ws.AW1','ws.GA1','ws.GA2','ws.GA3','ws.GA4','ws.GE1','ws.GF1','ws.KA1','ws.KA2','ws.MA1','ws.MD1','ws.MW1','ws.MW2','ws.OC1','ws.OD1','ws.OD2','ws.REM','ws.EQD','ws.AW2','ws.AX4','ws.GD1','ws.AW5','ws.GN1','ws.AJ1','ws.AW3','ws.MK1','ws.KA4','ws.GG3','ws.AN1','ws.RH1','ws.AU5','ws.HL1','ws.OB1','ws.AT8','ws.AW7','ws.AZ1','ws.CH1','ws.RH3','ws.GK1','ws.IB1','ws.AX1','ws.CT1','ws.AK1','ws.CN2','ws.OE1','ws.MW5','ws.AO1','ws.KA3','ws.AA3','ws.CR1','ws.CF2','ws.KB2','ws.GM1','ws.AT5','ws.AY2','ws.MW6','ws.MG1','ws.AH6','ws.AU2','ws.GD2','ws.AW4','ws.MF1','ws.AA1','ws.AH2','ws.AH3','ws.OE3','ws.AT6','ws.AL2','ws.AL3','ws.AX5','ws.IB2','ws.AI3','ws.CV3','ws.WA1','ws.GH1','ws.KF1','ws.CU2','ws.CT3','ws.SA1','ws.AU1','ws.KD2','ws.AI5','ws.GO1','ws.GD3','ws.CG3','ws.AI1','ws.AL1','ws.AW6','ws.MW4','ws.AX6','ws.CV1','ws.ME1','ws.KC2','ws.CN1','ws.UA1','ws.GD5','ws.UG2','ws.AT3','ws.AT4','ws.GJ1','ws.MV1','ws.GA5','ws.CT2','ws.CG2','ws.ED1','ws.AE1','ws.CO1','ws.KE1','ws.KB1','ws.AI4','ws.MW3','ws.KG2','ws.AA2','ws.AX2','ws.AY1','ws.RH2','ws.OE2','ws.CU3','ws.MH1','ws.AM1','ws.AU4','ws.GA6','ws.KG1','ws.AU3','ws.AT7','ws.KD1','ws.GL1','ws.IA1','ws.GG2','ws.OD3','ws.UG1','ws.CB1','ws.AI6','ws.CI1','ws.CV2','ws.AZ2','ws.AD1','ws.AH1','ws.WD1','ws.AA4','ws.KC1','ws.IA2','ws.CF3','ws.AI2','ws.AT1','ws.GD4','ws.AX3','ws.AH4','ws.KB3','ws.CU1','ws.CN4','ws.AT2','ws.CG1','ws.CF1','ws.GG1','ws.MV2','ws.CW1','ws.GG4','ws.AB1','ws.AH5','ws.CN3')

# COMMAND ----------

display(joined_df)

# COMMAND ----------

joined_df.count()

# COMMAND ----------

## write joined dataset to blob storage
joined_df.write.mode('overwrite').parquet(f"{blob_url}/joined_data")

# COMMAND ----------

# DBTITLE 1,Modeling: Start here to avoid rerunning
#read in joined dataset from blob
joined_df = spark.read.parquet(f"{blob_url}/joined_data")
joined_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wind Speed

# COMMAND ----------

# Create df in pandas
pd_df = joined_df.toPandas()

# Isolate wind speed from WND variable
pd_df["WND_SPD"] = pd_df["WND"].str.split(",", expand=True)[3]

# Add .1 meters per second of windspeed to aid the log transformation
pd_df["WND_SPD"] = pd_df["WND_SPD"].astype(int) + 1

# Remove missing wind speeds
pd_df = pd_df.loc[pd_df["WND_SPD"] != 10000]

# COMMAND ----------

# For missing delays, substitute the cancellation code
pd_df.loc[pd_df["DEP_DEL15"].isna(), "DEP_DEL15"] = pd_df.loc[pd_df["DEP_DEL15"].isna(), "CANCELLED"]

# COMMAND ----------

# Look at Wind speed data
pd_df["WND_SPD"].hist()

# COMMAND ----------

# Normalize wind speed data
pd_df["WND_SPD"] = np.sqrt(pd_df["WND_SPD"])
pd_df["WND_SPD"].hist()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Freezing Level

# COMMAND ----------

# Get the temperature in celsius
pd_df["TMP_C"] = pd_df["TMP"].str.split(",", expand=True)[0].astype(int)/10

# Remove missing temperatures
pd_df = pd_df.loc[pd_df["TMP_C"] != 999.9]

# COMMAND ----------

# Current temperature histogram
pd_df["TMP_C"].hist()

# COMMAND ----------

# Get the dew point in celsius
pd_df["DEW_C"] = pd_df["DEW"].str.split(",", expand=True)[0].astype(int)/10

# Remove missing dew point
pd_df = pd_df.loc[pd_df["DEW_C"] != 999.9]

# COMMAND ----------

# Current dewpoint histogram
pd_df["DEW_C"].hist()

# COMMAND ----------

# Create variable for the freezing level in feet
pd_df["freezing_level"] = ((pd_df["TMP_C"] - pd_df["DEW_C"])*500)

# COMMAND ----------

# Visualize freezing level
pd_df["freezing_level"].hist()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cloud Base

# COMMAND ----------

# Extract cloud base in meters
pd_df["cloud_base"] = pd_df["CIG"].str.split(",", expand=True)[0].astype(int)

# Remove missing clouds
pd_df = pd_df.loc[pd_df["cloud_base"] != 99999]

# Convert Cloud base from meters to feet
pd_df["cloud_base"] = pd_df["cloud_base"]*3.281

# COMMAND ----------

pd_df["cloud_base"].hist()

# COMMAND ----------

pd_df["clouds_over_freezing"] = pd_df["cloud_base"] - pd_df["freezing_level"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Basic Modeling

# COMMAND ----------

# Sort training data by day
train_df = pd_df.sort_values("FL_DATE").reset_index()

# COMMAND ----------

# Split training and testing data by day
train = train_df[:18069]
test = train_df[18069:]

# COMMAND ----------

train.tail()

# COMMAND ----------

test.head()

# COMMAND ----------

# Train-test split
train.shape[0]/train_df.shape[0]

# COMMAND ----------

train_cols = [
  "WND_SPD",
  "TMP_C",
  "DEW_C"
]

# COMMAND ----------

# Instantiate and train model
lr = LogisticRegressionCV()
lr.fit(train[train_cols], train["DEP_DEL15"])

# COMMAND ----------

# Score model with default
train_score = lr.score(train[train_cols], train["DEP_DEL15"])
test_score = lr.score(test[train_cols], test["DEP_DEL15"])

print(f"Model Training Accuracy: {train_score}")
print(f"Model Testing Accuracy: {test_score}")

# COMMAND ----------

lr.predict(train[train_cols]).mean()

# COMMAND ----------

