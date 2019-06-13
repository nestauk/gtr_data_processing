#We extract the data from the Nesta S3 database

#Data getter imports
from data_getters.core import get_engine
from data_getters.inspector import get_schemas

#Processing imports
import pandas as pd
import os
import time
from datetime import datetime

today_str = str(datetime.now()).split(' ')[0]

#Config path
config_path = '../../mysqldb_team.config'

#Create output directories

if 'raw' not in os.listdir('../../data/'):
    os.mkdir('../../data/raw/gtr')

os.mkdir(f'../../data/raw/gtr/{today_str}')

#Get schemas
schemas = get_schemas(config_path)

#Get gtr tables
gtr_tables = schemas['gtr'].keys()

#Initialise the SQL engine
engine = get_engine(config_path,pool_size=30, max_overflow=0)

#Get the data

#Queries the database
print("Querying the database")

check = []

for table in gtr_tables:
    print(table)

    check.append(pd.read_sql_table(table,engine,chunksize=5000)) 

    time.sleep(3)

#Appends the data into a df and saves
print("Saving the data")

all_data = []

for n,t in enumerate(gtr_tables):
    print(t)

    dfs = []
    
    for results in check[n]:
        dfs.append(results)
    
    df = pd.concat(dfs)
    
    df.to_csv(f'../../data/raw/gtr/{today_str}/{t}.csv')
    
    all_data.append(df)