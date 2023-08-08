import pandas as pd
import geopandas as gpd
import streamlit as st
from streamlit_folium import st_folium
import numpy as np
import pickle
import warnings
import datetime
warnings.filterwarnings("ignore")
from PIL import Image
import folium as fl
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
#%%
nta = gpd.read_file("C:/Users/arunt/nyc/nynta2010_22c/nynta2010.shp")
nta_reprojected = nta.to_crs(4326)
pop = pd.read_csv("C:/Users/arunt/nyc/nynta2010_22c/Census_Demographics_at_the_Neighborhood_Tabulation_Area__NTA__level.csv")
new_pop = pop[['Geographic Area - Neighborhood Tabulation Area (NTA)* Code','Total Population 2010 Number']]
new_nta = nta_reprojected.merge(new_pop, how='left', left_on='NTACode', right_on='Geographic Area - Neighborhood Tabulation Area (NTA)* Code')
new_nta['area_km2'] = new_nta['Shape_Area'].values * 0.00000009290304
#%%
m = fl.Map(dragging=False,zoom_control=False,scrollWheelZoom=True)
m.add_child(fl.LatLngPopup())
m.fit_bounds([[40.8658,-74.00], [40.58,-73.9]]) 
maps = st_folium(m, height=1000, width=700)
    #data = get_pos(map['last_clicked']['lat'],map['last_clicked']['lng'])
    #fl.Marker([map['last_clicked']['lat'],map['last_clicked']['lng']]).add_to(m)
pickup_lat  = st.number_input("Select Pickup Latitude", 40.60000, 40.90000,step=1e-4,format="%.4f")
pickup_long  = st.number_input("Select Pickup Longitude" ,-74.05,-73.75,step=1e-4,format="%.4f")
drop_lat  = st.number_input("Select Dropoff Latitude", 40.60000, 40.90000,step=1e-4,format="%.4f")
drop_long  = st.number_input("Select Dropoff Longitude" ,-74.05,-73.75,step=1e-4,format="%.4f")
def haversine_array(lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        AVG_EARTH_RADIUS = 6371  # in km
        lat = lat2 - lat1
        lng = lng2 - lng1
        d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
        h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
        return h

def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

cosine = distance.cosine([pickup_lat,pickup_long],[drop_lat,drop_long])
haversine = haversine_array(pickup_lat,pickup_long,drop_lat,drop_long)
bearing = bearing_array(pickup_lat,pickup_long,drop_lat,drop_long)

no_pass = int(st.number_input("Select Number of Passengers",min_value = 0,max_value = 9))
v_id  = int(st.radio("Select Vendor ID" , [1,2]))
st.write(v_id)
flag  = st.radio("Select whether the trip was recorded" , ('N', 'Y'))
pick_time  = st.time_input("Select Pickup time" , datetime.time(10, 30))
pick_date  = st.date_input("Select Pickup date" , datetime.date(2016, 12, 12))
pick_hour = pick_time.hour
pick_day = pick_date.day
pick_month = pick_date.month
pick_week_year = pick_date.isocalendar().week
pick_day_week = pick_date.weekday()

pick_boro  = st.selectbox("Pickup Borogh: ",np.unique(nta['BoroName'].values)) 
pick_nta  = st.selectbox("Pickup NTA: ",np.unique(nta['NTACode'].values))
pick_pop = new_nta[new_nta['NTACode'] == pick_nta]['Total Population 2010 Number'].values[0]
pick_area = new_nta[new_nta['NTACode'] == pick_nta]['area_km2'].values[0]

drop_boro  = st.selectbox("Dropoff Borogh: ",np.unique(nta['BoroName'].values))
drop_nta  = st.selectbox("Dropoff NTA: ",np.unique(nta['NTACode'].values))
drop_pop = new_nta[new_nta['NTACode'] == drop_nta]['Total Population 2010 Number'].values[0]
drop_area = new_nta[new_nta['NTACode'] == drop_nta]['area_km2'].values[0]

final_columns = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',
       'pickup_hour', 'pickup_day', 'pickup_day_of_week','pickup_month', 'pickup_weekyear', 'haversine_dist', 'bearing_dist',
       'cosine_dist', 'pickup_boro', 'pickup_nta', 'dropoff_boro','dropoff_nta', 'pickup_pop', 'pickup_area', 'dropoff_pop','dropoff_area']

test_data = pd.DataFrame(np.array([[v_id,no_pass,pickup_long,pickup_lat,drop_long,drop_lat,flag,
        pick_hour,pick_day,pick_day_week,pick_month,pick_week_year,haversine,bearing,cosine,pick_boro,pick_nta,drop_boro,drop_nta,pick_pop,pick_area,drop_pop,drop_area]]), 
     columns = final_columns)
#%%
train_data = pd.read_csv('C:/Users/arunt/nyc/preprocessed_train_data.csv')
corrmat = np.round(train_data.corr(),3)
upper_tri = corrmat.where(np.triu(np.ones(corrmat.shape),k=1).astype(np.bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
to_drop.append('sqeuclidean_dist')
noncoll_train_data = train_data.drop(to_drop,axis=1)

ohe = OneHotEncoder(handle_unknown='ignore')
categorical_features = ['vendor_id','passenger_count','store_and_fwd_flag','pickup_day','pickup_month','pickup_hour','pickup_weekyear','pickup_day_of_week','pickup_boro','pickup_nta','dropoff_boro','dropoff_nta']
for i in categorical_features:
    noncoll_train_data[i]=noncoll_train_data[i].astype('object')
    test_data[i]=test_data[i].astype('object')  
train_cat = noncoll_train_data.select_dtypes('object')

test_cat = test_data[train_cat.columns]
#%%
convert_dict = {'vendor_id': int,
                'passenger_count': int,
                'pickup_hour': int,
                'pickup_day': int,
                'pickup_day_of_week': int,
                'pickup_month': int,
                'pickup_weekyear': int
                }
df = test_cat.astype(convert_dict)

#%%
conversion_dict = {'vendor_id': int,
                'passenger_count': int,
                'pickup_hour': int,
                'pickup_day': int,
                'pickup_day_of_week': int,
                'pickup_month': int,
                'pickup_weekyear': int,
                'pickup_longitude':float, 
                'pickup_latitude':float,
                'dropoff_longitude':float, 
                'dropoff_latitude':float,
                'haversine_dist':float, 
                'bearing_dist':float,
                'cosine_dist':float,
                'pickup_pop':float, 
                'pickup_area':float, 
                'dropoff_pop':float,
                'dropoff_area':float,
                'store_and_fwd_flag':object,
                'pickup_boro':object,
                'pickup_nta':object,
                'dropoff_boro':object,
                'dropoff_nta':object,
                }
new_test = test_data.astype(conversion_dict)
#%%
categorical_features = ['vendor_id', 'passenger_count', 'store_and_fwd_flag','pickup_hour', 'pickup_day', 'pickup_day_of_week', 'pickup_month','pickup_weekyear', 'pickup_boro', 'pickup_nta', 'dropoff_boro','dropoff_nta']
num_features = ['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude','haversine_dist', 'bearing_dist','cosine_dist','pickup_pop', 'pickup_area', 'dropoff_pop','dropoff_area']

test_cat = new_test[categorical_features]

train_cat_features = ohe.fit_transform(train_cat.values).toarray()
train_feature_names = ohe.get_feature_names_out(train_cat.columns.values)
test_cat_features = ohe.transform(test_cat.values).toarray()
test_feature_names = ohe.get_feature_names_out(test_cat.columns.values)

train_ohe = pd.concat([train_data[num_features],pd.DataFrame(train_cat_features,columns=train_feature_names).astype(int)], axis=1)
test_ohe = pd.concat([new_test[num_features],pd.DataFrame(test_cat_features,columns=test_feature_names).astype(int)], axis=1)

#%%
ss = StandardScaler()
X_train_ss = ss.fit_transform(train_ohe)
X_test_ss = ss.transform(test_ohe)
#%%
y_train = train_data['trip_duration'].values
#%%
from pickle import dump
dump(ohe, open('ohe.pkl', 'wb'))
dump(ss, open('ss.pkl', 'wb'))
#%%
from xgboost import XGBRegressor
xgb_model = XGBRegressor(objective="reg:squarederror",
                         random_state=100,
                         max_depth=20,
                         min_child_weight=7,
                         reg_lambda=0.5,
                         nthread=-1)

xgb_model.fit(X_train_ss,np.log(y_train)+1)