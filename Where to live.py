#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

get_ipython().system("conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab")
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system("conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab")
import folium # map rendering library

print('Libraries imported.')


# In[34]:


df = pd.read_csv("DC.csv")
df.head()


# In[35]:


address = 'Washington, DC'

geolocator = Nominatim(user_agent="dc_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude


# In[36]:


map_dc = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df['Latitude'], df['Longitude'], df['Ward'], df['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_dc)  
    
map_dc
#fn='testmap.html'
#map_dc.save(fn)


# In[37]:


best_venues = pd.read_csv("best_venues_dc.csv")
best_venues = best_venues.drop("Unnamed: 0", axis =1)
best_venues.head()


# In[38]:


best_onehot = pd.get_dummies(best_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
best_onehot['Neighborhood'] = best_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [best_onehot.columns[-1]] + list(best_onehot.columns[:-1])
best_onehot = best_onehot[fixed_columns]

best_onehot.head()


# In[39]:


best_grouped = best_onehot.groupby('Neighborhood').mean().reset_index()
best_grouped


# In[40]:


num_top_venues = 10

for hood in best_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = best_grouped[best_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[41]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[42]:


num_top_venues = 5

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = best_grouped['Neighborhood']

for ind in np.arange(best_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(best_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[43]:


kclusters = 3

best_grouped_clustering = best_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(best_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[44]:


neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

best_merged = best_venues

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
best_merged = best_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

best_merged = best_merged.dropna()

best_merged.head() # check the last columns!


# In[46]:


map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(best_merged['Neighborhood Latitude'], best_merged['Neighborhood Longitude'], best_merged['Neighborhood'], best_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)

map_clusters


# In[14]:


best_merged.loc[best_merged['Cluster Labels'] == 2, best_merged.columns[[1] + list(range(5, best_merged.shape[1]))]]


# In[15]:


df = pd.read_csv("Arrests 2017 Public.csv")
df.head()


# In[16]:


columns = ["Defendant Race", "Defendant Sex", "Arrest Category","Arrest Location District","Offense Latitude","Offense Longitude"]
df = df[columns]
df = df.dropna()
df["Arrest Location District"] = df["Arrest Location District"].str.replace("D","")
#df=df[:500]


# In[17]:


'''address = 'Washington, DC'

geolocator = Nominatim(user_agent="dc_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude


map_dc = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df["Offense Latitude"], df["Offense Longitude"], df["Arrest Location District"], df["Arrest Category"]):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_dc)  
    
map_dc'''


# In[18]:


best_onehot = pd.get_dummies(df[["Arrest Category"]])
best_onehot["Arrest Location District"] = df["Arrest Location District"]     
fixed_columns = [best_onehot.columns[-1]] + list(best_onehot.columns[:-1])
best_onehot = best_onehot[fixed_columns]

best_onehot.head()


# In[19]:


best_grouped = best_onehot.groupby("Arrest Location District").mean().reset_index()
best_grouped


# In[20]:


num_top_crime = 10

for hood in best_grouped["Arrest Location District"]:
    print("----"+hood+"----")
    temp = best_grouped[best_grouped["Arrest Location District"] == hood].T.reset_index()
    temp.columns = ['crime','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_crime))
    print('\n')


# In[21]:


def return_most_common_venues(row, num_top_crimes):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_crimes]


# In[22]:


num_top_crimes = 10
indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ["Arrest Location District"]
for ind in np.arange(num_top_crimes):
    try:
        columns.append('{}{} Most Common Crime'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Crime'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted["Arrest Location District"] = best_grouped["Arrest Location District"]

for ind in np.arange(best_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(best_grouped.iloc[ind, :], num_top_crimes)


neighborhoods_venues_sorted.head()


# In[23]:


kclusters = 4

best_grouped_clustering = best_grouped.drop("Arrest Location District", 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(best_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[24]:


#neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

best_merged = df.drop(16, axis = 0)

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
best_merged = best_merged.join(neighborhoods_venues_sorted.set_index("Arrest Location District"), on="Arrest Location District")

best_merged = best_merged.dropna()

best_merged = best_merged[:500]

dis= [1,2,3,4,5,6,7]
cluser = [3, 2, 2, 1, 3, 2, 1, 0]
geo = pd.DataFrame(list(zip(dis, cluser)), 
               columns =["Arrest Location District", 'Cluster Labels'])
geo


# In[25]:


import json

with open('police-districts-mpd.geojson.txt') as f:
    data = json.load(f)
print(data)


# In[48]:


map_clusters.choropleth(
        geo_data=data,
        data= geo,
        columns=['Arrest Location District', 'Cluster Labels'],
        key_on='feature.properties.OBJECTID',
        fill_color="YlOrBr",
        highlight = False,
        fill_opacity=0.4,
        line_opacity=0.7,
        legend_name='Clusters in DC'
        )

map_clusters


# In[ ]:




