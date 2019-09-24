import seaborn as sns
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize
from pysal.lib import weights
from sklearn import cluster
from shapely.geometry import Point


# # # # # PET DATA # # # # #


# filename = "pets.json"
# with open(filename, 'r') as f:
#    objects = ijson.items
# austin dangerous dog api
urlD = 'https://data.austintexas.gov/resource/ykw4-j3aj.json'
# austin stray dog data
urlS = 'https://data.austintexas.gov/resource/hye6-gvq2.json'

# found_df / austin found pets pandas data frame constructor
pets_df = pd.read_json(urlS, orient='records')
location_df = json_normalize(pets_df['location'])
concat_df = pd.concat([pets_df, location_df], axis=1)
found_df = concat_df.drop(concat_df.columns[0:7], axis=1)
found_df = found_df.drop(found_df.columns[[2, 4, 6, 10]], axis=1)
address_df = pd.DataFrame(columns=['address', 'city', 'zip_code'])
for i, row in location_df.iterrows():
    rowStr = row['human_address']
    splitRow = rowStr.split('\"')
    address = splitRow[3]
    city = splitRow[7]
    zipCode = splitRow[15]
    address_df = address_df.append({'address': address, 'city': city, 'zip_code': zipCode}, ignore_index=True)
found_df = pd.concat([found_df, address_df], axis=1)
#       formatting address correctly
for i, row in found_df.iterrows():
    rowStr = row['city']
    splitRow = rowStr.split(' ')
#       ADD MORE LOCALITIES HERE IF NEEDED IN DATASET
    if splitRow[0] not in ('AUSTIN', 'PFLUGERVILLE', 'LAKEWAY', ''):
        for j in splitRow:
            if j in ('AUSTIN', 'PFLUGERVILLE', 'LAKEWAY'):
                found_df.at[i, 'city'] = j
            else:
                found_df.at[i, 'city'] = ''
        found_df.at[i, 'address'] = ''


# danger_df austin dangerous dogs pandas data frame constructor
danger_df = pd.read_json(urlD)
danger_df = danger_df.drop(danger_df.columns[[0, 1, 4, 5]], axis=1)
location_df = json_normalize(danger_df['location'])
address_df = pd.DataFrame(columns=['address'])
for i, row in location_df.iterrows():
    rowStr = row['human_address']
    splitRow = rowStr.split('\"')
    address = splitRow[3]
    address_df = address_df.append({'address': address}, ignore_index=True)
danger_df = danger_df.drop(danger_df.columns[[2]], axis=1)
location_df = location_df.drop(location_df.columns[[0]], axis=1)
danger_df = pd.concat([danger_df, address_df, location_df], axis=1)

# converting data types
found_df["latitude"] = pd.to_numeric(found_df["latitude"])
found_df["longitude"] = pd.to_numeric(found_df["longitude"])
found_df["zip_code"] = pd.to_numeric(found_df["zip_code"])
danger_df["latitude"] = pd.to_numeric(found_df["latitude"])
danger_df["longitude"] = pd.to_numeric(found_df["longitude"])
danger_df["zip_code"] = pd.to_numeric(found_df["zip_code"])

# aggregate/averages by cat vs dog
sort_zip = found_df.sort_values(by=["zip_code", "type"], ascending=[True, False])
# plotting austin zip codes
f, ax = plt.subplots(1, figsize=(7, 7))
zc = gpd.read_file('/home/nathanh/Documents/spatial_cluster/austin_zipcodes.geojson')
zc.plot(linewidth=0.1, ax=ax)
found_df['geometry'] = found_df[['longitude', 'latitude']].apply(Point, axis=1)
found_gdf = gpd.GeoDataFrame(found_df)
found_gdf.crs = {'init': 'epsg:4269'}
found_gdf.plot(color='red', ax=ax)
ax.set_axis_off()
plt.axis('equal')
plt.savefig('PetPlot')


# # # # # LONDON DATA # # # # #


london_abb = gpd.read_file('/home/nathanh/Documents/spatial_cluster/ilm_abb.geojson')
# crs = coordinate reference systems
london_abb.crs = {'init': u'epsg:27700'}
# london_abb has a MSOA_id column meaning Middle Super Output Area
# .info() prints info
# london_abb.info()
# outputs geometric data for geopandas
# print(london_abb["geometry"].head())
austin_abb = gpd.read_file('/home/nathanh/Documents/spatial_cluster/ilm_abb.geojson')

ratings = ['review_scores_rating', 'review_scores_accuracy',
           'review_scores_cleanliness', 'review_scores_checkin',
           'review_scores_communication', 'review_scores_location',
           'review_scores_value']
# Possible User Input
choice_abb = london_abb

# Create figure and axes (this time it's 9, arranged 3 by 3)
f, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
# Make the axes accessible with single indexing
axs = axs.flatten()
# Start the loop over all the variables of interest
for i, col in enumerate(ratings):
    # select the axis where the map will go
    ax = axs[i]
    # PLOT the map
    choice_abb.plot(column=col, ax=ax, scheme='Quantiles', linewidth=0, cmap='Greens', alpha=0.75)
    # Remove axis clutter
    ax.set_axis_off()
    # Set the axis title to the name of variable being plotted
    ax.set_title(col)
plt.savefig('Quantiles')

# PLOT for bivariate correlations, a useful tool is the correlation matrix plot, available in seaborn
_ = sns.pairplot(choice_abb[ratings], kind='reg', diag_kind='kde', )
plt.savefig('BivariateCorrelation')

# KMeans Classification
kmeans3 = cluster.KMeans(n_clusters=5)
kmeans4 = cluster.KMeans(n_clusters=5)
kmeans5 = cluster.KMeans(n_clusters=5)
kmeans6 = cluster.KMeans(n_clusters=5)
# Possible User Input
choice_kmeans = kmeans5
# Possible User Input
choice_seed = 9133
np.random.seed(choice_seed)
# Run Cluster Algo
kclust = choice_kmeans.fit(choice_abb[ratings])
# Append Data to GeoPandas Table
choice_abb['kclust'] = kclust.labels_

# KMeans Map
# Setup figure and ax
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot unique values choropleth including a legend and with no boundary lines
choice_abb.plot(column='kclust', categorical=True, legend=True, linewidth=0, ax=ax)
# Remove axis
ax.set_axis_off()
# Keep axes proportionate
plt.axis('equal')
# PLOT kmeans
plt.savefig('KMeans')

# Returns the number of elements in each subgroup
ksizes = choice_abb.groupby('kclust').size()
# print(ksizes)
plt.subplot()
_ = ksizes.plot.bar(color='m')
plt.savefig('KMeansElements')

# Calculate the mean by group
group_kmeans = choice_abb.groupby('kclust')[ratings].mean()
# Show the table transposed (so it's not too wide)
# print(group_kmeans.T)
# Calculate the summary by group (description)
desc_kmeans = choice_abb.groupby('kclust')[ratings].describe().head()
# print(desc_kmeans.T)


# Regionalization Algorithms
# Regions Representing a Set of Similar Areas in terms of ratings
# ESPD: Exploratory Spatial Data Analysis
# Regionalization methods require a formal representation of space
# that is statistics-friendly. In practice, this means that we will
# need to create a spatial weights matrix for the areas to be aggregated.
# A commonly-used type of weight is a queen contigutiy weight,
# which reflects adjacency relationships as a binary indicator variable
# denoting whether or not a polygon shares an edge or a vertex with another polygon
w_knn = weights.KNN.from_dataframe(choice_abb)
# w_knn.plot(choice_abb)
sagg12 = cluster.AgglomerativeClustering(n_clusters=12, connectivity=w_knn.sparse)
sagg12cls = sagg12.fit(choice_abb[ratings])
choice_abb['sagg13cls'] = sagg12cls.labels_
# Setup figure and ax
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot unique values choropleth including a legend and with no boundary lines
choice_abb.plot(column='sagg13cls', categorical=True, legend=True, linewidth=0, ax=ax)
# Remove axis
ax.set_axis_off()
# Keep axes proportionate
plt.axis('equal')
# Add title
# Display the figure
plt.savefig('KNN')




# displays unique values
# print(found_df.head())
# print(danger_df.dtypes)
# print(found_df.city.unique())
# print(location_df.at[0, 'human_address'])
# print(address_df.dtypes)
# print(address_df.head())
# print(found_df.dtypes)
# print(found_df.head())
# print(pets_df['location', 'human_address'].head())

# trying to convert raw json, is incomplete
# r = requests.get('https://data.austintexas.gov/api/views/hye6-gvq2/rows.json?accessType=DOWNLOAD')
# petjson = r.text
# df = pd.read_json(petjson, orient='records')
# print(df)
# with petjson as j:
#    objects = ijson.items(j, 'location.human_address')
#    columns = list(objects)"""
