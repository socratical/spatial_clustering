import seaborn as sns
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pandas.io.json import json_normalize
from pysal.lib import weights
from sklearn import cluster
from shapely.geometry import Point

# london_abb = gpd.read_file('/home/nathanh/Documents/spatial_cluster/ilm_abb.geojson')
# crs = coordinate reference systems
# london_abb.crs = {'init': u'epsg:27700'}
# london_abb has a MSOA_id column meaning Middle Super Output Area
# london_abb.info()
# london_abb.info()
# outputs geometric data for geopandas
# print(london_abb["geometry"].head())
# print(london_abb["beds"])

dataInput = '/home/nathanh/Documents/spatial_cluster/seattle_listings.csv'
geoInput = '/home/nathanh/Documents/spatial_cluster/seattle_neighbourhoods.geojson'
pd_abb = pd.read_csv(dataInput, usecols=["accommodates", "bathrooms", "bedrooms",
                                         "beds", "number_of_reviews", "reviews_per_month",
                                         "review_scores_rating", "review_scores_accuracy",
                                         "review_scores_cleanliness", "review_scores_checkin",
                                         "review_scores_communication", "review_scores_location",
                                         "review_scores_value", "neighbourhood"])
pd_abb = pd_abb.sort_values(by=['neighbourhood'], ascending=['True'])
gpd_abb = gpd.read_file(geoInput)
pd_abb = pd_abb.groupby('neighbourhood')["accommodates", "bathrooms", "bedrooms",
                                         "beds", "number_of_reviews", "reviews_per_month",
                                         "review_scores_rating", "review_scores_accuracy",
                                         "review_scores_cleanliness", "review_scores_checkin",
                                         "review_scores_communication", "review_scores_location",
                                         "review_scores_value"].mean()

# print(pd_abb.head())
# print(list(pd_abb.columns.values))
# gpd_abb = gpd_abb.set_index('neighbourhood').sort_index()
# gpd_abb = gpd_abb.rename(columns={'neighbourhood': 'zipcode'})
# gpd_abb['zipcode'] = gpd_abb['zipcode'].astype(int)
# gpd_abb.info()
gpd_abb = gpd_abb.merge(pd_abb, on='neighbourhood').sort_values(['neighbourhood'], ascending=[True])
gpd_abb = gpd_abb.drop(['neighbourhood_group'], axis=1)
print(gpd_abb['review_scores_value'].to_string())
print(gpd_abb.tail(10))
gpd_abb = gpd_abb[np.isfinite(gpd_abb['review_scores_value'])]
gpd_abb[["review_scores_checkin",
         "review_scores_communication", "review_scores_location",
         "review_scores_value"]].to_csv('test_stuff.csv', sep='\t')
# pd_zipAvg = gpd_abb.merge(pd_abb.groupby('zipcode')['accommodates', 'bathrooms'].mean())
# print(list(pd_zipAvg.columns.values))
# pd_zipAvg.info()
# print(pd_zipAvg['zipcode'].sort_values().to_string())
ratings = ['review_scores_rating', 'review_scores_accuracy',
           'review_scores_cleanliness', 'review_scores_checkin',
           'review_scores_communication', 'review_scores_location',
           'review_scores_value']

# Create figure and axes (this time it's 9, arranged 3 by 3)
f, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
# Make the axes accessible with single indexing
axs = axs.flatten()
# Start the loop over all the variables of interest
for i, col in enumerate(ratings):
    # select the axis where the map will go
    ax = axs[i]
    # PLOT the map
    gpd_abb.plot(column=col, ax=ax, scheme='Quantiles', linewidth=0, cmap='Dark2', alpha=0.75, legend=True)
    # Remove axis clutter
    ax.set_axis_off()
    # Set the axis title to the name of variable being plotted
    ax.set_title(col)
plt.savefig('SeatleQuantiles')

# PLOT for bivariate correlations, a useful tool is the correlation matrix plot, available in seaborn
_ = sns.pairplot(gpd_abb[ratings], kind='reg', diag_kind='kde', )
plt.savefig('BivariateCorrelation')
plt.savefig('SeatleBivariate')

# KMeans Classification
kmeans3 = cluster.KMeans(n_clusters=3)
kmeans4 = cluster.KMeans(n_clusters=4)
kmeans5 = cluster.KMeans(n_clusters=5)
kmeans6 = cluster.KMeans(n_clusters=6)
# Possible User Input
choice_kmeans = kmeans4
# Possible User Input
choice_seed = 9133
np.random.seed(choice_seed)
# Run Cluster Algo
kclust = choice_kmeans.fit(gpd_abb[ratings])
# Append Data to GeoPandas Table
gpd_abb['kclust'] = kclust.labels_

# KMeans Map
# Setup figure and ax
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot unique values choropleth including a legend and with no boundary lines
gpd_abb.plot(column='kclust', categorical=True, legend=True, linewidth=0, ax=ax)
# Remove axis
ax.set_axis_off()
# Keep axes proportionate
plt.axis('equal')
# PLOT kmeans
plt.savefig('SeatleKMeans')
