import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
import matplotlib.pyplot as plt
import folium
from sklearn.cluster import KMeans
from folium.plugins import MarkerCluster
from matplotlib import cm, colors

# Caricamento dati e preparazione coordinate
df = pd.read_csv("clienti_validi_geocodificati.csv")
coordinates = df[['latitudine', 'longitudine']]

# Standardizzazione delle coordinate geografiche
scaled = StandardScaler().fit_transform(coordinates)

# K-distance plot per determinare il parametro eps
neighbors = NearestNeighbors(n_neighbors=4)
d, _ = neighbors.fit(scaled).kneighbors(scaled)
k_d = np.sort(d[:, -1])

plt.figure(figsize=(8, 4))
plt.plot(k_d)
plt.xlabel("Punti ordinati")
plt.ylabel("Distanza al 4° vicino")
plt.title("Curva k-distance (min_samples = 4)")
plt.grid(True)
plt.show()

# Clustering con DBSCAN
db = DBSCAN(eps=0.3, min_samples=4)
df['cluster'] = db.fit_predict(scaled)
labels = db.labels_

# Calcolo del Silhouette Score escludendo gli outlier
if len(set(labels)) > 1 and -1 in labels:
    mask = labels != -1
    score = silhouette_score(scaled[mask], labels[mask])
    print(f"Silhouette Score: {score:.3f}")
else:
    print("Silhouette Score non calcolabile")

# Distribuzione dei clienti in ciascun cluster
print(df['cluster'].value_counts())

# Funzione per calcolo della densità locale (raggio: 10 km)
def local_density(lat, lon, data, r_km=10.0):
    from geopy.distance import geodesic
    return sum(
        geodesic((lat, lon), (r['latitudine'], r['longitudine'])).km <= r_km
        for _, r in data.iterrows()
    )

# Calcolo densità locale per i punti identificati come outlier da DBSCAN
outliers = df[df['cluster'] == -1].copy()
outliers['local_density'] = outliers.apply(
    lambda r: local_density(r['latitudine'], r['longitudine'], df), axis=1
)

# Sottoclusterizzazione del cluster 0 con KMeans (4 gruppi)
cluster0_df = df[df['cluster'] == 0].copy()
coords0 = cluster0_df[['latitudine', 'longitudine']].values
coords0_scaled = StandardScaler().fit_transform(coords0)
kmeans0 = KMeans(n_clusters=4, random_state=42).fit(coords0_scaled)
cluster0_df['subcluster'] = kmeans0.labels_

# Sottoclusterizzazione del cluster 1 con KMeans (4 gruppi)
cluster1_df = df[df['cluster'] == 1].copy()
coords1 = cluster1_df[['latitudine', 'longitudine']].values
coords1_scaled = StandardScaler().fit_transform(coords1)
kmeans1 = KMeans(n_clusters=4, random_state=42).fit(coords1_scaled)
cluster1_df['subcluster'] = kmeans1.labels_

# Composizione finale dei cluster
cluster0_df['cluster_finale'] = '0_' + cluster0_df['subcluster'].astype(str)
cluster1_df['cluster_finale'] = '1_' + cluster1_df['subcluster'].astype(str)
rest_df = df[df['cluster'].isin([2, 3])].copy()
rest_df['cluster_finale'] = rest_df['cluster'].astype(str)
outlier_df = df[df['cluster'] == -1].copy()
outlier_df['cluster_finale'] = 'outlier'

# Unione di tutti i dataframe in uno unico finale
df_finale = pd.concat([cluster0_df, cluster1_df, rest_df, outlier_df], ignore_index=True)

# Riassegnazione condizionata degli outlier
core_pts = scaled[labels != -1]
core_lbl = labels[labels != -1]
outlier_pts = scaled[labels == -1]

# Calcolo del cluster più vicino per ogni outlier
nearest_cluster, distances = pairwise_distances_argmin_min(outlier_pts, core_pts)
assigned = core_lbl[nearest_cluster]

# Condizioni per riassegnazione: vicino + sufficiente densità locale
density_thresh = 3
distance_thresh = 0.35
out_idx = outliers.index
reassign_mask = (distances < distance_thresh) & (outliers['local_density'] >= density_thresh)

# Applicazione della riassegnazione
df_finale['cluster_riparato'] = df_finale['cluster']
df_finale['riassegnato'] = False
df_finale.loc[out_idx[reassign_mask], 'cluster_riparato'] = assigned[reassign_mask.values]
df_finale.loc[out_idx[reassign_mask], 'riassegnato'] = True

print(f"\nOutlier riassegnati: {reassign_mask.sum()} su {len(outliers)}")

# Visualizzazione su mappa interattiva con Folium
map_center = [df_finale['latitudine'].mean(), df_finale['longitudine'].mean()]
m = folium.Map(location=map_center, zoom_start=10)
mc = MarkerCluster().add_to(m)

# Preparazione dei colori per ogni cluster riparato
n_clusters = df_finale[df_finale['cluster_riparato'] != -1]['cluster_riparato'].nunique()
cmap = cm.get_cmap('Set1', n_clusters)
cluster_labels = sorted(df_finale[df_finale['cluster_riparato'] != -1]['cluster_riparato'].unique())
cluster_colors = {label: colors.to_hex(cmap(i)) for i, label in enumerate(cluster_labels)}

# Aggiunta dei marker alla mappa
for _, r in df_finale.iterrows():
    if r['cluster_riparato'] != -1:
        col = cluster_colors[r['cluster_riparato']]
        folium.CircleMarker(
            location=[r['latitudine'], r['longitudine']],
            radius=5,
            color=col,
            fill=True,
            fill_color=col,
            fill_opacity=0.9,
            popup=f"Cliente {r['Codice Cliente']}, Cluster {r['cluster_riparato']}"
        ).add_to(mc)
    else:
        folium.Marker(
            location=[r['latitudine'], r['longitudine']],
            popup=f"Cliente {r['Codice Cliente']} (Outlier)",
            icon=folium.Icon(color='red')
        ).add_to(m)


# Esportazione dei risultati
m.save("clienti_clusters_outliers.html")
df_finale.to_csv("clienti_con_cluster_riparato.csv", index=False)

print(df_finale['cluster'].value_counts())

