import pandas as pd
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Path to the cleaned CSV file with snippets
DATA_PATH = "data/Extended_Restaurant_Dataset_Unique.csv"

def recommend_top_rated(user_lat, user_lon, radius_km=2.0, top_n=10):
    df = pd.read_csv(DATA_PATH)

    # Drop rows with missing coordinates
    df = df.dropna(subset=["latitude", "longitude"])

    # Filter by radius
    def is_within_radius(row):
        distance = geodesic((user_lat, user_lon), (row["latitude"], row["longitude"])).km
        return distance <= radius_km

    df["within_radius"] = df.apply(is_within_radius, axis=1)
    nearby_df = df[df["within_radius"]].copy()

    if nearby_df.empty:
        return pd.DataFrame(columns=["name", "address", "rating", "category_embedding", "snippet"])

    # Cluster by location (optional but useful for visualizations)
    coords = nearby_df[["latitude", "longitude"]]
    coords_scaled = StandardScaler().fit_transform(coords)
    db = DBSCAN(eps=0.5, min_samples=3).fit(coords_scaled)
    nearby_df["cluster"] = db.labels_

    # Sort by rating (descending) and return top results
    top_df = nearby_df.sort_values(by="rating", ascending=False).head(top_n)

    return top_df[["name", "address", "rating", "category_embedding", "snippet", "latitude", "longitude"]]

# Optional: visualization (if needed in Streamlit or notebooks)
def visualize_map(df, center=(40.7580, -73.9855)):
    import folium

    m = folium.Map(location=center, zoom_start=14)
    for _, row in df.iterrows():
        popup = f"{row['name']} - {row['category_embedding']}<br>{row['snippet']}"
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=popup,
            icon=folium.Icon(color="blue")
        ).add_to(m)
    return m