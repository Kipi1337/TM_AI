import pickle
import matplotlib.pyplot as plt

HEATMAP_FILE = "track_heatmap.pkl"

with open(HEATMAP_FILE, "rb") as f:
    data = pickle.load(f)

road = data["road"]
offtrack = data["offtrack"]
walls = data["walls"]

def plot_layer(layer, title, color):
    xs = [k[0] for k in layer.keys()]
    zs = [k[1] for k in layer.keys()]
    values = list(layer.values())

    plt.figure(figsize=(8,8))
    plt.scatter(xs, zs, c=values, cmap=color, s=5)
    plt.title(title)
    plt.axis("equal")
    plt.show()

plot_layer(road, "Road Heatmap", "viridis")
plot_layer(walls, "Wall / Collision Heatmap", "inferno")
plot_layer(offtrack, "Off-Track Heatmap", "cool")
