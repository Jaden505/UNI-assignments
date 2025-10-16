import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import folium

np.random.seed(42)

title = "Noord-Holland: 20-steden migratiegrafiek"
region_desc = "Noord-Holland gekozen vanwege hoge dichtheid, OV-netwerk en realistische verhuisstromen."

cities = [
    ("Amsterdam", 52.3676, 4.9041),
    ("Amstelveen", 52.3080, 4.8422),
    ("Haarlem", 52.3874, 4.6462),
    ("Hoofddorp", 52.3061, 4.6900),
    ("Zaandam", 52.4385, 4.8260),
    ("Diemen", 52.3390, 4.9620),
    ("Weesp", 52.3075, 5.0413),
    ("Aalsmeer", 52.2590, 4.7599),
    ("Uithoorn", 52.2371, 4.8292),
    ("Hilversum", 52.2233, 5.1765),
    ("Naarden", 52.2950, 5.1611),
    ("Bussum", 52.2730, 5.1610),
    ("Alkmaar", 52.6324, 4.7534),
    ("Hoorn", 52.6425, 5.0594),
    ("Purmerend", 52.5050, 4.9597),
    ("Volendam", 52.4950, 5.0705),
    ("Monnickendam", 52.4568, 5.0376),
    ("IJmuiden", 52.4600, 4.6100),
    ("Zandvoort", 52.3710, 4.5333),
    ("Schiphol", 52.3086, 4.7639),
]

df = pd.DataFrame(cities, columns=["city","lat","lon"])
n = len(df)

# helper function for locations
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p = np.radians(lat2-lat1)
    q = np.radians(lon2-lon1)
    a = np.sin(p/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(q/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

D = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if i==j:
            D[i,j]=0.0
        else:
            D[i,j]=haversine(df.lat[i], df.lon[i], df.lat[j], df.lon[j])

# Build graph with constraints
G = nx.Graph()
for i, r in df.iterrows():
    G.add_node(i)

base_k = 3 # each node at least this many edges
for j in range(n):
    idx = np.argsort(D[j])[1:base_k+1]
    for i in idx:
        G.add_edge(j, i)

# A hub is a city that should have at least 6 connections
hubs = {df.index[df.city=="Amsterdam"][0], df.index[df.city=="Haarlem"][0], df.index[df.city=="Alkmaar"][0], df.index[df.city=="Hilversum"][0], df.index[df.city=="Schiphol"][0]}
for h in hubs:
    need = 6 - G.degree[h]
    if need > 0:
        candidates = [i for i in np.argsort(D[h])[1:12] if i!=h and not G.has_edge(h,i)]
        for i in candidates[:need]:
            G.add_edge(h,i)


if not nx.is_connected(G):
    comps = list(nx.connected_components(G))
    while len(comps) > 1:
        c1 = list(comps[0])
        c2 = list(comps[1])
        best = None
        bestd = 1e9
        for a in c1:
            for b in c2:
                if D[a,b] < bestd:
                    bestd = D[a,b]
                    best = (a,b)
        G.add_edge(best[0], best[1])
        comps = list(nx.connected_components(G))


max_deg = int(np.floor(n/2))-1
for node in list(G.nodes()):
    deg = G.degree[node]
    if deg > max_deg:
        nbrs = sorted(G.neighbors(node), key=lambda x: D[node,x], reverse=True)
        for v in nbrs:
            if G.degree[node] <= max_deg:
                break
            if G.degree[v] > 2 and (node not in hubs):
                G.remove_edge(node, v)


A_adj = np.zeros((n,n))
for u,v in G.edges():
    d = D[u,v]
    w = 2.0 if d < 10 else (1.0 if d < 20 else 0.5)
    A_adj[v,u] = w
    A_adj[u,v] = w


loops = {
    df.index[df.city=="Amsterdam"][0]: 2.0,
    df.index[df.city=="Schiphol"][0]: 2.0,
    df.index[df.city=="Haarlem"][0]: 1.0,
    df.index[df.city=="Alkmaar"][0]: 1.0,
    df.index[df.city=="Hilversum"][0]: 1.0,
}


for i, w in loops.items():
    A_adj[i,i] = w

for j in range(n):
    if A_adj[:,j].sum() == 0:
        near = np.argsort(D[j])[1:3]
        for i in near:
            A_adj[i,j] = 1.0

col_sums = A_adj.sum(axis=0)
A_prob = A_adj / col_sums



# WRITING OUTPUTS FOR REPORT
pd.DataFrame(A_adj, index=df.city, columns=df.city).to_csv("adjacency_matrix_weighted.csv")
pd.DataFrame(A_prob, index=df.city, columns=df.city).to_csv("probability_matrix.csv")

m = folium.Map(location=[df.lat.mean(), df.lon.mean()], zoom_start=10, tiles="OpenStreetMap")
for i, r in df.iterrows():
    folium.CircleMarker(location=[r.lat, r.lon], radius=5, popup=r.city).add_to(m)
for u,v in G.edges():
    a = df.iloc[u]
    b = df.iloc[v]
    folium.PolyLine([[a.lat, a.lon],[b.lat,b.lon]], weight=2, opacity=0.7).add_to(m)
m.save("map_20_cities_connections.html")

def mat_power(A, p): # utils func
    return np.linalg.matrix_power(A, int(p))

def simulate_spread(A, start_idx, friends=100, moves=6):
    """ Simulate spread of population over the graph"""
    v0 = np.zeros(A.shape[0]) 
    v0[start_idx] = friends
    P = mat_power(A, moves)
    return P @ v0, P

moves_20y = 6
estimate_text = "Aanname: gemiddeld 1 verhuizing per ~3 jaar in jonge/werkzame populatie in deze regio; over 20 jaar ~6 stappen."

start_a = int(df.index[df.city=="Amstelveen"][0])
start_b = int(df.index[df.city=="Amsterdam"][0])

v_from_a, A_pow_x = simulate_spread(A_prob, start_a, friends=100, moves=moves_20y)
v_from_b, _ = simulate_spread(A_prob, start_b, friends=100, moves=moves_20y)

A8 = mat_power(A_prob, 8)
u = np.ones(n)
A32u = mat_power(A_prob, 32) @ u

def bar_plot(values, labels, title, outpng):
    plt.figure()
    x = np.arange(len(values))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()

bar_plot(v_from_a, df.city.tolist(), f"Verdeling na {moves_20y} stappen vanaf Amstelveen", "spread_from_amstelveen.png")
bar_plot(v_from_b, df.city.tolist(), f"Verdeling na {moves_20y} stappen vanaf Amsterdam", "spread_from_amsterdam.png")
bar_plot(A32u, df.city.tolist(), "A^32 * u", "a32u.png")

summary = pd.DataFrame({
    "city": df.city,
    "v_from_amstelveen": v_from_a,
    "v_from_amsterdam": v_from_b,
    "A32u": A32u,
    "in_degree": [int((A_adj[:,j]>0).sum() - (A_adj[j,j]>0)) for j in range(n)],
    "out_degree": [int((A_adj[j,:]>0).sum() - (A_adj[j,j]>0)) for j in range(n)],
})
summary.to_csv("results_summary.csv", index=False)

with open("moves_estimate.txt","w") as f:
    f.write(f"x = {moves_20y}\n"+estimate_text+"\n")

