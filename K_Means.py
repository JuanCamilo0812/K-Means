import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # para 3D plotting
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



df = pd.read_csv('penguins.csv', sep= ',')
print(df.head())

data = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
data = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']]
print(data.info())


# Extraer las columnas necesarias (ya lo hiciste)
data = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']]

# Crear la figura y el eje 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Graficar los puntos
ax.scatter(
    data['bill_length_mm'],
    data['bill_depth_mm'],
    data['flipper_length_mm'],
    c='skyblue',        # puedes usar color por categoría si quieres
    s=60,
    edgecolor='k'
)

# Etiquetas de ejes
ax.set_xlabel('Bill Length (mm)')
ax.set_ylabel('Bill Depth (mm)')
ax.set_zlabel('Flipper Length (mm)')
ax.set_title('Gráfico 3D de características de pingüinos')

# plt.show()


# 1. Extraer las variables numéricas
data = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']].dropna()

# 2. Escalar las variables (muy importante para K-means)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 3. Entrenar el modelo K-means con k clusters (por ejemplo, k=3)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=100)
kmeans.fit(data_scaled)

# 4. Obtener las etiquetas asignadas
labels = kmeans.labels_

# 5. Agregar etiquetas al dataframe original
data_with_labels = data.copy()
data_with_labels['Cluster'] = labels

# 6. Visualización 3D de los clusters
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Colores para los clusters
colors = ['red', 'green', 'blue']

for i in range(k):
    cluster_data = data_with_labels[data_with_labels['Cluster'] == i]
    ax.scatter(
        cluster_data['bill_length_mm'],
        cluster_data['bill_depth_mm'],
        cluster_data['flipper_length_mm'],
        c=colors[i],
        label=f'Cluster {i}',
        s=60,
        edgecolor='k',
        alpha=0.7
    )

# Etiquetas
ax.set_xlabel('Bill Length (mm)')
ax.set_ylabel('Bill Depth (mm)')
ax.set_zlabel('Flipper Length (mm)')
ax.set_title(f'K-means con k={k}')
ax.legend()
plt.show()


