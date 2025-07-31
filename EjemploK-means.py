import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # para 3D plotting
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns


df = pd.read_csv('penguins.csv', sep= ',').dropna()
print(df.head())

data = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']]
print(data.info())


# Crear la figura y el eje 3D
fig1 = plt.figure(figsize=(10, 7))
ax1 = fig1.add_subplot(111, projection='3d')

# Graficar los puntos
ax1.scatter(
    data['bill_length_mm'],
    data['bill_depth_mm'],
    data['flipper_length_mm'],
    c='skyblue',        # puedes usar color por categoría si quieres
    s=60,
    edgecolor='k'
)

# Etiquetas de ejes
ax1.set_xlabel('Bill Length (mm)')
ax1.set_ylabel('Bill Depth (mm)')
ax1.set_zlabel('Flipper Length (mm)')
ax1.set_title('Gráfico 3D de características de pingüinos')

plt.show()


# 1. Extraer las variables numéricas
#data = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']]

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
fig2 = plt.figure(figsize=(10, 7))
ax2 = fig2.add_subplot(111, projection='3d')

# Colores para los clusters
colors = ['red', 'green', 'blue']

for i in range(k):
    cluster_data = data_with_labels[data_with_labels['Cluster'] == i]
    ax2.scatter(
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
ax2.set_xlabel('Bill Length (mm)')
ax2.set_ylabel('Bill Depth (mm)')
ax2.set_zlabel('Flipper Length (mm)')
ax2.set_title(f'K-means con k={k}')
ax2.legend()
plt.show()

data_with_species = data_with_labels.copy()
data_with_species['Species'] = df['species']

# 1. Crear tabla de conteo cruzado (Species vs Cluster)
cluster_counts = data_with_species.groupby(['Species', 'Cluster']).size().reset_index(name='Count')
print(cluster_counts)

# 2. Graficar barras agrupadas con seaborn
fig3 = plt.figure(figsize=(10, 6))
sns.barplot(
    data=cluster_counts,
    x='Species',
    y='Count',
    hue='Cluster',
    palette='Set2',
    edgecolor='black'
)
plt.show()

# 2. Calcular los límites (mínimo y máximo) por variable y cluster
limits = data_with_labels.groupby('Cluster').agg(['min', 'max'])
print(limits)

Correlacion_cluster_Specie = ['Adelie', 'Gentoo', 'Chinstrap']


#"""

###########################    streamlit   ##########################
#####################################################################

st.set_page_config(layout='centered', page_title='Clusters K-Means', page_icon=(" :pingüino:"))
st.markdown("<h1 style='text-align: center;'>Clasificación de pingüinos con KMeans </h1>",
            unsafe_allow_html=True)
st.image("ref_penguin.png",width=800)

# Secciones 
steps = st.tabs(['Informacion', 'Data', 'Gráfica'])
with steps[0]:
    st.image("k1.jpeg",width=1200)
    st.image("k2.jpeg",width=1200)
    st.image("k3.jpeg",width=1200)
    st.image("k4.jpeg",width=1200)
    st.image("k5.jpeg",width=1200)

with steps[1]:
    st.header('Datos Originales')
    st.dataframe(df.head(8))
    st.header('Datos alimentados')
    st.dataframe(data.head(8))
    st.header('Datos Escalados')
    st.dataframe(pd.DataFrame(data_scaled).head(8))

with steps[2]:
    st.header('Gráfica de datos Originales')
    st.pyplot(fig1)
    st.header('Cluster generados')
    st.pyplot(fig2)
    st.header('Comparación y errores')
    st.pyplot(fig3)
    

#"""

