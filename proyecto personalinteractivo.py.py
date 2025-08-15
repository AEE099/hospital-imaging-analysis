import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


#Importando el archivo
try: 
    df = pd.read_csv("DA-INCART-IMAGENES-JUN2015-JUN2023.csv", encoding='latin1')
except FileNotFoundError:
    st.error("Error: Archivo no encontrado.")
    exit()

# Interfaz Streamlit: Titulos y Descripciones
st.set_page_config(page_title="DA-INCART-IMAGENES-JUN2015-JUN2023", layout="wide")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
st.title("Dashboard interactivo de Estudios/Servicios de imagenes")
st.write("Estaremos viendo graficos sobre la data explorada.")

#Reemplazando valores

df.rename(columns={'A¤o': 'Ano'}, inplace=True)
df.rename(columns={"Servicio en 67A" : "Tipo de Servicio"}, inplace=True)
df.rename (columns={"Sin dato" : "Sin datos del seguro"} , inplace=True)
df.rename(columns={"Sin dato.1" : "Sin datos de la nacionalidad"} , inplace=True)
df.rename(columns={"Otros.1" : "Otra nacionalidad"} , inplace=True)

#Definiendo barra de navegacion lateral

st.sidebar.header("Filtros: ")

sexos = st.sidebar.multiselect("Sexos: ", df["Masculino,Fenemino"].unique())
Ano = st.sidebar.multiselect("Ano: ", df["Ano"].unique())
Mes = st.sidebar.multiselect("Mes: ", df["Mes"].unique())
Nacionalidad = st.sidebar.multiselect("nacionalidad: ", df[['RD', 'Haiti', 'Otra nacionalidad', 'Sin datos de la nacionalidad']].unique())
Tipo_de_Servicio = st.sidebar.multiselect("Tipo de Servicio: ", df["Tipo de Servicio"].unique())
Seguro= st.sidebar.multiselect("Seguro: ", df["Senasa", "Otros","Sin datos del seguro"].unique())
Nombre_del_servicio= st.sidebar.multiselect("Nombre del servicio: ", df["Nombre del servicio"].unique())


#Filtrado

df_filtered = df
if sexos:
    df_filtered = df_filtered[df_filtered["Masculino,Fenemino"].isin(sexos)]
if Ano:
    df_filtered = df_filtered[df_filtered["Ano"].isin(Ano)]
if Mes:
    df_filtered = df_filtered[df_filtered["Mes"].isin(Mes)]
if Nacionalidad:
    df_filtered = df_filtered[df_filtered['RD', 'Haiti', 'Otra nacionalidad', 'Sin datos de la nacionalidad'].isin(Nacionalidad)]
if Tipo_de_Servicio:
    df_filtered = df_filtered[df_filtered["Tipo de Servicio"].isin(Tipo_de_Servicio)]
if Seguro:
    df_filtered = df_filtered[df_filtered["Senasa", "Otros","Sin datos del seguro"].isin(Seguro)]
if Nombre_del_servicio:
    df_filtered = df_filtered[df_filtered["Nombre del servicio"].isin(Nombre_del_servicio)]
    

# Mostrar los datos filtrados
st.write("Datos filtrados:")
st.dataframe(df_filtered)

#Scatter Plot: Relacion entre Edad y Colesterol
st.write("### grafico sexo")
# Calculate the total counts for Masculino and Femenino
sexos = df[['Masculino', 'Femenino']].sum()

# Map the labels for the pie chart
mapa = ['Masculino', 'Femenino']

# Plot the pie chart
plt.pie(sexos, labels=mapa, autopct='%1.1f%%', startangle=90)
plt.title('Distribucion por Sexo')
plt.show()

#Scatter Plot: relacion sexo con nacionalidad
st.write("### Scatter Plot: Distribucion de sexo por nacionalidad")
# Agrupar los datos por nacionalidad y sumar las columnas de Masculino y Femenino
sexo_vs_nacionalidad = df.groupby(['RD', 'Haiti', 'Otra nacionalidad', 'Sin datos de la nacionalidad'])[['Masculino', 'Femenino']].sum()

# Crear un gráfico de dispersión con colores asignados a cada nacionalidad
colors = {'RD': 'blue', 'Haiti': 'green', 'Otra nacionalidad': 'orange', 'Sin datos de la nacionalidad': 'red'}
plt.figure(figsize=(12, 6))
for nationality, color in colors.items():
    plt.scatter(sexo_vs_nacionalidad.index.get_level_values(nationality), 
                sexo_vs_nacionalidad['Masculino'], 
                label=f'Masculino - {nationality}', 
                color=color, 
                alpha=0.6)
    plt.scatter(sexo_vs_nacionalidad.index.get_level_values(nationality), 
                sexo_vs_nacionalidad['Femenino'], 
                label=f'Femenino - {nationality}', 
                color=color, 
                alpha=0.3)
plt.title('Distribución de Sexo por Nacionalidad')
plt.xlabel('Nacionalidad')
plt.ylabel('Cantidad')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sexo')
plt.tight_layout()
plt.show()

# To run this Streamlit app, execute the following command in your terminal:
# streamlit run 
