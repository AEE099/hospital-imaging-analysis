from json import encoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pip
import seaborn as sns
import streamlit as st
import plotly.express as px
import warnings
#train the model
from sklearn.model_selection import train_test_split ,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from pickle import NONE
from math import log2
import joblib
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')


 #Importando el archivo
df=pd.read_csv("DA-INCART-IMAGENES-JUN2015-JUN2023.csv", encoding='latin1')


        # Interfaz Streamlit: Titulos y Descripciones
st.set_page_config(page_title="Servicios de imagenes", layout="wide")
        # Agregar un logo al Streamlit
st.sidebar.image("logo.jpg", use_container_width=True)
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

        #Reemplazando valores

df.rename(columns={'A¤o': 'Ano'}, inplace=True)
df.rename(columns={"Servicio en 67A" : "Tipo de Servicio"}, inplace=True)
df.rename (columns={"Sin dato" : "Sin datos del seguro"} , inplace=True)
df.rename(columns={"Sin dato.1" : "Sin datos de la nacionalidad"} , inplace=True)
df.rename(columns={"Otros.1" : "Otra nacionalidad"} , inplace=True)

        #Corregir spelling y palabras incorrectas
from dataclasses import replace
df ["Tipo de Servicio"] = df["Tipo de Servicio"].str.replace ("¡" ,  "í", regex=False)
df ["Tipo de Servicio"] = df["Tipo de Servicio"].str.replace ("Endoscopías G\xa0strica" ,  "Endoscopías Gastrica", regex=False)
df ["Tipo de Servicio"] = df["Tipo de Servicio"].str.replace ("Resonancia Magn\x82tica" ,  "Resonancia Magnetica", regex=False)

        #unificacion 
df['sexo'] = df.apply( lambda row: 'Masculino' if row['Masculino'] >= 1 else 'Femenino', axis=1)
df["Nacionalidad"] = df.apply(lambda row: "RD" if row ["RD"] >= 1 else "Haiti" if row["Haiti"] >= 1 else "Otra nacionalidad" if row["Otra nacionalidad"] ==1 else "Sin datos de la nacionalidad", axis=1)
df["Seguro"] = df.apply(lambda row: "Senasa" if row ["Senasa"] >= 1 else "Otros" if row["Otros"] >= 1 else "Sin datos del seguro", axis=1)





menu = st.sidebar.radio("Selecciona una sección", ["📊 Gráficos", "🤖 Predicciones"])






if menu == "📊 Gráficos":

 try:
       
        st.title("Dashboard interactivo de Estudios/Servicios de imagenes")
        st.write("Estaremos viendo graficos sobre la data explorada.")







        #Definiendo barra de navegacion lateral 

        st.sidebar.header("Filtros: ")

        sexo_seleccionado = st.sidebar.multiselect("Sexo:", options=df['sexo'].unique())

        Ano = st.sidebar.multiselect("Ano: ", df["Ano"].unique())
        Mes = st.sidebar.multiselect("Mes: ", df["Mes"].unique())
        Nacionalidad = st.sidebar.multiselect("Nacionalidad: ", options=df["Nacionalidad"].unique())
        Tipo_de_Servicio = st.sidebar.multiselect("Tipo de Servicio: ", df["Tipo de Servicio"].unique())
        Seguro = st.sidebar.multiselect("Seguro: ", options= df["Seguro"].unique())
        Nombre_Servicio = st.sidebar.multiselect("Nombre Servicio: ", df["Nombre Servicio"].unique())


        #Filtrado

        # Initialize df_filtered with the original dataframe
        df_filtered = df

        if sexo_seleccionado:
            df_filtered=  df[df['sexo'].isin(sexo_seleccionado)]
        if Ano:
            df_filtered = df_filtered[df_filtered["Ano"].isin(Ano)]
        if Mes:
            df_filtered = df_filtered[df_filtered["Mes"].isin(Mes)]
        if Nacionalidad:
            df_filtered = df_filtered[df_filtered["Nacionalidad"].isin(Nacionalidad)]
        if Tipo_de_Servicio:
            df_filtered = df_filtered[df_filtered["Tipo de Servicio"].isin(Tipo_de_Servicio)]
        if Nombre_Servicio:
            df_filtered = df_filtered[df_filtered["Nombre Servicio"].isin(Nombre_Servicio)]
        if Seguro:
            df_filtered = df_filtered[df_filtered["Seguro"].isin(Seguro)]



            



        # Mostrar los datos filtrados
        st.write("Datos filtrados:")
        st.dataframe(df_filtered)


        # Calcular la cantidad total de estudios realizados
        total_estudios = df_filtered['Cantidad'].sum()

        # Calcular la cantidad total de estudios realizados por hombres y mujeres (Filtrado)
        genero_totales_filtrado = df_filtered[['Masculino', 'Femenino']].sum()

        # Mostrar la cantidad total de estudios en Streamlit (Filtrado)
        st.write(f"### Cantidad total de estudios realizados (Filtrado): {total_estudios}")

        # Crear el gráfico de barras para mostrar la cantidad de estudios por género (Filtrado)
        plt.figure(figsize=(8, 6))
        plt.bar(genero_totales_filtrado.index, genero_totales_filtrado.values, color=['blue', 'pink'])
        plt.title('Cantidad de Estudios Realizados por Género (Filtrado)')
        plt.xlabel('Género')
        plt.ylabel('Cantidad de Estudios')
        plt.tight_layout()
        st.pyplot(plt.gcf())



        # Graficos dependientes de los filtros aplicados
        mapa = ['Masculino', 'Femenino']

        # Pie Chart: Distribucion por Sexo (Filtrado)
        st.write("### Distribucion por Sexo (Filtrado)")
        if not df_filtered.empty:
            sexos_filtrados = df_filtered[['Masculino', 'Femenino']].sum()
            plt.figure()
            plt.pie(sexos_filtrados, labels=mapa, autopct='%1.1f%%', startangle=90)
            plt.title('Distribucion por Sexo (Filtrado)')
            st.pyplot(plt.gcf())
        else:
            st.write("No hay datos disponibles para los filtros seleccionados.")

        # Line Chart: Distribucion de Sexo por Mes (Filtrado)
        st.write("### Line Chart: Distribucion de Sexo por Mes (Filtrado)")
        if not df_filtered.empty:
            sexo_mes_filtrado = df_filtered.groupby('Mes')[['Masculino', 'Femenino']].sum().reset_index()
            plt.figure(figsize=(12, 6))
            plt.plot(sexo_mes_filtrado['Mes'], sexo_mes_filtrado['Masculino'], label='Masculino', marker='o', color='blue')
            plt.plot(sexo_mes_filtrado['Mes'], sexo_mes_filtrado['Femenino'], label='Femenino', marker='o', color='pink')
            plt.title('Distribucion de Sexo por Mes (Filtrado)')
            plt.xlabel('Mes')
            plt.ylabel('Cantidad')
            plt.xticks(range(1, 13))
            plt.legend(title='Sexo')
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(plt.gcf())
        else:
            st.write("No hay datos disponibles para los filtros seleccionados.")



        # Crear un gráfico de líneas para la distribución de sexo por año (Filtrado)
        st.write("### Distribución de Sexo por Año (Filtrado)")
        if not df_filtered.empty:
            sexo_por_ano_filtrado = df_filtered.groupby('Ano')[['Masculino', 'Femenino']].sum().reset_index()
            plt.figure(figsize=(12, 6))
            plt.plot(sexo_por_ano_filtrado['Ano'], sexo_por_ano_filtrado['Masculino'], label='Masculino', marker='o', color='blue')
            plt.plot(sexo_por_ano_filtrado['Ano'], sexo_por_ano_filtrado['Femenino'], label='Femenino', marker='o', color='pink')
            plt.title('Distribución de Sexo por Año (Filtrado)')
            plt.xlabel('Año')
            plt.ylabel('Cantidad')
            plt.xticks(sexo_por_ano_filtrado['Ano'], rotation=45)  # Ajustar las etiquetas del eje X
            plt.legend(title='Sexo')
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(plt.gcf())
        else:
            st.write("No hay datos disponibles para los filtros seleccionados.")
            

        # Bar Chart: Distribución por Nacionalidad (Filtrado)
        st.write("### Distribución por Nacionalidad (Filtrado)")
        if not df_filtered.empty:
            nacionalidad_filtrada = df_filtered[['RD', 'Haiti', "Otra nacionalidad", "Sin datos de la nacionalidad"]].sum()
            mapa = ['RD', 'Haiti', "Otra nacionalidad", "Sin datos"]
            colors = ['blue', 'green', 'orange', 'red']
            plt.figure(figsize=(10, 6))
            bars = plt.bar(mapa, nacionalidad_filtrada, color=colors)
            plt.title('Distribución por Nacionalidad (Filtrado)')
            plt.xlabel('Nacionalidad')
            plt.ylabel('Cantidad')
            plt.tight_layout()
            st.pyplot(plt.gcf())
        else:
            st.write("No hay datos disponibles para los filtros seleccionados.")


        # Pie Chart: Distribución por Seguro (Filtrado)
        st.write("### Distribución por Seguro (Filtrado)")
        if not df_filtered.empty:
            seguro_filtrado = df_filtered[['Senasa', "Otros"]].sum()
            mapa = ['Senasa', "Otros"]
            plt.figure()
            plt.pie(seguro_filtrado, labels=mapa, autopct='%1.1f%%', startangle=90)
            plt.title('Distribución por Seguro (Filtrado)')
            st.pyplot(plt.gcf())
        else:
            st.write("No hay datos disponibles para los filtros seleccionados.")



        # Agrupar por tipo de servicio y sumar las columnas de seguro
        st.write("### Distribución de Seguros por Tipo de Estudio (Filtrado)")
        if not df_filtered.empty:
            seguro_vs_tipo = df_filtered.groupby('Tipo de Servicio')[['Senasa', 'Otros']].sum()
                
            # Crear el gráfico de barras apiladas
            plt.figure(figsize=(12, 6))
            seguro_vs_tipo.plot(kind='bar', stacked=True, color=['red', 'green'], ax=plt.gca())
            plt.title('Distribución de Seguros por Tipo de Estudio (Filtrado)')
            plt.xlabel('Tipo de Servicio')
            plt.ylabel('Cantidad')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Seguro')
            plt.tight_layout()
            st.pyplot(plt.gcf())
        else:
            st.write("No hay datos disponibles para los filtros seleccionados.")
            
            
        # Distribución de Seguros por Mes
        st.write("### Distribución de Seguros por Mes")
        if not df_filtered.empty:
            mes_vs_seguro = df_filtered.groupby('Mes')[['Senasa', 'Otros']].sum()
                
            plt.figure(figsize=(12, 6))
            mes_vs_seguro.plot(kind='bar', color=['blue', 'orange'], ax=plt.gca())
            plt.title('Distribución de Seguros por Mes (Filtrado)')
            plt.xlabel('Mes')
            plt.ylabel('Cantidad')
            plt.xticks(rotation=0)
            plt.legend(title='Seguro')
            plt.tight_layout()
            st.pyplot(plt.gcf())
        else:
            st.write("No hay datos disponibles para los filtros seleccionados.")
                
                
        # Distribución de Seguros por Año (2016-2022)
        st.write("### Distribución de Seguros por Año (2016-2022)")
        if not df_filtered.empty:
            ano_vs_seguro = df_filtered[(df_filtered['Ano'] >= 2016) & (df_filtered['Ano'] <= 2022)].groupby('Ano')[['Senasa', 'Otros']].sum()
                    
            # Crear el gráfico de barras separadas para Senasa y Otros
            plt.figure(figsize=(12, 6))
            ano_vs_seguro.plot(kind='bar', color=['blue', 'orange'], ax=plt.gca())
            plt.title('Distribución de Seguros por Año (2016-2022)')
            plt.xlabel('Año')
            plt.ylabel('Cantidad')
            plt.xticks(rotation=0)
            plt.legend(title='Seguro')
            plt.tight_layout()
            st.pyplot(plt.gcf())
        else:
            st.write("No hay datos disponibles para los filtros seleccionados.")
            

        # Agrupar por tipo de servicio y sumar las cantidades (Filtrado)
        st.write("### Cantidad de estudios por tipo de servicio (Filtrado)")
        if not df_filtered.empty:
            tipo_estudios_filtrado = df_filtered.groupby('Tipo de Servicio')['Cantidad'].sum().reset_index()
            tipo_estudios_filtrado = tipo_estudios_filtrado.sort_values(by='Cantidad', ascending=False).head(20)
                
            # Crear el gráfico de barras
            plt.figure(figsize=(10, 6))
            plt.bar(tipo_estudios_filtrado['Tipo de Servicio'], tipo_estudios_filtrado['Cantidad'], color='skyblue')
            plt.title('Cantidad de estudios por tipo de servicio (Filtrado)')
            plt.xlabel('Tipo de Servicio')
            plt.ylabel('Cantidad')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(plt.gcf())
        else:
            st.write("No hay datos disponibles para los filtros seleccionados.")



        # Agrupar por tipo de servicio y sumar las cantidades (Filtrado)
        st.write("### Top 20 estudios por Nombre Servicio (Filtrado)")
        if not df_filtered.empty:
            nombre_estudios_filtrado = df_filtered.groupby('Nombre Servicio')['Cantidad'].sum().reset_index()
            top_estudios_filtrado = nombre_estudios_filtrado.sort_values(by='Cantidad', ascending=False).head(20)
                
                # Crear el gráfico de barras
            plt.figure(figsize=(12, 6))
            plt.bar(top_estudios_filtrado['Nombre Servicio'], top_estudios_filtrado['Cantidad'], color='skyblue')
            plt.title('Top 20 estudios por Nombre Servicio (Filtrado)')
            plt.xlabel('Nombre Servicio')
            plt.ylabel('Cantidad')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(plt.gcf())
        else:
            st.write("No hay datos disponibles para los filtros seleccionados.")
            
            
        # Scatter plot: Masculino vs Femenino por Tipo de Servicio
        st.write("### Scatter plot Masculino vs Femenino por Tipo de Servicio")
        if not df_filtered.empty:
            palette = sns.color_palette("tab20", len(df['Tipo de Servicio'].unique()))
            plt.figure(figsize=(12, 6))
            sns.scatterplot(data=df_filtered, x='Masculino', y='Femenino', hue='Tipo de Servicio', alpha=0.9, s=100, palette=palette)
            plt.title('Scatter plot Masculino vs Femenino por Tipo de Servicio')
            plt.legend(title='Tipo de Servicio', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(plt.gcf())
        else:
            st.write("No hay datos disponibles para los filtros seleccionados.")
            
            
        # Scatter plot: Masculino vs Femenino por Nombre de Servicio (Top 20)
        st.write("### Scatter plot Masculino vs Femenino por Nombre de Servicio (Top 20)")
        if not df_filtered.empty:
            # Filter the top 20 Nombre Servicio
            top_20_servicios = df_filtered[df_filtered['Nombre Servicio'].isin(
            df_filtered.groupby('Nombre Servicio')['Cantidad'].sum().nlargest(20).index)]
                
            # Define a color palette for the top 20 Nombre Servicio
            palette = sns.color_palette("tab20", len(top_20_servicios['Nombre Servicio'].unique()))
                
            # Create the scatterplot
            plt.figure(figsize=(12, 6))
            sns.scatterplot(data=top_20_servicios, x='Masculino', y='Femenino', hue='Nombre Servicio', alpha=0.9, s=100, palette=palette)
            plt.title('Scatter plot Masculino vs Femenino por Nombre de Servicio (Top 20)')
            plt.legend(title='Nombre Servicio', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(plt.gcf())
        else:
            st.write("No hay datos disponibles para los filtros seleccionados.")


        # Tendencias Anuales y Mensuales
        st.write("### Tendencias Anuales y Mensuales")
        if not df_filtered.empty:
            # Agrupar por año y sumar las cantidades (Filtrado)
            tendencia_anual_filtrada = df_filtered.groupby('Ano')['Cantidad'].sum().reset_index()

            # Agrupar por mes y sumar las cantidades (Filtrado)
            tendencia_mensual_filtrada = df_filtered.groupby('Mes')['Cantidad'].sum().reset_index()

            # Crear gráficos de líneas
            plt.figure(figsize=(14, 6))

            # Gráfico de tendencias anuales
            plt.subplot(1, 2, 1)
            plt.plot(tendencia_anual_filtrada['Ano'], tendencia_anual_filtrada['Cantidad'], marker='o', color='blue')
            plt.title('Tendencia Anual (Filtrado)')
            plt.xlabel('Año')
            plt.ylabel('Cantidad')
            plt.grid(True)

            # Gráfico de tendencias mensuales
            plt.subplot(1, 2, 2)
            plt.plot(tendencia_mensual_filtrada['Mes'], tendencia_mensual_filtrada['Cantidad'], marker='o', color='green')
            plt.title('Tendencia Mensual (Filtrado)')
            plt.xlabel('Mes')
            plt.ylabel('Cantidad')
            plt.xticks(range(1, 13))
            plt.grid(True)

            plt.tight_layout()
            st.pyplot(plt.gcf())
        else:
            st.write("No hay datos disponibles para los filtros seleccionados.")
            
        
        # Gráfico de líneas para la evolución de servicios por mes y tipo de servicio (Filtrado)
        st.write("### Evolución de Servicios por Mes y Tipo de Servicio (Filtrado)")
        if not df_filtered.empty:
        # Agrupar por mes y tipo de servicio, y sumar las cantidades
            tendencia_mensual_filtrada = df_filtered.groupby(['Mes', 'Tipo de Servicio'])['Cantidad'].sum().unstack()

        # Crear el gráfico de líneas para cada tipo de servicio
        plt.figure(figsize=(20, 10))
        colors = ['blue', 'green', 'orange', 'purple', 'red', 'brown', 'pink', 'cyan', 'magenta', 'yellow', 'lime', 'teal', 'gold']
        for i, column in enumerate(tendencia_mensual_filtrada.columns):
            plt.plot(tendencia_mensual_filtrada.index, tendencia_mensual_filtrada[column], marker='o', label=column, color=colors[i % len(colors)])
            plt.title('Evolución de Servicios por Mes y Tipo de Servicio (Filtrado)')
            plt.xlabel('Mes')
            plt.ylabel('Cantidad de Servicios')
            plt.legend(title='Tipo de Servicio', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(plt.gcf())
        else:
            st.write("No hay datos disponibles para los filtros seleccionados.")
        

        # Gráfico de líneas para la evolución de servicios por año y tipo de servicio (Filtrado)
        st.write("### Evolución de Servicios por Año y Tipo de Servicio (Filtrado)")
        if not df_filtered.empty:
        # Agrupar por año y tipo de servicio, y sumar las cantidades
            tendencia_ano_servicio_filtrada = df_filtered.groupby(['Ano', 'Tipo de Servicio'])['Cantidad'].sum().unstack()

        # Crear el gráfico de líneas para cada tipo de servicio
        plt.figure(figsize=(20, 10))
        palette = sns.color_palette("tab10", n_colors=len(tendencia_ano_servicio_filtrada.columns))

        for i, column in enumerate(tendencia_ano_servicio_filtrada.columns):
            plt.plot(tendencia_ano_servicio_filtrada.index, 
                     tendencia_ano_servicio_filtrada[column], 
                     marker='o', label=column, color=palette[i] )

            plt.title('Evolución de Servicios por Año y Tipo de Servicio (Filtrado)')
            plt.xlabel('Año')
            plt.ylabel('Cantidad de Servicios')
            plt.legend(title='Tipo de Servicio', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(plt.gcf())
        else:
            st.write("No hay datos disponibles para los filtros seleccionados.")
 except FileNotFoundError:
        st.error("⚠️ No se encontró el archivo datos_estudios.csv")
    
    
elif menu == "🤖 Predicciones":
    
    #VAMOS A PRESENTAR UN MODELO DE PREDICCION 
    import joblib
    import pandas as pd
    import streamlit as st

    # Función para predecir cantidad
    def predecir_cantidad(nuevos_datos_df):
        # Cargar modelo y encoders
        model = joblib.load('best_rf_model.pkl')
        encoder_tipo_servicio = joblib.load('encoder.pkl')
        encoder_nombre_servicio = joblib.load('enconder_Servicio.pkl')    

        # Codificar nuevos datos
        tipo_servicio_encoded = encoder_tipo_servicio.transform(nuevos_datos_df[["Tipo de Servicio"]])
        nombre_servicio_encoded = encoder_nombre_servicio.transform(nuevos_datos_df[["Nombre Servicio"]])

        # Crear DataFrames codificados
        tipo_servicio_df = pd.DataFrame(tipo_servicio_encoded, columns=encoder_tipo_servicio.get_feature_names_out(["Tipo de Servicio"]))
        nombre_servicio_df = pd.DataFrame(nombre_servicio_encoded, columns=encoder_nombre_servicio.get_feature_names_out(["Nombre Servicio"]))

        # Combinar con el resto de las columnas
        X_nuevo = pd.concat([
            nuevos_datos_df.drop(columns=["Tipo de Servicio", "Nombre Servicio"]),
            tipo_servicio_df,
            nombre_servicio_df
        ], axis=1)

        # Predecir
        predicciones = model.predict(X_nuevo)
        return predicciones

    # Interfaz de Streamlit
    st.title("Predicción de Cantidad de Estudios")
    st.write("Por favor, seleccione los datos para predecir la cantidad de estudios.")


    # Entradas del usuario
    Ano = st.selectbox("Ano", list(range(2015, 2028)))
    Mes = st.selectbox("Mes", list(range(1, 13)))
    Tipo_Servicio = st.selectbox("Tipo de Servicio", ['Sonografía', 'Mamografía', 'Radiografía', 'Sonomamografía',
        'Otros servicios de im\xa0genes', 'Tomografía', 'Cistoscopía',
        'Colposcopía', 'Rectocismoidoscopia', 'Endoscopías Gastrica',
        'Endoscopías Aparato Respiratorio', 'Espirometría', 'Resonancia Magnetica'])

    Nombre_servicio= st.selectbox ("Nombre Servicio", ['BIOPSIA SONODIRIGIDA DE PROSTATA',
        'BIOPSIA SONODIRIGIDA DE TIROIDES', 'MAMOGRAFIA BILATERAL',
        'RX-TORAX AP', 'RX-TORAX PA', 'SONOGRAFIA ABDOMINAL',
        'SONOGRAFIA DE MAMAS', 'SONOGRAFIA DE TIROIDES',
        'SONOGRAFIA PELVICA', 'SONOGRAFIA TRANSVAGINAL',
        'BIOPSIA DE HUESO ( NO INCLUYE KIT CON AGUJA JAMSHIDI NI INMUNOHISTOQU?',
        'BIOPSIA POR TOMOGRAFIA ',
        'BIOPSIA PULMON DIRIGIDA POR SONOGRAFIA',
        'BIOPSIA SONODIRIGIDA DE CUELLO (GANGLIOS Y TIROIDES Y MASAS)',
        'BIOPSIA SONODIRIGIDA DE MAMAS',
        'BIOPSIA SONODIRIGIDA DE MAMAS CON TRUCUT + AGUJA',
        'BIOPSIA SONODIRIGIDA PARTES BLANDAS (CON TRUCUT)',
        'BIOPSIA TOMODIRIGIDA HEPATICA', 'CITOSCOPIA ',
        'COLPOSCOPIA CON BIOPSIA', 'MAMOGRAFIA UNILATERAL',
        'MARCAJE 6 PLACAS', 'RX-COLUMNA LUMBO-SACRA AP Y LATERAL',
        'RX-FEMUR (MUSLO) INCLUYENDO UNA ARTICULACION AP Y LATERAL',
        'RX-HOMBRO AP', 'RX-MANO (O AMBAS) 2 POSICIONES (AP Y OBLICUA)',
        'RX-MANO (O AMBAS) TRES POSICIONES (AP, LATERAL Y OBLICUA)',
        'RX-MU¥ECA UNA POSICION',
        'RX-TORAX 2 POSICIONES (PA Y 1 OBLICUA / PA Y LATERAL)',
        'SONOGRAFIA DE AXILA', 'SONOGRAFIA DE CUELLO',
        'SONOGRAFIA ESCROTAL (TESTICULO)', 'SONOGRAFIA PROSTATICA',
        'SONOGRAFIA RENAL', 'TOMA DE MUESTRA PAPANICOLAOU',
        'TOMOGRAFIA ABDOMEN', 'TOMOGRAFIA ABDOMEN Y PELVIS',
        'TOMOGRAFIA COLUMNA LUMBAR', 'TOMOGRAFIA CRANEO',
        'TOMOGRAFIA CUELLO (LARINGE O PARTES BLANDAS)',
        'TOMOGRAFIA PELVIS', 'TOMOGRAFIA SENOS PARA NASALES',
        'TOMOGRAFIA TORAX', 'TOMOGRAFIA TORAX Y ABDOMEN',
        'ULTRASONOGRAF?A DE PROSTATA TRANSRECTAL',
        'ULTRASONOGRAF?A DE TEJIDOS BLANDOS EN LAS EXTREMIDADES SUPERIORES CON ',
        'ULTRASONOGRAFIA DIAGNOSTICA DE MAMA, CON TRANSDUCTOR DE 7 MHZ O MAS',
        'UTRASONOGRAFIA PELVICA GINECOLOGICA TRANSVAGINAL',
        'COMPRESION FOCAL UNILATERAL',
        'INMUNOHISTOQUIMICA TEJIDOS SOLIDOS',
        'RX-ABDOMEN SIMPLE 2 POSICIONES', 'RX-ABDOMEN SIMPLE AP',
        'RX-CRANEO AP Y LATERAL', 'SONOGRAFIA DE HOMBRO',
        'TOMOGRAFIA EXTREMIDADES INFERIORES',
        'ULTRASONOGRAF?A DE TEJIDOS BLANDOS EN LAS EXTREMIDADES INFERIORES CON ',
        'ULTRASONOGRAFIA DIAGNOSTICA DE TIROIDES, CON TRANSDUCTOR DE 7 MHZ O MA',
        'BIOPSIA SONODIRIGIDA CUELLO (GANGLIOS Y TIROIDES Y MASAS)',
        'BIOPSIA-ASPIRADO MEDULA ?SEA',
        'RX-TOBILLO 2 POSICIONES (AP Y LATERAL)',
        'TOMOGRAFIA COLUMNA CERVICAL',
        'ULTRASONOGRAFIAS DIAGNOSTICAS DE TEJIDOS BLANDOS DE PARED ABDOMINAL Y ',
        'ESTUDIO DE BIOPSIA DE APENDICE', 'MARCAJE 12 PLACAS',
        'ESTUDIO DE CITOLOGIA DE TIROIDES', 'TOMOGRAFIA COLUMNA DORSAL',
        'SONOGRAFIA DE GLUTEO', 'TOMOGRAFIA CADERA', 'TOMOGRAFIA OIDOS',
        'RX-CADERA COMPARATIVA',
        'RX-CERVICAL COMPLETA, INCLUYENDO ESTUDIOS EN FLEXION Y EXTENS',
        'RX-COLUMNA LUMBOSACRA', 'RX-CR?NEO SIMPLE', 'RX-PELVIS AP',
        'RX-RODILLA (AP Y LATERAL)', 'RX-SENOS PARANASALES',
        'MAGNIFICACION BILATERAL',
        'RX-ANTEBRAZO INCLUYENDO UNA ARTICULACION (AP Y LATERAL)',
        'RX-CODO AP Y LATERAL', 'RX-COLUMNA CERVICAL AP',
        'RX-COLUMNA CERVICAL AP Y LATERAL', 'RX-COLUMNA DORSAL AP',
        'RX-COLUMNA DORSAL AP, LATERLA Y 1 OBLICUA',
        'RX-COLUMNA DORSO LUMBAR AP Y LATERAL PARA ESCOLIOSIS',
        'RX-COLUMNA DORSOLUMBAR',
        'RX-COLUMNA LUMBO-SACRA AP, LATERAL Y UNA OBLICUA',
        'RX-HOMBRO AP Y OTRA PROYECCION',
        'RX-HUMERO AP Y LATERAL, INCLUYENDO UNA ARTICULACION',
        'RX-MU¥ECA AP Y LATERAL', 'RX-PIE 2 POSICIONES',
        'RX-PIE UNA POSICION', 'RX-CARA (CADA POSICION)',
        'RX-DEDOS EN MANO', 'RX-FEMUR (MUSLO) UNA POSICION',
        'RX-PELVIS AP Y LATERAL', 'RX-TIBIA Y PERONE (AP Y LATERAL)',
        'ASPIRACION DE QUISTE', 'RX-BASE DE CRANEO',
        'RX-COLUMNA CERVICAL AP LAT Y OBLICUA',
        'RX-COLUMNA DORSAL 2 POSICIONES',
        'RX-MU¥ECA AP, LATERAL Y OBLICUA',
        'VIDEO COLONOSCOP?A DIAGN?STICA', 'GASTROSTOM?A ENDOSC?PICA',
        'RX-CLAVICULA AP',
        'UTRASONOGRAFIA PELVICA GINECOLOGICA TRANSABDOMINAL',
        'BIOPSIA CERRADA (ENDOSCOPICA) DE ESTOMAGO SOD',
        'BIOPSIA CERRADA (ENDOSCOPICA) DE RECTO O SIGMOIDE SOD',
        'RX-COLUMNA LUMBO-SACRA COMPLETA, MINIMO 4 POSICIONES',
        'RX-CRANEO AP Y LATERAL Y OTRA PROYECCION',
        'RX-CUELLO PARA TEJIDOS BLANDOS',
        'ULTRASONOGRAF?A TESTICULAR CON ANALISIS DOPPLER',
        'VIDEO COLONOSCOP?A DIAGN?STICA (CON BIOPSIA)',
        'RX-MAXILAR SUPERIOR', 'RX-TORAX OBLICUAS',
        'TOMOGRAFIAS SILLA TURCA',
        'VIDEO COLONOSCOP?A TERAP\x90UTICA/ POLIPECTOM?A',
        'DRENAJE PERCUTANEO DE ABSCESOS Y COLECCIONES LIQUIDA',
        'MARCAJE 9 PLACAS', 'MARCAJE POR SONOGRAFIA',
        'NASOFIBROLARINGOSCOPIA', 'RX-CLAVICULA AP Y LATERAL',
        'RX-COLUMNA CERVICAL UN MINIMO DE 4 POSICIONES', 'RX-ESCAPULA',
        'SONOGRAFIA PAROTIDAS',
        'ESTUDIO DE BIOPSIA DE ENDOMETRIO (LEGRADO)',
        'ESTUDIO DE COLORACION BASICA EN CITOLOGIA CERVICOVAGINAL (PAPANICOLAU)',
        'ESTUDIO MOLECULAR ADENOCARCINOMA DE PULMON(EGFR/ALK) ESTADOS UNIDOS',
        'RX-ARTICULACION SACRO-ILIACA AP Y OBLICUA',
        'RX-MEDICION DE MIEMBROS INFERIORES, ESTUDIO DE PIE PLANO (PIES CON APO',
        'DRENAJE ', 'TOMOGRAFIA ORBITA',
        'ULTRASONOGRAFIA DIAGNOSTICA DE GLANDULAS SALIVALES CON TRANSDUCTOR DE ',
        'CONIZACION ASA DIAMETRICA EN CONSULTORIO',
        'RX-PELVIS AP Y ABDUCCION (RANA)',
        'BIOPSIA DE ENDOMETRIO Y LESION ENDOMETRIAL POR HISTEROSCOPIA',
        'GASTROSCOPIA A TRAVES DE ESTOMA ARTIFICIAL SOD',
        'POLIPECTOM?A ENDOSCOPICA DE RECTO SOD',
        'RECTOSIGMOIDOSCOP?A CA CU',
        'ESTUDIO DE BIOPSIA DE HUESO, RESECCION PEQUE¥A',
        'LIGADURA ENDOSCOPICA DE VARICES ESOFAGICAS', 'POLIPECTOM?A',
        'ESTUDIO DE MARCADORES MOLECULARES LEUCEMIAS, LINFOMAS',
        'REVISION DE BIOPSIA EXTERNA', 'ESPIROMETRIA SOD',
        'INMUNOHISTOQUIMICA TEJIDOS L?QUIDOS (MEDULA Y SANGRE)',
        'RX-COSTILLA AP Y OBLICUA',
        'RX-PIE ESTUDIO DE RUTINA COMPLETO MINIMO DE 4 POS',
        'TERAPIA DESCONGESTIVA COMPLEJA',
        'RX-HOMBRO AP Y ROTACION INT Y EXT (3PROYECCIONES)',
        'RX-HUESO CALCANEO (TALON) MINIMO DE DOS POSICIONES',
        'BIOPSIA DE ESOFAGO CERRADA (ENDOSCOPICA) SOD',
        'ESTUDIO DE MARCADORES MOLECULARES, OTROS',
        'RX-SENOS NASALES COMPLETO (POSICIONES: WATER, CADWELL, LATERAL)',
        'NEFROSTOMIA CERRADA', 'RX-COLUMNA LUMBAR', 'PARACENTESIS',
        'VIDEO COLONOSCOP?A TERAP\x90UTICA',
        'TERAPIA FISICA BASICA MAS MODALIDAD CINETICA',
        'SONOGRAFIA DE PIERNA',
        'TERAPIA FISICA BASICA MAS MODALIDAD ELECTRICA',
        'TERAPIA FISICA BASICA MAS TRACCION CERVICAL O LUMBAR',
        'ECOCARDIOGRAMA', 'MRI VEJIGA URINARIA (PELVIS FEMENINA)',
        'MRI COLUMNA SACRA', 'MRI DE ABDOMEN', 'MRI DE MAMA',
        'POLIPECTOMIA ENDOSCOPICA DE ESOFAGO', 'MRI COLUMNA LUMBAR',
        'RX-COLUMNA UNION CERVICO DORSAL',
        'RX-CADERA Y PELVIS, INFANTES Y NI¥OS 2 POSICIONES'])

    # Crear DataFrame de entrada
    df_input = pd.DataFrame([{
        "Ano": Ano,
        "Mes": Mes,
        "Tipo de Servicio": Tipo_Servicio,
        "Nombre Servicio": Nombre_servicio
    }])

    # Predecir
    if st.button("Predecir cantidad"):
        try:
            resultado = predecir_cantidad(df_input)
            st.success(f"🔢 Cantidad estimada de : {int(resultado[0])}")
        except Exception as e:
            st.error(f"❌ Error al hacer la predicción: {e}")

