import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Cargar el modelo y objetos necesarios
model_Tree, model_Knn, model_NN, variables, min_max_scaler = pickle.load(open('modelo-reg-tree-knn-nn.pkl', 'rb'))

# Configuración de página
st.set_page_config(page_title="Predicción Compra Videojuegos", layout="centered")
# Título principal
st.title("🎮 Predicción Compra video juegos en tienda")

# --- Formulario de entrada ---
with st.form("formulario_prediccion"):
    st.header("🧾 Parámetros del usuario")

    edad = st.number_input("Edad del jugador", min_value=10, max_value=60, value=14)

    st.subheader("🎮 Tipo de videojuego (marca los que aplican):")
    col1, col2, col3 = st.columns(3)
    with col1:
        battlefield = st.checkbox("Battlefield")
        crysis = st.checkbox("Crysis")
    with col2:
        dead_space = st.checkbox("Dead Space")
        f1 = st.checkbox("F1")
    with col3:
        fifa = st.checkbox("Fifa")
        mass_effect = st.checkbox("Mass Effect")
        reckoning = st.checkbox("KOA: Reckoning")
        sim_city = st.checkbox("Sim City")

    plataforma = st.selectbox("Plataforma", ['PC', 'Play Station', 'Xbox', 'Otros'])
    sexo = st.selectbox("Sexo", ['Hombre', 'Mujer'])
    consumidor = st.selectbox("¿Es consumidor habitual?", ['Sí', 'No'])

    submit = st.form_submit_button("🎯 Realizar Predicción")

# --- Procesamiento y predicción ---
if submit:
    # Mostrar parámetros ingresados
    st.markdown("## 📋 Parámetros ingresados:")
    st.write(f"**Edad:** {edad}")
    st.write(f"**Plataforma:** {plataforma}")
    st.write(f"**Sexo:** {sexo}")
    st.write(f"**Consumidor Habitual:** {consumidor}")

    # Tabla visible para el usuario con ✅ y ❌
    datos_visibles = {
        'Edad': edad,
        'Battlefield': "✅" if battlefield else "❌",
        'Crysis': "✅" if crysis else "❌",
        'Dead Space': "✅" if dead_space else "❌",
        'F1': "✅" if f1 else "❌",
        'Fifa': "✅" if fifa else "❌",
        'Mass Effect': "✅" if mass_effect else "❌",
        'KOA: Reckoning': "✅" if reckoning else "❌",
        'Sim City': "✅" if sim_city else "❌",
        'Plataforma': plataforma,
        'Sexo': sexo,
        'Consumidor habitual': consumidor
    }
    st.markdown("## 🧮 Datos que ingresan al modelo (vistos por el usuario):")
    st.dataframe(pd.DataFrame([datos_visibles]))

    # Crear DataFrame con columnas exactas que espera el modelo
    entrada = pd.DataFrame(columns=variables)

    # Edad
    entrada.loc[0, 'Edad'] = edad

    # Videojuegos
    entrada.loc[0, "videojuego_'Battlefield'"] = int(battlefield)
    entrada.loc[0, "videojuego_'Crysis'"] = int(crysis)
    entrada.loc[0, "videojuego_'Dead Space'"] = int(dead_space)
    entrada.loc[0, "videojuego_'F1'"] = int(f1)
    entrada.loc[0, "videojuego_'Fifa'"] = int(fifa)
    entrada.loc[0, "videojuego_'Mass Effect'"] = int(mass_effect)
    entrada.loc[0, "videojuego_'KOA: Reckoning'"] = int(reckoning)
    entrada.loc[0, "videojuego_'Sim City'"] = int(sim_city)

    # Inicializar columnas categóricas en 0
    for col in variables:
        if col.startswith('Plataforma_') or col == 'Sexo_Mujer' or col == 'Consumidor_habitual_True':
            entrada.loc[0, col] = 0

    # Marcar plataforma
    plat_col = 'Plataforma_' + plataforma.replace(' ', '')
    if plat_col in entrada.columns:
        entrada.loc[0, plat_col] = 1

    # Marcar sexo
    entrada.loc[0, 'Sexo_Mujer'] = 1 if sexo == 'Mujer' else 0

    # Marcar consumidor habitual
    entrada.loc[0, 'Consumidor_habitual_True'] = 1 if consumidor == 'Sí' else 0

    # Mostrar columnas enviadas al modelo
  #  st.write("❗ Columnas que envías al modelo:")
   # st.write(entrada.columns.tolist())

    # Mostrar el DataFrame entrada completo
   # st.write("❗ Datos que entran al modelo:")
    #st.write(entrada)

    # Escalar Edad
    if 'Edad' in entrada.columns:
        entrada[['Edad']] = min_max_scaler.transform(entrada[['Edad']])
       # st.write("❗ Edad escalada:")
       # st.write(entrada[['Edad']])

    # Predicción con modelo árbol
    prediccion = model_Tree.predict(entrada)[0]
    st.markdown("## 📈 Resultado de la predicción:")
    st.success(f"💵 El valor estimado del videojuego es: **${prediccion:,.2f} dólares**")








