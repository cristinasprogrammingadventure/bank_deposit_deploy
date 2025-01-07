import streamlit as st
import pickle
import numpy as np

# Cargar el modelo y los escaladores
with open('best_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scalers.pkl', 'rb') as scaler_file:
    scalers = pickle.load(scaler_file)

scaler_robust = scalers['scaler_robust']
scaler_standard = scalers['scaler_standard']
scaler_minmax = scalers['scaler_minmax']

# Título de la aplicación
st.title("Predicción de Depósitos Bancarios")

# Inputs del usuario
balance = st.number_input("Saldo del cliente (balance)", step=100.0)
age = st.number_input("Edad del cliente", min_value=18, max_value=100, step=1)
pdays_option = st.selectbox("¿Se tiene información sobre días desde el último contacto?", ["Sí", "No"])
pdays = st.slider("Días desde el último contacto:", min_value=0, max_value=30) if pdays_option == "Sí" else -1

# Escalar las entradas
balance_scaled = scaler_robust.transform([[balance]])
age_scaled = scaler_standard.transform([[age]])
pdays_scaled = scaler_minmax.transform([[pdays]]) if pdays != -1 else np.array([[-1]])

# Preprocesar las demás variables
# Aquí ajusta tus mapas de codificación según lo necesario para educación, préstamos, etc.

# Crear la entrada para el modelo
user_input = np.array([balance_scaled[0][0], age_scaled[0][0], pdays_scaled[0][0]])

# Botón para predecir
if st.button("Predecir"):
    prediction = model.predict([user_input])
    prediction_proba = model.predict_proba([user_input])

    if prediction[0] == 1:
        st.success(f"El cliente probablemente hará un depósito bancario. (Confianza: {prediction_proba[0][1]:.2f})")
    else:
        st.error(f"El cliente probablemente NO hará un depósito bancario. (Confianza: {prediction_proba[0][0]:.2f})")
