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
st.title("Predicción de Depósitos Bancarios de clientes")

# Número de llamadas efectuadas en esta campaña:
campaign_range_dict = {
    "1 llamada": 1,
    "2 llamadas": 2,
    "3-5 llamadas": 3,
    "6-10 llamadas": 4,
    "Más de 10 llamadas": 5
}

# Inputs del usuario
st.header("Ingrese la información del cliente:")
campaign_range = st.selectbox(
    "Selecciona el número de llamadas en esta campaña", 
    list(campaign_range_dict.keys())
)

balance = st.number_input("Saldo del cliente (balance en euros):", step=1.0)
age = st.number_input("Edad del cliente (en años):", min_value=18, max_value=100, step=1)
pdays_option = st.selectbox("¿Se tiene información sobre días desde el último contacto?", ["Sí", "No"])
pdays = st.slider("Días desde el último contacto (en días):", min_value=0, max_value=500, step=1) if pdays_option == "Sí" else -1

education = st.selectbox("Nivel de educación", ["Primaria", "Secundaria", "Terciaria", "Desconocida"])
housing = st.radio("¿Tiene vivienda?", ["Sí", "No"])
loan = st.radio("¿Tiene préstamo personal?", ["Sí", "No"])
poutcome_success = st.radio("¿Éxito en la campaña previa?", ["Sí", "No"])

# Escalar las entradas
balance_scaled = scaler_robust.transform([[balance]])
age_scaled = scaler_standard.transform([[age]])
pdays_scaled = scaler_minmax.transform([[pdays]]) if pdays != -1 else np.array([[-1]])

# Mapear las entradas categóricas
education_map = {"Primaria": 0, "Secundaria": 1, "Terciaria": 2, "Desconocida": 3}
housing_map = {"Sí": 1, "No": 0}
loan_map = {"Sí": 1, "No": 0}
poutcome_map = {"Sí": 1, "No": 0}

education_encoded = education_map[education]
housing_encoded = housing_map[housing]
loan_encoded = loan_map[loan]
poutcome_encoded = poutcome_map[poutcome_success]

# Obtener el valor numérico correspondiente del rango de campaña
campaign_range_value = campaign_range_dict[campaign_range]

# Crear la entrada para el modelo
user_input = np.array([[
    campaign_range_value, 
    balance_scaled[0][0], 
    age_scaled[0][0], 
    education_encoded, 
    housing_encoded, 
    loan_encoded, 
    poutcome_encoded, 
    pdays_scaled[0][0]
]])

# Botón para predecir
if st.button("Predecir"):
    prediction = model.predict(user_input)
    prediction_proba = model.predict_proba(user_input)

    if prediction[0] == 1:
        st.success(f"El cliente probablemente hará un depósito bancario. (Confianza: {prediction_proba[0][1]:.2f})")
    else:
        st.error(f"El cliente probablemente NO hará un depósito bancario. (Confianza: {prediction_proba[0][0]:.2f})")
