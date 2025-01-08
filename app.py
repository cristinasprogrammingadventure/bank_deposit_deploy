# Variables seleccionadas para el entreno del modelo elegido
# selected_features = ['campaign_ranges', 'balance', 'age', 'education', 'housing', 'loan', poutcome_success', 'pdays']

import streamlit as st
import pickle
import numpy as np

# Cargar el modelo
with open('best_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Cargar los escaladores individuales
with open('robust_scaler.pkl', 'rb') as robust_file:
    scaler_robust = pickle.load(robust_file)

with open('standard_scaler.pkl', 'rb') as standard_file:
    scaler_standard = pickle.load(standard_file)

with open('minmax_scaler.pkl', 'rb') as minmax_file:
    scaler_minmax = pickle.load(minmax_file)

# Título de la aplicación
st.title("Predicción de Depósitos Bancarios de clientes")

# Inputs del usuario
campaign_range_dict = {
    "1 llamada": 1,
    "2 llamadas": 2,
    "3-5 llamadas": 3,
    "6-10 llamadas": 4,
    "Más de 10 llamadas": 5
}

campaign_range = st.selectbox(
    "Selecciona el número de llamadas recibidas por el cliente en esta campaña", 
    list(campaign_range_dict.keys())
)

balance = st.number_input("Saldo del cliente (balance en euros):", step=1.0)
age = st.number_input("Edad del cliente (en años):", min_value=18, max_value=100, step=1)
pdays_option = st.selectbox("¿Se tiene información sobre días desde el último contacto?", ["Sí", "No"])
pdays = st.slider("Días desde el último contacto (en días):", min_value=0, max_value=500, step=1) if pdays_option == "Sí" else -1
education = st.selectbox("Máximo nivel educativo", ["Primaria", "Secundaria", "Terciaria", "Desconocida"])
housing = st.radio("¿Tiene préstamo hipotecario ahora?", ["Sí", "No"])
loan = st.radio("¿Tiene préstamo personal ahora?", ["Sí", "No"])
poutcome_success = st.radio("¿Se suscribió con éxito en campañas previas?", ["Sí", "No"])

# Escalar las entradas
balance_scaled = scaler_robust.transform([[balance]])
age_scaled = scaler_standard.transform([[age]])
pdays_scaled = scaler_minmax.transform([[pdays]]) if pdays != -1 else np.array([[-1]])

education_map = {"Primaria": 0, "Secundaria": 1, "Terciaria": 2, "Desconocida": 3}
housing_map = {"Sí": 1, "No": 0}
loan_map = {"Sí": 1, "No": 0}
poutcome_map = {"Sí": 1, "No": 0}

education_encoded = education_map[education]
housing_encoded = housing_map[housing]
loan_encoded = loan_map[loan]
poutcome_encoded = poutcome_map[poutcome_success]

campaign_range_value = campaign_range_dict[campaign_range]

user_input = np.array([[campaign_range_value, 
                        balance_scaled[0][0], 
                        age_scaled[0][0], 
                        education_encoded, 
                        housing_encoded, 
                        loan_encoded, 
                        poutcome_encoded, 
                        pdays_scaled[0][0]]])

# Botón para predecir
if st.button("Predecir"):
    prediction = model.predict(user_input)
    prediction_proba = model.predict_proba(user_input)

    if prediction[0] == 1:
        st.success(f"POSITIVO: el cliente probablemente se suscribirá al depósito. (Confianza: {prediction_proba[0][1]:.2f})")
    else:
        st.error(f"NEGATIVO: el cliente probablemente NO se suscribirá al depósito. (Confianza: {prediction_proba[0][0]:.2f})")
