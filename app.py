
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Datos de entrenamiento corregidos y en español
data = {
    "Sexo": ["Hombre", "Mujer", "Hombre", "Hombre", "Hombre", "Mujer", "Mujer", "Hombre", "Hombre", "Mujer"],
    "Presión arterial": ["Baja", "Normal", "Baja", "Alta", "Normal", "Baja", "Normal", "Alta", "Baja", "Alta"],
    "Colesterol": ["Alto", "Alto", "Alto", "Normal", "Alto", "Normal", "Alto", "Normal", "Normal", "Alto"],
    "Edad agrupada": ["20s", "20s", "40s", "20s", "20s", "20s", "20s", "30s", "50s", "60s"],
    "Na/K agrupado": ["<10", "10-20", "20-30", ">30", "20-30", "10-20", "<10", "10-20", "<10", "10-20"],
    "Tipo de medicamento": ["fármaco X", "fármaco X", "fármaco Y", "fármaco Y", "fármaco Y", "fármaco X", "fármaco X", "fármaco X", "fármaco X", "fármaco B"]
}

df = pd.DataFrame(data)

# Codificadores
le_dict = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Entrenamiento del modelo
X = df.drop("Tipo de medicamento", axis=1)
y = df["Tipo de medicamento"]
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Opciones del formulario
fields = ["Sexo", "Presión arterial", "Colesterol", "Edad agrupada", "Na/K agrupado"]
user_input = {}

# Título de la app
st.set_page_config(page_title="Predicción de Medicamento", page_icon="💊")
st.title("💊 Sistema de Predicción Médica para Neumonía")
st.markdown("---")

# Formulario de entrada
with st.form("prediccion_form"):
    st.subheader("📝 Ingrese los datos del paciente:")
    for field in fields:
        options = list(le_dict[field].classes_)
        user_input[field] = st.selectbox(field, options, key=field)
    submit = st.form_submit_button("🔍 Predecir medicamento")

# Predicción
if submit:
    try:
        input_vals = [le_dict[field].transform([user_input[field]])[0] for field in fields]
        pred_num = model.predict([input_vals])[0]
        pred_label = le_dict["Tipo de medicamento"].inverse_transform([pred_num])[0]
        st.success(f"💊 Medicamento recomendado: **{pred_label}**")
    except Exception as e:
        st.error(f"Error en la predicción: {e}")

# Mostrar árbol de decisión
if st.button("🌳 Mostrar árbol de decisión"):
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(
        model,
        feature_names=fields,
        class_names=le_dict["Tipo de medicamento"].classes_,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax
    )
    st.pyplot(fig)
