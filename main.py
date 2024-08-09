import streamlit as st
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import classification_report
import numpy as np

# Заголовок приложения
st.title("Логистическая регрессия")
# Загрузка файла
uploaded_file = st.file_uploader("Загрузите файл Excel", type=["xlsx"])

if uploaded_file is not None:
    # Чтение данных
    data = pd.read_excel(uploaded_file)

    st.write("Данные:")
    st.write(data.head())

    # Разделение на особенности и целевую переменную
    X = data.iloc[:, :-1]  # Все столбцы, кроме последнего, как особенности
    y = data.iloc[:, -1]  # Последний столбец как целевая переменная

    # Добавление константы
    X_with_const = sm.add_constant(X)

    # Обучение модели
    model = sm.Logit(y, X_with_const).fit()

    # Вывод коэффициентов, p-значений, ОШ и 95% ДИ
    st.write("Результаты логистической регрессии:")
    results_table = model.summary2().tables[1]
    results_table['Отношение шансов (ОШ)'] = np.exp(results_table['Coef.'])
    results_table = results_table.rename(columns={'Coef.': 'Коэффициент'})
    conf = model.conf_int()
    conf['Нижняя 95%-ая ДИ'] = np.exp(conf[0])
    conf['Верхняя 95%-ая ДИ'] = np.exp(conf[1])
    results_table = results_table.join(conf[['Нижняя 95%-ая ДИ', 'Верхняя 95%-ая ДИ']])
    st.write(results_table[['Коэффициент', 'Отношение шансов (ОШ)', 'Нижняя 95%-ая ДИ', 'Верхняя 95%-ая ДИ', 'P>|z|']])

    # Логарифмическое отношение правдоподобия
    lr_test = model.llr_pvalue
    st.write(f"P-значение всей модели (Тест отношения правдоподобия): {lr_test:.4f}")

