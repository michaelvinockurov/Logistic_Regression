import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from scipy.stats import norm, chi2

# Заголовок приложения
st.title("Расчет логистической регрессии")

# Поля для ввода целевой переменной
st.subheader("Введите целевую переменную (0 или 1)")
y_input = st.text_input("Целевая переменная (разделите пробелами)", value="0 1 0 1 0")
y = np.array([int(x) for x in y_input.split()])

# Определение количества наблюдений
num_rows = len(y)

# Инициализация списка предикторов
if 'predictors' not in st.session_state:
    st.session_state['predictors'] = []

# Функции для добавления и удаления предикторов
def add_predictor():
    st.session_state.predictors.append([])

def remove_predictor():
    if st.session_state.predictors:
        st.session_state.predictors.pop()

# Кнопки для взаимодействия
st.button("Добавить предиктор", on_click=add_predictor)
st.button("Удалить предиктор", on_click=remove_predictor)

# Поля для предикторов
st.subheader("Введите данные для предикторов:")
num_predictors = len(st.session_state.predictors)
predictor_names = []

# Создание полей для ввода предикторов
if num_predictors > 0:
    for i in range(num_predictors):
        col1, col2 = st.columns([2, 3])  # Две колонки для названия и ввода
        with col1:
            pred_name = st.text_input(f"Название предиктора {i + 1}", value=f"Предиктор {i + 1}", key=f"name_{i}")
            predictor_names.append(pred_name)
        with col2:
            pred_input = st.text_input(f"Данные для предиктора {i + 1} (разделите пробелами)", value="", key=f"pred_{i}")
            if pred_input:
                st.session_state.predictors[i] = np.array([float(x) for x in pred_input.split()])
else:
    col1, col2 = st.columns([2, 3])
    with col1:
        pred_name = st.text_input("Название предиктора 1", value="Предиктор 1", key="name_0")
        predictor_names.append(pred_name)
    with col2:
        pred_input = st.text_input("Данные для предиктора 1 (разделите пробелами)", value="", key="pred_0")
        if pred_input:
            st.session_state.predictors.append(np.array([float(x) for x in pred_input.split()]))

# Проверка, если все данные введены
if st.button("Запустить логистическую регрессию"):
    if len(y) != num_rows or any(len(pred) != num_rows for pred in st.session_state.predictors):
        st.error("Убедитесь, что все данные введены правильно.")
    else:
        # Подготовка данных для логистической регрессии
        X = np.column_stack(st.session_state.predictors)

        # Обучение модели
        model = LogisticRegression()
        model.fit(X, y)

        # Получение коэффициентов модели
        coef = np.concatenate(([model.intercept_[0]], model.coef_[0]))  # Константа и коэффициенты
        intercept = model.intercept_[0]

        # Получение стандартных ошибок и p-значений через бутстрап
        conf_intervals = []
        standard_errors = []
        p_values = []

        for i in range(len(coef)):
            bootstrap_samples = []
            for _ in range(1000):
                X_sample, y_sample = resample(X, y)
                model.fit(X_sample, y_sample)
                bootstrap_samples.append(model.intercept_[0] if i == 0 else model.coef_[0][i - 1])

            lower = np.percentile(bootstrap_samples, 2.5)
            upper = np.percentile(bootstrap_samples, 97.5)
            conf_intervals.append((lower, upper))

            standard_error = np.std(bootstrap_samples)
            standard_errors.append(standard_error)

            z_score = coef[i] / standard_error if standard_error != 0 else 0
            p_value = 2 * (1 - norm.cdf(abs(z_score)))  # Двусторонний тест
            p_values.append(p_value)

        # Подготовка результатов
        results_table = pd.DataFrame({
            'Коэффициент': coef,
            'Стандартная ошибка': standard_errors,
            'p-значение': p_values,
            'Отношение шансов (ОШ)': np.exp(coef),
            'Нижняя 95%-ая ДИ': [ci[0] for ci in conf_intervals],
            'Верхняя 95%-ая ДИ': [ci[1] for ci in conf_intervals],
        }, index=['Константа'] + predictor_names)

        # Вывод результатов в красивую таблицу
        st.write("### Результаты логистической регрессии")
        st.dataframe(results_table, use_container_width=True)

        # Статистика модели
        chi_square = model.score(X, y) * len(y)  # Хи-квадрат
        df = len(st.session_state.predictors)  # Степени свободы
        model_p_value = 1 - chi2.cdf(chi_square, df)  # p-значение для модели

        st.markdown("### Статистика модели:")
        st.markdown(f"<div style='border:1px solid #ccc; padding: 10px;'>"
                    f"<strong>Хи-квадрат:</strong> {chi_square:.4f}<br>"
                    f"<strong>Степени свободы:</strong> {df}<br>"
                    f"<strong>p-значение:</strong> {model_p_value:.4f}<br>"
                    f"</div>", unsafe_allow_html=True)