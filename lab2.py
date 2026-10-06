import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from statsmodels.tsa.stattools import (
    adfuller,
    kpss,
    acf,
    pacf
)

from statsmodels.graphics.tsaplots import (
    plot_acf,
    plot_pacf
)

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

import warnings

warnings.filterwarnings("ignore")


# =====================================================================
# ЛАБОРАТОРНАЯ РАБОТА №2
# ИССЛЕДОВАНИЕ ПРОЦЕССОВ AR, MA, ARMA И ARIMA
# =====================================================================
#
# Данные:
# GlobalLandTemperaturesByCity.csv
#
# Город:
# Novosibirsk
#
# ВАЖНО:
# test не используется при выборе p, q, d и модели.
#
# Сглаживание исходного ряда НЕ применяется.
#
# Последние 1000 наблюдений используются после:
# 1. заполнения пропусков;
# 2. обработки выбросов.
# =====================================================================


# =====================================================================
# НАСТРОЙКИ
# =====================================================================

FILE_PATH = "GlobalLandTemperaturesByCity.csv"

CITY = "Novosibirsk"

N_LAST = 1000
N_TRAIN = 900
N_TEST = 100

ALPHA = 0.05

SEASONAL_PERIOD = 12

# -------------------------
# Hampel
# -------------------------

HAMPEL_WINDOW = 13
HAMPEL_THRESHOLD = 3.5

# -------------------------
# ACF / PACF
# -------------------------

MAX_LAGS = 30
MAX_PQ = 5


# =====================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================================


def print_separator(title):

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


# ---------------------------------------------------------------------
# ADF + KPSS
# ---------------------------------------------------------------------

def stationarity_test(series, name):

    series = series.dropna()

    print("\n" + "-" * 75)
    print(name)
    print("-" * 75)

    # ================================================================
    # ADF
    # ================================================================

    adf_result = adfuller(
        series,
        autolag="AIC"
    )

    adf_stat = adf_result[0]
    adf_pvalue = adf_result[1]

    print("\nADF:")
    print(
        f"Статистика: {adf_stat:.6f}"
    )

    print(
        f"p-value:    {adf_pvalue:.6f}"
    )

    print(
        "H0: ряд имеет единичный корень "
        "(ряд нестационарен)."
    )

    if adf_pvalue < ALPHA:

        adf_stationary = True

        print(
            "p-value < α → H0 отвергается → "
            "ряд стационарен."
        )

    else:

        adf_stationary = False

        print(
            "p-value ≥ α → H0 не отвергается → "
            "ряд нестационарен."
        )

    # ================================================================
    # KPSS
    # ================================================================

    try:

        kpss_result = kpss(
            series,
            regression="c",
            nlags="auto"
        )

        kpss_stat = kpss_result[0]
        kpss_pvalue = kpss_result[1]

        print("\nKPSS:")
        print(
            f"Статистика: {kpss_stat:.6f}"
        )

        print(
            f"p-value:    {kpss_pvalue:.6f}"
        )

        print(
            "H0: ряд стационарен."
        )

        if kpss_pvalue < ALPHA:

            kpss_stationary = False

            print(
                "p-value < α → H0 отвергается → "
                "ряд нестационарен."
            )

        else:

            kpss_stationary = True

            print(
                "p-value ≥ α → H0 не отвергается → "
                "ряд стационарен."
            )

    except Exception as error:

        print(
            f"\nОшибка KPSS: {error}"
        )

        kpss_stat = np.nan
        kpss_pvalue = np.nan
        kpss_stationary = False

    # ================================================================
    # Общий вывод
    # ================================================================

    if adf_stationary and kpss_stationary:

        conclusion = "стационарен"

    elif not adf_stationary and not kpss_stationary:

        conclusion = "нестационарен"

    else:

        conclusion = "неоднозначный результат"

    print(
        f"\nОбщий вывод: {conclusion}"
    )

    return {

        "adf_stat": adf_stat,
        "adf_pvalue": adf_pvalue,

        "kpss_stat": kpss_stat,
        "kpss_pvalue": kpss_pvalue,

        "adf_stationary": adf_stationary,
        "kpss_stationary": kpss_stationary,

        "stationary": (
            adf_stationary
            and
            kpss_stationary
        ),

        "conclusion": conclusion
    }


# ---------------------------------------------------------------------
# Hampel
# ---------------------------------------------------------------------

def hampel_filter(
    series,
    window=13,
    threshold=3.5
):

    values = series.values

    n = len(values)

    mask = np.zeros(
        n,
        dtype=bool
    )

    replacements = {}

    half_window = window // 2

    for i in range(n):

        left = max(
            0,
            i - half_window
        )

        right = min(
            n,
            i + half_window + 1
        )

        local_window = values[
            left:right
        ]

        current_position = i - left

        local_values = np.delete(
            local_window,
            current_position
        )

        if len(local_values) == 0:

            continue

        local_median = np.median(
            local_values
        )

        local_mad = np.median(
            np.abs(
                local_values -
                local_median
            )
        )

        if local_mad == 0:

            local_mad = 1e-8

        modified_z = (
            0.6745 *
            abs(
                values[i] -
                local_median
            ) /
            local_mad
        )

        if modified_z > threshold:

            mask[i] = True

            replacements[i] = (
                local_median
            )

    return (
        pd.Series(
            mask,
            index=series.index
        ),
        replacements
    )


# =====================================================================
# ЭТАП 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# =====================================================================

print_separator(
    "ЭТАП 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ"
)


# ---------------------------------------------------------------------
# Структура CSV
# ---------------------------------------------------------------------

columns = [
    "date",
    "AverageTemperature",
    "AverageTemperatureUncertainty",
    "City",
    "Country",
    "Latitude",
    "Longitude"
]


df = pd.read_csv(
    FILE_PATH,
    header=None,
    names=columns
)


print(
    f"Исходный размер данных: {df.shape}"
)


# ---------------------------------------------------------------------
# Выбор города
# ---------------------------------------------------------------------

df = df[
    df["City"]
    .astype(str)
    .str.strip()
    ==
    CITY
].copy()


print(
    f"Город: {CITY}"
)

print(
    f"Количество записей: {len(df)}"
)


# ---------------------------------------------------------------------
# Даты
# ---------------------------------------------------------------------

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.dropna(
    subset=["date"]
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


# ---------------------------------------------------------------------
# Дубликаты
# ---------------------------------------------------------------------

duplicate_count = (
    df["date"]
    .duplicated()
    .sum()
)

print(
    f"Дубликатов дат: {duplicate_count}"
)


if duplicate_count > 0:

    df = (
        df
        .groupby(
            "date",
            as_index=False
        )
        .agg({
            "AverageTemperature": "mean",
            "AverageTemperatureUncertainty": "mean",
            "City": "first",
            "Country": "first",
            "Latitude": "first",
            "Longitude": "first"
        })
        .sort_values("date")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Формирование временного ряда
# ---------------------------------------------------------------------

ts_full = (
    df
    .set_index("date")
    ["AverageTemperature"]
    .sort_index()
)


# ---------------------------------------------------------------------
# Регулярная месячная шкала
# ---------------------------------------------------------------------

expected_index = pd.date_range(
    start=ts_full.index.min(),
    end=ts_full.index.max(),
    freq="MS"
)


missing_dates = (
    expected_index
    .difference(ts_full.index)
)


print(
    f"Пропущенных месяцев: "
    f"{len(missing_dates)}"
)


ts_regular = (
    ts_full
    .reindex(expected_index)
)


ts_regular.index.name = "date"


# ---------------------------------------------------------------------
# Заполнение пропусков
# ---------------------------------------------------------------------

missing_before = (
    ts_regular.isna().sum()
)

print(
    f"Пропусков до заполнения: "
    f"{missing_before}"
)


ts_regular = (
    ts_regular
    .interpolate(
        method="linear",
        limit_direction="both"
    )
)


print(
    f"Пропусков после заполнения: "
    f"{ts_regular.isna().sum()}"
)


# =====================================================================
# ОБРАБОТКА ВЫБРОСОВ
# =====================================================================

print_separator(
    "ОБРАБОТКА ВЫБРОСОВ"
)


hampel_mask, hampel_replacements = (
    hampel_filter(
        ts_regular,
        window=HAMPEL_WINDOW,
        threshold=HAMPEL_THRESHOLD
    )
)


hampel_outliers = (
    ts_regular[hampel_mask]
)


print(
    f"Найдено выбросов Hampel: "
    f"{len(hampel_outliers)}"
)


ts_cleaned_full = (
    ts_regular.copy()
)


for position, replacement in (
    hampel_replacements.items()
):

    ts_cleaned_full.iloc[position] = (
        replacement
    )


print(
    "Выбросы заменены локальной медианой."
)

print(
    f"Окно Hampel: {HAMPEL_WINDOW}"
)

print(
    f"Порог Hampel: {HAMPEL_THRESHOLD}"
)


# ---------------------------------------------------------------------
# График до/после обработки выбросов
# ---------------------------------------------------------------------

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    ts_regular.index,
    ts_regular.values,
    linewidth=0.7,
    alpha=0.5,
    label="После заполнения пропусков"
)

plt.plot(
    ts_cleaned_full.index,
    ts_cleaned_full.values,
    linewidth=0.9,
    label="После обработки выбросов"
)

if len(hampel_outliers) > 0:

    plt.scatter(
        hampel_outliers.index,
        hampel_outliers.values,
        marker="o",
        facecolors="none",
        s=45,
        label="Выбросы Hampel"
    )


plt.title(
    "Обработка выбросов методом Hampel"
)

plt.xlabel("Дата")
plt.ylabel("AverageTemperature")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# =====================================================================
# ПОСЛЕДНИЕ 1000 НАБЛЮДЕНИЙ
# =====================================================================

if len(ts_cleaned_full) < N_LAST:

    raise ValueError(
        "Недостаточно наблюдений для анализа."
    )


ts_1000 = (
    ts_cleaned_full
    .tail(N_LAST)
    .copy()
)


print(
    f"\nИспользуется последних "
    f"{len(ts_1000)} наблюдений."
)

print(
    f"Период: "
    f"{ts_1000.index.min().date()} — "
    f"{ts_1000.index.max().date()}"
)


# =====================================================================
# ЭТАП 2. TRAIN / TEST
# =====================================================================

print_separator(
    "ЭТАП 2. РАЗДЕЛЕНИЕ НА TRAIN И TEST"
)


train = (
    ts_1000
    .iloc[:N_TRAIN]
    .copy()
)


test = (
    ts_1000
    .iloc[
        N_TRAIN:
        N_TRAIN + N_TEST
    ]
    .copy()
)


print(
    f"Всего: {len(ts_1000)}"
)

print(
    f"Train: {len(train)}"
)

print(
    f"Test:  {len(test)}"
)


print(
    f"\nTrain: "
    f"{train.index.min().date()} — "
    f"{train.index.max().date()}"
)

print(
    f"Test:  "
    f"{test.index.min().date()} — "
    f"{test.index.max().date()}"
)


# ---------------------------------------------------------------------
# График train/test
# ---------------------------------------------------------------------

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    train.index,
    train.values,
    label="Train"
)

plt.plot(
    test.index,
    test.values,
    label="Test"
)

plt.axvline(
    train.index[-1],
    linestyle="--",
    label="Граница train/test"
)

plt.title(
    f"Последние 1000 наблюдений — {CITY}"
)

plt.xlabel("Дата")
plt.ylabel("AverageTemperature")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# =====================================================================
# ЭТАП 3. СТАЦИОНАРНОСТЬ
# =====================================================================

print_separator(
    "ЭТАП 3. ИССЛЕДОВАНИЕ СТАЦИОНАРНОСТИ"
)


# ---------------------------------------------------------------------
# d = 0
# ---------------------------------------------------------------------

stationarity_d0 = stationarity_test(
    train,
    "TRAIN без дифференцирования (d = 0)"
)


d = 0

stationary_series = train.copy()


# ---------------------------------------------------------------------
# d = 1
# ---------------------------------------------------------------------

if not stationarity_d0["stationary"]:

    d = 1

    stationary_series = (
        train
        .diff()
        .dropna()
    )

    stationarity_d1 = stationarity_test(
        stationary_series,
        "TRAIN после первого дифференцирования (d = 1)"
    )

else:

    stationarity_d1 = None


# ---------------------------------------------------------------------
# d = 2
# ---------------------------------------------------------------------

if (
    not stationarity_d0["stationary"]
    and
    not stationarity_d1["stationary"]
):

    d = 2

    stationary_series = (
        train
        .diff()
        .diff()
        .dropna()
    )

    stationarity_d2 = stationarity_test(
        stationary_series,
        "TRAIN после второго дифференцирования (d = 2)"
    )

else:

    stationarity_d2 = None


print_separator(
    "ВЫБРАННЫЙ ПОРЯДОК ДИФФЕРЕНЦИРОВАНИЯ"
)

print(
    f"Минимальный выбранный порядок: d = {d}"
)


# ---------------------------------------------------------------------
# График исходного train
# ---------------------------------------------------------------------

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    train.index,
    train.values
)

plt.title(
    "Исходный обучающий временной ряд"
)

plt.xlabel("Дата")
plt.ylabel("AverageTemperature")

plt.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------
# График стационарного ряда
# ---------------------------------------------------------------------

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    stationary_series.index,
    stationary_series.values
)

plt.axhline(
    0,
    linestyle="--"
)

plt.title(
    f"Стационарный ряд после дифференцирования, d = {d}"
)

plt.xlabel("Дата")
plt.ylabel("Дифференцированное значение")

plt.grid(True)

plt.tight_layout()
plt.show()


# =====================================================================
# ЭТАП 4. ACF / PACF
# =====================================================================

print_separator(
    "ЭТАП 4. ACF И PACF"
)


max_lags = min(
    MAX_LAGS,
    len(stationary_series) // 3
)


# ---------------------------------------------------------------------
# ACF
# ---------------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

plot_acf(
    stationary_series,
    lags=max_lags,
    ax=plt.gca()
)

plt.title(
    f"ACF стационарного ряда, d = {d}"
)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------
# PACF
# ---------------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

plot_pacf(
    stationary_series,
    lags=max_lags,
    ax=plt.gca(),
    method="ywm"
)

plt.title(
    f"PACF стационарного ряда, d = {d}"
)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------
# Численные значения ACF/PACF
# ---------------------------------------------------------------------

acf_values = acf(
    stationary_series,
    nlags=max_lags,
    fft=True
)

pacf_values = pacf(
    stationary_series,
    nlags=max_lags,
    method="ywm"
)


significance_level = (
    1.96 /
    np.sqrt(
        len(stationary_series)
    )
)


significant_acf_lags = [
    lag
    for lag in range(
        1,
        max_lags + 1
    )
    if abs(acf_values[lag])
    > significance_level
]


significant_pacf_lags = [
    lag
    for lag in range(
        1,
        max_lags + 1
    )
    if abs(pacf_values[lag])
    > significance_level
]


print(
    f"Граница значимости: "
    f"±{significance_level:.4f}"
)

print(
    "Значимые лаги ACF:",
    significant_acf_lags
)

print(
    "Значимые лаги PACF:",
    significant_pacf_lags
)


# =====================================================================
# ФОРМИРОВАНИЕ КАНДИДАТОВ p И q
# =====================================================================

p_candidates = [
    lag
    for lag in significant_pacf_lags
    if lag <= MAX_PQ
]

q_candidates = [
    lag
    for lag in significant_acf_lags
    if lag <= MAX_PQ
]


# ---------------------------------------------------------------------
# Добавляем простые модели, если графики не дали
# достаточно кандидатов.
# ---------------------------------------------------------------------

if len(p_candidates) == 0:

    p_candidates = [1]


if len(q_candidates) == 0:

    q_candidates = [1]


# Добавляем второй кандидат для выполнения
# требования о нескольких вариантах.

if len(p_candidates) == 1:

    second_p = (
        2
        if p_candidates[0] != 2
        else 3
    )

    if second_p <= MAX_PQ:

        p_candidates.append(
            second_p
        )


if len(q_candidates) == 1:

    second_q = (
        2
        if q_candidates[0] != 2
        else 3
    )

    if second_q <= MAX_PQ:

        q_candidates.append(
            second_q
        )


p_candidates = sorted(
    set(p_candidates)
)

q_candidates = sorted(
    set(q_candidates)
)


print_separator(
    "КАНДИДАТЫ p И q"
)

print(
    f"p_candidates = {p_candidates}"
)

print(
    f"q_candidates = {q_candidates}"
)


# =====================================================================
# ЭТАП 5. ПОСТРОЕНИЕ МОДЕЛЕЙ
# =====================================================================

print_separator(
    "ЭТАП 5. ПОСТРОЕНИЕ МОДЕЛЕЙ"
)


model_candidates = []


# =====================================================================
# AR
# =====================================================================

for p in p_candidates:

    model_candidates.append({

        "family": "AR",

        "order": (
            p,
            d,
            0
        ),

        "p": p,
        "d": d,
        "q": 0,

        "series_type": "original"
    })


# =====================================================================
# MA
# =====================================================================

for q in q_candidates:

    model_candidates.append({

        "family": "MA",

        "order": (
            0,
            d,
            q
        ),

        "p": 0,
        "d": d,
        "q": q,

        "series_type": "original"
    })


# =====================================================================
# ARMA
# =====================================================================
#
# ARMA применяется к стационарному ряду.
#
# Если d = 0:
#     стационарным является исходный train.
#
# Если d > 0:
#     ARMA строится на стационарном
#     дифференцированном ряду.
#
# =====================================================================

for p in p_candidates:

    for q in q_candidates:

        model_candidates.append({

            "family": "ARMA",

            "order": (
                p,
                0,
                q
            ),

            "p": p,
            "d": 0,
            "q": q,

            "series_type": "stationary"
        })


# =====================================================================
# ARIMA
# =====================================================================

for p in p_candidates:

    for q in q_candidates:

        model_candidates.append({

            "family": "ARIMA",

            "order": (
                p,
                d,
                q
            ),

            "p": p,
            "d": d,
            "q": q,

            "series_type": "original"
        })


print(
    "\nКоличество кандидатных моделей:",
    len(model_candidates)
)


for candidate in model_candidates:

    print(
        f'{candidate["family"]}'
        f'{candidate["order"]}'
        f' — '
        f'{candidate["series_type"]}'
    )


# =====================================================================
# ОЦЕНКА МОДЕЛЕЙ
# =====================================================================

model_results = []


for candidate in model_candidates:

    family = candidate["family"]

    order = candidate["order"]

    series_type = candidate[
        "series_type"
    ]

    try:

        # -------------------------------------------------------------
        # Выбор ряда для обучения
        # -------------------------------------------------------------

        if series_type == "original":

            model_series = train

            trend = (
                "c"
                if order[1] == 0
                else None
            )

        else:

            model_series = (
                stationary_series
            )

            trend = "c"


        # -------------------------------------------------------------
        # Построение модели
        # -------------------------------------------------------------

        model = ARIMA(

            model_series,

            order=order,

            trend=trend
        )


        fitted_model = model.fit()


        model_results.append({

            "family": family,

            "order": order,

            "series_type": series_type,

            "model": fitted_model,

            "AIC": fitted_model.aic,

            "BIC": fitted_model.bic
        })


    except Exception as error:

        print(
            f"Ошибка при построении "
            f"{family}{order}: {error}"
        )


# =====================================================================
# ТАБЛИЦА AIC / BIC
# =====================================================================

results_table = pd.DataFrame([

    {

        "Model": result["family"],

        "Order": result["order"],

        "Series": result["series_type"],

        "AIC": result["AIC"],

        "BIC": result["BIC"]

    }

    for result in model_results
])


results_table = (
    results_table
    .sort_values(
        "AIC"
    )
    .reset_index(
        drop=True
    )
)


print_separator(
    "СРАВНЕНИЕ AIC / BIC"
)


print(
    results_table.to_string(
        index=False
    )
)


# =====================================================================
# ПРЕДВАРИТЕЛЬНАЯ ДИАГНОСТИКА ОСТАТКОВ
# =====================================================================
#
# Для выбора модели используем Ljung–Box.
#
# ВАЖНО:
# подробные графики остатков будут построены
# только для окончательно выбранной модели.
# =====================================================================


diagnostic_results = []


for result in model_results:

    residuals = pd.Series(
        result["model"].resid
    ).dropna()


    lb_lag = min(
        20,
        max(
            5,
            len(residuals) // 10
        )
    )


    try:

        lb_result = acorr_ljungbox(

            residuals,

            lags=[lb_lag],

            return_df=True
        )


        lb_pvalue = (
            lb_result[
                "lb_pvalue"
            ].iloc[0]
        )


    except Exception:

        lb_pvalue = np.nan


    diagnostic_results.append({

        "Model": result["family"],

        "Order": result["order"],

        "Series": result["series_type"],

        "ResidualMean":
            residuals.mean(),

        "LjungBox_pvalue":
            lb_pvalue,

        "WhiteNoise":
            (
                lb_pvalue >= ALPHA
                if not np.isnan(lb_pvalue)
                else False
            )
    })


diagnostic_table = pd.DataFrame(
    diagnostic_results
)


comparison_table = results_table.merge(

    diagnostic_table,

    on=[
        "Model",
        "Order",
        "Series"
    ],

    how="left"
)


# =====================================================================
# ВЫБОР МОДЕЛИ
# =====================================================================

print_separator(
    "ВЫБОР МОДЕЛИ"
)


# ---------------------------------------------------------------------
# Для моделей ARMA на дифференцированном ряду AIC/BIC относятся
# к другой зависимой переменной.
#
# Поэтому если d > 0, окончательный выбор модели для прогноза
# исходной температуры выполняется среди моделей, построенных
# непосредственно на исходном train:
#
# AR, MA, ARIMA.
#
# ARMA при этом остаётся рассмотренной моделью согласно заданию.
# ---------------------------------------------------------------------

if d > 0:

    selection_table = (
        comparison_table[
            comparison_table["Series"]
            ==
            "original"
        ]
        .copy()
    )

else:

    selection_table = (
        comparison_table
        .copy()
    )


# ---------------------------------------------------------------------
# Сначала предпочитаем модели с белыми остатками.
# ---------------------------------------------------------------------

white_noise_models = (
    selection_table[
        selection_table[
            "WhiteNoise"
        ]
        ==
        True
    ]
    .copy()
)


if len(white_noise_models) > 0:

    selected_row = (

        white_noise_models

        .sort_values(
            [
                "AIC",
                "BIC"
            ]
        )

        .iloc[0]
    )

    selection_reason = (
        "Выбрана модель с остатками, "
        "совместимыми с белым шумом, "
        "и минимальным AIC среди таких моделей."
    )

else:

    selected_row = (

        selection_table

        .sort_values(
            [
                "AIC",
                "BIC"
            ]
        )

        .iloc[0]
    )

    selection_reason = (
        "Ни одна модель не прошла критерий "
        "Льюнга–Бокса; выбрана модель "
        "с минимальным AIC."
    )


# ---------------------------------------------------------------------
# Получаем объект выбранной модели
# ---------------------------------------------------------------------

selected_model_result = None


for result in model_results:

    if (

        result["family"]
        ==
        selected_row["Model"]

        and

        result["order"]
        ==
        selected_row["Order"]

        and

        result["series_type"]
        ==
        selected_row["Series"]

    ):

        selected_model_result = result

        break


if selected_model_result is None:

    raise RuntimeError(
        "Не удалось найти выбранную модель."
    )


selected_model = (
    selected_model_result["model"]
)


selected_model_name = (
    selected_row["Model"]
)


selected_order = (
    selected_row["Order"]
)


print(
    f"Выбранная модель: "
    f"{selected_model_name}"
    f"{selected_order}"
)


print(
    f"AIC: "
    f"{selected_row['AIC']:.4f}"
)


print(
    f"BIC: "
    f"{selected_row['BIC']:.4f}"
)


print(
    f"Среднее остатков: "
    f"{selected_row['ResidualMean']:.6f}"
)


print(
    f"Ljung–Box p-value: "
    f"{selected_row['LjungBox_pvalue']:.6f}"
)


print(
    f"\nОбоснование: "
    f"{selection_reason}"
)


# =====================================================================
# ЭТАП 6. ПОДРОБНАЯ ДИАГНОСТИКА ОСТАТКОВ
# =====================================================================

print_separator(
    "ЭТАП 6. ДИАГНОСТИКА ОСТАТКОВ ВЫБРАННОЙ МОДЕЛИ"
)


residuals = pd.Series(
    selected_model.resid
).dropna()


print(
    f"Среднее значение остатков: "
    f"{residuals.mean():.6f}"
)


# ---------------------------------------------------------------------
# График остатков
# ---------------------------------------------------------------------

plt.figure(
    figsize=(14, 4)
)

plt.plot(
    residuals.index,
    residuals.values
)

plt.axhline(
    0,
    linestyle="--"
)

plt.title(
    f"Остатки модели "
    f"{selected_model_name}"
    f"{selected_order}"
)

plt.xlabel("Дата")
plt.ylabel("Остаток")

plt.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------
# ACF остатков
# ---------------------------------------------------------------------

residual_lags = min(
    20,
    len(residuals) // 3
)


plt.figure(
    figsize=(12, 4)
)

plot_acf(
    residuals,
    lags=residual_lags,
    ax=plt.gca()
)

plt.title(
    f"ACF остатков "
    f"{selected_model_name}"
    f"{selected_order}"
)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------
# Ljung–Box
# ---------------------------------------------------------------------

lb_lag = min(
    20,
    max(
        5,
        len(residuals) // 10
    )
)


lb_result = acorr_ljungbox(

    residuals,

    lags=[lb_lag],

    return_df=True
)


lb_stat = (
    lb_result[
        "lb_stat"
    ].iloc[0]
)


lb_pvalue = (
    lb_result[
        "lb_pvalue"
    ].iloc[0]
)


print(
    f"\nLjung–Box, lag={lb_lag}"
)


print(
    f"Статистика: "
    f"{lb_stat:.6f}"
)


print(
    f"p-value: "
    f"{lb_pvalue:.6f}"
)


print(
    "H0: отсутствуют совместно значимые "
    "автокорреляции остатков."
)


if lb_pvalue >= ALPHA:

    print(
        "H0 не отвергается → "
        "остатки совместимы с белым шумом."
    )

    residuals_are_white = True

else:

    print(
        "H0 отвергается → "
        "в остатках сохраняется "
        "автокорреляция."
    )

    residuals_are_white = False


# =====================================================================
# ЭТАП 7. ФИНАЛЬНЫЙ ПРОГНОЗ
# =====================================================================
#
# TEST до этого момента НЕ использовался.
# =====================================================================

print_separator(
    "ЭТАП 7. ФИНАЛЬНЫЙ ПРОГНОЗ НА 100 НАБЛЮДЕНИЙ"
)


# ---------------------------------------------------------------------
# Важный случай:
#
# Если выбрана ARMA на стационарном дифференцированном ряду,
# она не должна напрямую использоваться как прогноз модели
# исходной температуры.
#
# В таком случае для финального прогноза используется
# соответствующая ARIMA(p,d,q) на исходном train.
# ---------------------------------------------------------------------

forecast_model = selected_model


if (

    selected_row["Series"]
    ==
    "stationary"

    and

    d > 0

):

    forecast_order = (
        selected_order[0],
        d,
        selected_order[2]
    )


    print(
        "ARMA построена на стационарном "
        "дифференцированном ряду."
    )

    print(
        "Для прогноза исходной температуры "
        "используется соответствующая "
        f"ARIMA{forecast_order}."
    )


    forecast_model = ARIMA(

        train,

        order=forecast_order,

        trend=None
    ).fit()


# ---------------------------------------------------------------------
# Прогноз
# ---------------------------------------------------------------------

forecast_result = (
    forecast_model
    .get_forecast(
        steps=N_TEST
    )
)


forecast = (
    forecast_result
    .predicted_mean
)


forecast.index = (
    test.index
)


# ---------------------------------------------------------------------
# Доверительный интервал
# ---------------------------------------------------------------------

confidence_interval = (
    forecast_result
    .conf_int(
        alpha=ALPHA
    )
)


confidence_interval.index = (
    test.index
)


lower_bound = (
    confidence_interval.iloc[:, 0]
)


upper_bound = (
    confidence_interval.iloc[:, 1]
)


print(
    f"Количество прогнозов: "
    f"{len(forecast)}"
)


print(
    f"Требовалось: {N_TEST}"
)


# =====================================================================
# ГРАФИК ПРОГНОЗА
# =====================================================================

plt.figure(
    figsize=(15, 6)
)


# Train

plt.plot(
    train.index,
    train.values,
    label="Train"
)


# Фактический test

plt.plot(
    test.index,
    test.values,
    label="Test"
)


# Прогноз

plt.plot(
    forecast.index,
    forecast.values,
    linestyle="--",
    label="Прогноз"
)


# Интервал

plt.fill_between(

    test.index,

    lower_bound,

    upper_bound,

    alpha=0.2,

    label="95% доверительный интервал"
)


plt.axvline(

    train.index[-1],

    linestyle="--",

    label="Граница train/test"
)


plt.title(
    f"Финальный прогноз "
    f"{selected_model_name}"
    f"{selected_order}"
)


plt.xlabel("Дата")
plt.ylabel("AverageTemperature")


plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# =====================================================================
# ЭТАП 8. MAE И RMSE
# =====================================================================

print_separator(
    "ЭТАП 8. ОЦЕНКА КАЧЕСТВА ПРОГНОЗА"
)


actual = (
    test.values
)


predicted = (
    forecast.values
)


# ---------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------

mae_model = np.mean(
    np.abs(
        actual -
        predicted
    )
)


# ---------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------

rmse_model = np.sqrt(
    np.mean(
        (
            actual -
            predicted
        ) ** 2
    )
)


print(
    f"MAE выбранной модели: "
    f"{mae_model:.6f}"
)


print(
    f"RMSE выбранной модели: "
    f"{rmse_model:.6f}"
)


# =====================================================================
# ЭТАП 9. НАИВНЫЙ ПРОГНОЗ
# =====================================================================

print_separator(
    "ЭТАП 9. НАИВНЫЙ ПРОГНОЗ"
)


last_train_value = (
    train.iloc[-1]
)


naive_forecast = np.repeat(
    last_train_value,
    N_TEST
)


# ---------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------

mae_naive = np.mean(
    np.abs(
        actual -
        naive_forecast
    )
)


# ---------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------

rmse_naive = np.sqrt(
    np.mean(
        (
            actual -
            naive_forecast
        ) ** 2
    )
)


print(
    f"Последнее значение train: "
    f"{last_train_value:.6f}"
)


print(
    f"MAE naive: "
    f"{mae_naive:.6f}"
)


print(
    f"RMSE naive: "
    f"{rmse_naive:.6f}"
)


# =====================================================================
# СРАВНЕНИЕ С BASELINE
# =====================================================================

print_separator(
    "СРАВНЕНИЕ С НАИВНЫМ ПРОГНОЗОМ"
)


mae_improvement = (

    (
        mae_naive -
        mae_model
    )
    /
    mae_naive
    *
    100

)


rmse_improvement = (

    (
        rmse_naive -
        rmse_model
    )
    /
    rmse_naive
    *
    100

)


print(
    f"MAE модели:   {mae_model:.6f}"
)

print(
    f"MAE baseline: {mae_naive:.6f}"
)


print(
    f"\nRMSE модели:   {rmse_model:.6f}"
)

print(
    f"RMSE baseline: {rmse_naive:.6f}"
)


print(
    f"\nИзменение MAE: "
    f"{mae_improvement:.2f}%"
)


print(
    f"Изменение RMSE: "
    f"{rmse_improvement:.2f}%"
)


# =====================================================================
# ЭТАП 10. ИТОГОВЫЙ ВЫВОД
# =====================================================================

print_separator(
    "ЭТАП 10. ИТОГОВЫЙ ВЫВОД"
)


print(
    f"\nВыбранная модель: "
    f"{selected_model_name}"
    f"{selected_order}"
)


print(
    f"Порядок дифференцирования: "
    f"d = {d}"
)


print(
    f"AIC: "
    f"{selected_row['AIC']:.6f}"
)


print(
    f"BIC: "
    f"{selected_row['BIC']:.6f}"
)


print(
    f"Среднее остатков: "
    f"{residuals.mean():.6f}"
)


print(
    f"Ljung–Box p-value: "
    f"{lb_pvalue:.6f}"
)


if residuals_are_white:

    print(
        "Остатки не имеют статистически "
        "значимой автокорреляции и "
        "совместимы с белым шумом."
    )

else:

    print(
        "В остатках сохраняется "
        "статистически значимая "
        "автокорреляция."
    )


print(
    f"\nMAE модели: "
    f"{mae_model:.6f}"
)


print(
    f"RMSE модели: "
    f"{rmse_model:.6f}"
)


print(
    f"\nMAE naive: "
    f"{mae_naive:.6f}"
)


print(
    f"RMSE naive: "
    f"{rmse_naive:.6f}"
)


if (

    mae_model < mae_naive
    and
    rmse_model < rmse_naive

):

    print(
        "\nВыбранная модель превосходит "
        "наивный прогноз по обоим "
        "критериям MAE и RMSE."
    )

elif (

    mae_model < mae_naive
    or
    rmse_model < rmse_naive

):

    print(
        "\nВыбранная модель превосходит "
        "наивный прогноз по одному "
        "из двух критериев."
    )

else:

    print(
        "\nВыбранная модель не превосходит "
        "наивный прогноз по MAE и RMSE."
    )


# =====================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =====================================================================

final_results = {

    "model_name":
        selected_model_name,

    "order":
        selected_order,

    "p":
        selected_order[0],

    "d":
        selected_order[1],

    "q":
        selected_order[2],

    "AIC":
        selected_row["AIC"],

    "BIC":
        selected_row["BIC"],

    "residual_mean":
        residuals.mean(),

    "LjungBox_stat":
        lb_stat,

    "LjungBox_pvalue":
        lb_pvalue,

    "MAE_model":
        mae_model,

    "RMSE_model":
        rmse_model,

    "MAE_naive":
        mae_naive,

    "RMSE_naive":
        rmse_naive,

    "MAE_improvement_percent":
        mae_improvement,

    "RMSE_improvement_percent":
        rmse_improvement
}


print_separator(
    "ФИНАЛЬНЫЕ ПАРАМЕТРЫ"
)


for key, value in final_results.items():

    print(
        f"{key}: {value}"
    )


print_separator(
    "ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА"
)