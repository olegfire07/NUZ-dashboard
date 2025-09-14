# Запуск: streamlit run nuz_dashboard_app_v10_final_simplified_v6.py
from __future__ import annotations

import re
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative as qcolors
import streamlit as st

# A) Глобальные флаги
APP_VERSION = "v24.56-heatmap-fix"
# Режим: работать только с «Итого по месяцу» из файла, без формул/досчётов
SIMPLE_MODE = True

# A) Вспомогательные функции для года
YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")

def guess_year_from_filename(name: str) -> int | None:
    s = str(name).lower().replace("г.", " ").replace("г", " ")
    m = YEAR_RE.search(s)
    return int(m.group(1)) if m else None

# ------------------------- Конфигурация страницы -------------------------
st.set_page_config(page_title=f"НЮЗ — Дашборд {APP_VERSION}", layout="wide", page_icon="📊")
st.markdown("""
<style>
:root { --text-base: 15px; }
html, body, [class*="css"]  { font-size: var(--text-base); }
section[data-testid="stSidebar"] { min-width: 330px !important; }
h1, h2, h3 { font-weight: 700; }
h2 { margin-top: .6rem; }
.block-container { padding-top: 1rem; padding-bottom: 3rem; }
[data-testid="stMetricValue"] { font-weight: 800; }
[data-testid="stMetricDelta"] { font-size: 0.95rem; }
.dataframe th, .dataframe td { padding: 6px 10px !important; }
[data-testid="stTable"] td, [data-testid="stTable"] th { padding: 8px 10px !important; }
code, .number-cell { font-variant-numeric: tabular-nums; }
.gtitle, .xtitle, .ytitle { font-weight: 700 !important; }
.badge { display:inline-block; padding:2px 8px; border-radius:10px; background:#EEF5FF; color:#1d4ed8; font-size:12px; margin-left:6px; }
.small { color:#6b7280; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ------------------------- Константы и словари ---------------------------
class Metrics(str, Enum):
    REVENUE = "Выручка от распродажи НЮЗ (руб)"
    LOAN_ISSUE = "Выдано займов НЮЗ (руб)"
    BELOW_LOAN = "Товар проданный ниже суммы займа НЮЗ (руб)"
    LOAN_VALUE_OF_SOLD = "Ссуда вышедших изделий на аукцион НЮЗ (руб)"
    AUCTIONED_ITEMS_COUNT = "Количество вышедших изделий на аукцион НЮЗ"
    PENALTIES_RECEIVED = "Получено % и пени НЮЗ (руб)"
    MARKUP_AMOUNT = "Получено наценки от распродажи НЮЗ (руб)"
    PENALTIES_PLUS_MARKUP = "Получено % и пени + наценка на распродажу НЮЗ (руб)"
    LOAN_ISSUE_UNITS = "Выдано займов НЮЗ (шт)"
    BELOW_LOAN_UNITS = "Товар проданный ниже суммы займа НЮЗ (шт)"
    DEBT = "Ссудная задолженность (руб)"
    DEBT_UNITS = "Ссудная задолженность без распродажи НЮЗ (шт)"
    DEBT_NO_SALE = "Ссудная задолженность без распродажи НЮЗ (руб)"
    MARKUP_PCT = "Процент наценки НЮЗ"
    AVG_LOAN = "Средняя сумма займа НЮЗ (руб)"
    AVG_LOAN_TERM = "Средний срок займа НЮЗ (дней)"
    ILLIQUID_BY_COUNT_PCT = "Доля неликвида от количества (%)"
    ILLIQUID_BY_VALUE_PCT = "Доля неликвида от оценки (%)"
    YIELD = "Доходность"
    ISSUE_SHARE = "Доля НЮЗ по выдаче"
    DEBT_SHARE = "Доля НЮЗ по ссудной задолженности"
    RISK_SHARE = "Доля ниже займа, %"
    CALC_MARKUP_PCT = "Расчетная наценка за период, %"

    # Планы (проценты выполнения)
    PLAN_ISSUE_PCT = "% выполнения плана выданных займов НЮЗ"
    PLAN_PENALTIES_PCT = "% выполнения плана по полученным % и пеням НЮЗ"
    PLAN_REVENUE_PCT = "% выполнения плана по выручке от распродажи НЮЗ"

    # Клиенты / филиалы
    UNIQUE_CLIENTS = "Уникальные клиенты"
    NEW_UNIQUE_CLIENTS = "Новые уникальные клиенты"
    BRANCH_COUNT = "Количество ломбардов"
    BRANCH_NEW_COUNT = "Количество новых ломбардов"
    BRANCH_CLOSED_COUNT = "Количество закрытых ломбардов"

    # Операционные
    REDEEMED_ITEMS_COUNT = "Количество выкупленных залогов за период НЮЗ (шт)"
    LOAN_REPAYMENT_SUM = "Сумма погашения суммы займа НЮЗ (руб)"
    LOSS_BELOW_LOAN = "Убыток от товара проданного ниже суммы займа НЮЗ (руб)"

    # Доли
    INTEREST_SHARE = "Доля НЮЗ по полученным % и пени"

    # Новые метрики
    REDEEMED_SUM = "Сумма выкупленных за период НЮЗ (руб)"
    REDEEMED_SHARE_PCT = "Доля выкупов заложенных за период НЮЗ (%)"

# ⚪️ Белый список: берём из файла только эти показатели (после нормализации имени метрики)
ACCEPTED_METRICS_CANONICAL = {
    Metrics.DEBT_NO_SALE.value,
    Metrics.LOAN_ISSUE.value,
    Metrics.PLAN_ISSUE_PCT.value,
    Metrics.LOAN_ISSUE_UNITS.value,
    Metrics.PENALTIES_RECEIVED.value,
    Metrics.PLAN_PENALTIES_PCT.value,
    Metrics.REVENUE.value,
    Metrics.PLAN_REVENUE_PCT.value,
    Metrics.MARKUP_AMOUNT.value,
    Metrics.PENALTIES_PLUS_MARKUP.value,
    Metrics.BRANCH_COUNT.value,
    Metrics.BRANCH_NEW_COUNT.value,
    Metrics.BRANCH_CLOSED_COUNT.value,
    Metrics.LOAN_VALUE_OF_SOLD.value,
    Metrics.AUCTIONED_ITEMS_COUNT.value,
    Metrics.DEBT_SHARE.value,
    Metrics.ISSUE_SHARE.value,
    Metrics.INTEREST_SHARE.value,
    Metrics.AVG_LOAN.value,
    Metrics.MARKUP_PCT.value,
    Metrics.REDEEMED_SUM.value,
    Metrics.REDEEMED_ITEMS_COUNT.value,
    Metrics.REDEEMED_SHARE_PCT.value,
    Metrics.AVG_LOAN_TERM.value,
    Metrics.LOAN_REPAYMENT_SUM.value,
    Metrics.BELOW_LOAN_UNITS.value,
    Metrics.BELOW_LOAN.value,
    Metrics.LOSS_BELOW_LOAN.value,
    Metrics.DEBT_UNITS.value,
    Metrics.ILLIQUID_BY_COUNT_PCT.value,
    Metrics.ILLIQUID_BY_VALUE_PCT.value,
    Metrics.YIELD.value,
}
ACCEPTED_METRICS_CANONICAL |= {Metrics.DEBT.value}


ORDER = ["Январь","Февраль","Март","Апрель","Май","Июнь","Июль","Август","Сентябрь","Октябрь","Ноябрь","Декабрь"]
ORDER_WITH_TOTAL = ORDER + ["Итого"]

NUZ_ACTIVITY_METRICS = {
    Metrics.DEBT_NO_SALE.value,
    Metrics.LOAN_ISSUE.value,
    Metrics.LOAN_ISSUE_UNITS.value,
    Metrics.PENALTIES_RECEIVED.value,
    Metrics.REVENUE.value,
    Metrics.MARKUP_AMOUNT.value,
    Metrics.PENALTIES_PLUS_MARKUP.value,
    Metrics.LOAN_VALUE_OF_SOLD.value,
    Metrics.AUCTIONED_ITEMS_COUNT.value,
    Metrics.AVG_LOAN.value,
    Metrics.MARKUP_PCT.value,
    Metrics.REDEEMED_SUM.value,
    Metrics.REDEEMED_ITEMS_COUNT.value,
    Metrics.AVG_LOAN_TERM.value,
    Metrics.LOAN_REPAYMENT_SUM.value,
    Metrics.BELOW_LOAN_UNITS.value,
    Metrics.BELOW_LOAN.value,
    Metrics.LOSS_BELOW_LOAN.value,
    Metrics.DEBT_UNITS.value,
    Metrics.ILLIQUID_BY_COUNT_PCT.value,
    Metrics.ILLIQUID_BY_VALUE_PCT.value,
    Metrics.YIELD.value,
    Metrics.ISSUE_SHARE.value,
    Metrics.DEBT_SHARE.value,
    Metrics.INTEREST_SHARE.value,
    Metrics.PLAN_ISSUE_PCT.value,
    Metrics.PLAN_PENALTIES_PCT.value,
    Metrics.PLAN_REVENUE_PCT.value,
}

def _month_sort_key(m: str) -> int:
    return ORDER.index(m) if m in ORDER else len(ORDER) + 1

@st.cache_data
def get_monthly_totals_from_file(df_raw: pd.DataFrame, regions: Tuple[str, ...], metric: str) -> pd.DataFrame:
    """Возвращает помесячные значения из строк «Итого» по приоритету."""
    base = df_raw[
        df_raw["Регион"].isin(regions) &
        (df_raw["Показатель"] == metric) &
        (df_raw["Месяц"].astype(str) != "Итого") &
        df_raw["Подразделение"].str.contains(r"^\s*итого\b", case=False, na=False)
    ].copy()
    if base.empty:
        return pd.DataFrame()

    prio = pd.Series(
        np.where(base.get("ИсточникФайла", pd.Series(index=base.index, dtype=object)).eq("TOTALS_FILE"), 1, 2),
        index=base.index
    )
    base["__prio__"] = prio

    base.sort_values(["Регион", "Месяц", "__prio__"], inplace=True)
    best = (base.groupby(["Регион", "Месяц"], observed=True)
                 .first().reset_index()[["Регион", "Месяц", "Значение"]])
    return best

def month_series_from_file(df_all, regions, metric, months):
    dfm = get_monthly_totals_from_file(df_all, tuple(regions), metric)
    if dfm.empty:
        return pd.Series(dtype=float)
    s = (dfm[dfm["Месяц"].astype(str).isin(months)]
            .groupby("Месяц", observed=True)["Значение"].sum())
    # строгая сортировка по календарю
    s = s.reindex([m for m in months if m in s.index])
    return s

@st.cache_data
def sorted_months_safe(_values) -> list[str]:
    """Без кеша: аккуратно приводим к строкам и сортируем по нашему ORDER."""
    if _values is None:
        return []
    s = pd.Series(_values)
    if pd.api.types.is_categorical_dtype(s):
        s = s.astype(str)
    seq = [str(x) for x in s.dropna().astype(str)]
    seq = [m for m in seq if m in ORDER]
    seq = list(dict.fromkeys(seq))
    return sorted(seq, key=_month_sort_key)

# --- Агрегационные правила за период из строк «Итого по месяцу»
# SUM: потоковые суммы за месяц (руб/шт) — складываем.
AGG_SUM = {
    Metrics.REVENUE.value, Metrics.LOAN_ISSUE.value, Metrics.PENALTIES_RECEIVED.value,
    Metrics.MARKUP_AMOUNT.value, Metrics.PENALTIES_PLUS_MARKUP.value,
    Metrics.LOAN_ISSUE_UNITS.value, Metrics.BELOW_LOAN.value, Metrics.BELOW_LOAN_UNITS.value,
    Metrics.LOAN_VALUE_OF_SOLD.value, Metrics.AUCTIONED_ITEMS_COUNT.value,
    Metrics.REDEEMED_ITEMS_COUNT.value, Metrics.REDEEMED_SUM.value,
    Metrics.LOAN_REPAYMENT_SUM.value, Metrics.LOSS_BELOW_LOAN.value,
    Metrics.BRANCH_NEW_COUNT.value, Metrics.BRANCH_CLOSED_COUNT.value,
    Metrics.UNIQUE_CLIENTS.value, Metrics.NEW_UNIQUE_CLIENTS.value,
}

# MEAN: проценты/доли и средние показатели — усредняем по месяцам.
AGG_MEAN = {
    Metrics.MARKUP_PCT.value, Metrics.YIELD.value,
    Metrics.ILLIQUID_BY_COUNT_PCT.value, Metrics.ILLIQUID_BY_VALUE_PCT.value,
    Metrics.ISSUE_SHARE.value, Metrics.DEBT_SHARE.value, Metrics.INTEREST_SHARE.value,
    Metrics.PLAN_ISSUE_PCT.value, Metrics.PLAN_PENALTIES_PCT.value, Metrics.PLAN_REVENUE_PCT.value,
    Metrics.AVG_LOAN.value,      # если берется из файла как «Итого по месяцу»
    Metrics.AVG_LOAN_TERM.value, # ⬅️ ВАЖНО: средний срок — среднее по месяцам
    Metrics.REDEEMED_SHARE_PCT.value,
}

# LAST (снимок на конец месяца) — берём ПОСЛЕДНИЙ месяц периода.
AGG_LAST = {
    Metrics.DEBT.value, Metrics.DEBT_NO_SALE.value, Metrics.DEBT_UNITS.value,
    Metrics.BRANCH_COUNT.value,  # «Количество ломбардов» — снимок
}

# (оставим для совместимости, если где-то используются)
METRICS_SUM  = AGG_SUM.copy()
METRICS_MEAN = AGG_MEAN.copy()
METRICS_LAST = AGG_LAST.copy()

def aggregation_rule(metric: str) -> str:
    if metric in AGG_SUM:  return "sum"
    if metric in AGG_MEAN: return "mean"
    if metric in AGG_LAST: return "last"
    # по умолчанию: проценты — mean, деньги/шт — sum
    return "mean" if is_percent_metric(metric) else "sum"

def period_value_from_itogo(df_all: pd.DataFrame, regions: list[str], metric: str, months: list[str]) -> float | None:
    s = month_series_from_file(df_all, regions, metric, months)
    if s.empty:
        return None
    rule = aggregation_rule(metric)
    vals = pd.to_numeric(s, errors="coerce")
    if rule == "sum":
        return float(vals.sum())
    if rule == "mean":
        return float(vals.mean())
    if rule == "last":
        # берём последнее доступное значение внутри выбранного окна
        last_months = sorted_months_safe(vals.dropna().index)
        if not last_months: return None
        v = vals.get(last_months[-1], np.nan)
        return float(v) if pd.notna(v) else None
    return None

def period_value_from_itogo_for_region(df_all: pd.DataFrame, region: str, metric: str,
                                       months: list[str], *, snapshots_mode: str = "last") -> float | None:
    # Берём ровно строки «Итого по месяцу» для региона
    dfm = get_monthly_totals_from_file(df_all, (region,), metric)
    if dfm.empty:
        return None
    part = dfm[dfm["Месяц"].astype(str).isin(months)]
    if part.empty:
        return None

    vals = pd.to_numeric(part["Значение"], errors="coerce").dropna()
    if vals.empty:
        return None

    rule = aggregation_rule(metric)

    if rule == "sum":
        return float(vals.sum())
    if rule == "mean":
        return float(vals.mean())

    # rule == "last" → снимок: специально для сводки берём среднее, если так попросили
    if rule == "last":
        if snapshots_mode == "mean":
            return float(vals.mean())
        else:
            # берём последнее по календарю
            part = part.copy()
            part["Месяц"] = pd.Categorical(part["Месяц"].astype(str), categories=ORDER, ordered=True)
            part = part.sort_values("Месяц")
            return float(pd.to_numeric(part["Значение"], errors="coerce").iloc[-1])

    # дефолт
    return float(vals.mean())


def period_values_by_region_from_itogo(df_all, regions, metric, months) -> dict[str, float]:
    """
    Возвращает {Регион: значение за период} строго из строк «Итого по месяцу».
    Сумма/среднее/последний — как задано aggregation_rule(metric).
    """
    dfm = get_monthly_totals_from_file(df_all, tuple(regions), metric)
    if dfm.empty:
        return {}

    dfm = dfm[dfm["Месяц"].astype(str).isin(months)].copy()
    if dfm.empty:
        return {}

    rule = aggregation_rule(metric)
    out = {}
    for reg, t in dfm.groupby("Регион"):
        # Ensure months are sorted before taking the last value for 'last' rule
        t = t.copy()
        t['Месяц'] = pd.Categorical(t['Месяц'], categories=ORDER, ordered=True)
        t = t.sort_values("Месяц")
        vals = pd.to_numeric(t["Значение"], errors="coerce").dropna()
        if vals.empty:
            continue

        if rule == "sum":
            out[str(reg)] = float(vals.sum())
        elif rule == "mean":
            out[str(reg)] = float(vals.mean())
        elif rule == "last":
            out[str(reg)] = float(vals.iloc[-1])
        else:
            out[str(reg)] = float(vals.mean())  # дефолт
    return out


METRICS_BIGGER_IS_BETTER = {
    Metrics.REVENUE.value, Metrics.LOAN_ISSUE.value, Metrics.MARKUP_PCT.value,
    Metrics.YIELD.value, Metrics.PENALTIES_RECEIVED.value, Metrics.MARKUP_AMOUNT.value,
    Metrics.PENALTIES_PLUS_MARKUP.value, Metrics.AVG_LOAN_TERM.value
}
METRICS_SMALLER_IS_BETTER = {
    Metrics.BELOW_LOAN.value, Metrics.RISK_SHARE.value,
    Metrics.ILLIQUID_BY_COUNT_PCT.value, Metrics.ILLIQUID_BY_VALUE_PCT.value
}

METRIC_HELP: Dict[str, str] = {
    Metrics.MARKUP_PCT.value: "Показатель прибыльности распродажи: насколько цена реализации превышает оценочную (ссудную) стоимость. Высокий % наценки означает, что залоговые изделия продаются существенно дороже суммы выданных по ним займов, что хорошо для прибыли.",
    Metrics.RISK_SHARE.value: "Рискованный оборот: доля выручки от продаж, пришедшаяся на лоты, которые были проданы дешевле суммы выданного по ним займа. Чем выше этот показатель, тем больше случаев, когда компания понесла убыток при реализации залога. Рост доли ниже займа – тревожный сигнал.",
    Metrics.YIELD.value: "Эффективность выдач: отношение полученных процентов и пеней к сумме выданных займов за период. Проще говоря, средний процентный доход с каждого выданного рубля. На доходность влияет срок и процентная ставка займов.",
    Metrics.AVG_LOAN_TERM.value: "Среднее время, на которое клиенты брали займы. Длинный срок может увеличивать процентные доходы, но и откладывает возврат денег. Короткий срок свидетельствует либо о быстром возврате, либо о переходе займа в стадию продажи.",
    Metrics.ILLIQUID_BY_COUNT_PCT.value: "Часть товарных запасов (залогов), которая признается неликвидной (труднореализуемой) по количеству. Высокая доля означает, что значительная часть залоговых изделий долго не продается, замораживая капитал компании.",
    Metrics.ILLIQUID_BY_VALUE_PCT.value: "Часть товарных запасов (залогов), которая признается неликвидной (труднореализуемой) по стоимости. Аналогично доле по количеству, но учитывает денежный объем.",
    Metrics.REVENUE.value: "Деньги, суммируем по подразделениям и по времени.",
    Metrics.LOAN_ISSUE.value: "Деньги, суммируем.",
    Metrics.BELOW_LOAN.value: "Сумма продаж товаров, реализованных ниже суммы выданного по ним займа.",
    Metrics.DEBT.value: "Снимок на конец месяца. В KPI/сводке за период — среднее месячных остатков по выбранному окну.",
    Metrics.AVG_LOAN.value: "**Приоритет 'Итого':** если в файле есть итоговая строка, берутся данные из неё. Иначе — взвешенно: Σ(Выдано, руб)/Σ(Выдано, шт).",
    Metrics.MARKUP_AMOUNT.value: "Сумма. Приоритет — строковое «Итого по региону», если оно есть.",
    Metrics.DEBT_NO_SALE.value: "Задолженность без распродажи на конец месяца (снимок). В периоде считаем среднее/последний, а не сумму."
}
METRIC_HELP.update({
    Metrics.PLAN_ISSUE_PCT.value: "Насколько выполнен план по сумме выдач. Берём готовое значение из файла; агрегация — среднее.",
    Metrics.PLAN_PENALTIES_PCT.value: "Выполнение плана по процентам и пеням. Берём из файла; агрегация — среднее.",
    Metrics.PLAN_REVENUE_PCT.value: "Выполнение плана по выручке распродажи. Берём из файла; агрегация — среднее.",
    Metrics.UNIQUE_CLIENTS.value: "Количество уникальных клиентов в месяце. В периоде суммируется (без дедупликации между месяцами).",
    Metrics.NEW_UNIQUE_CLIENTS.value: "Новые уникальные клиенты за месяц. В периоде суммируется.",
    Metrics.BRANCH_COUNT.value: "Количество ломбардов на конец месяца (снимок). В периоде — среднее/последний.",
    Metrics.BRANCH_NEW_COUNT.value: "Открыто новых ломбардов за месяц (снимок/сумма в зависимости от источника).",
    Metrics.BRANCH_CLOSED_COUNT.value: "Закрыто ломбардов за месяц.",
    Metrics.REDEEMED_ITEMS_COUNT.value: "Выкупленные залоги за период (шт). Суммируется.",
    Metrics.LOAN_REPAYMENT_SUM.value: "Сумма погашений основной суммы займа (руб). Суммируется.",
    Metrics.LOSS_BELOW_LOAN.value: "Убыток от продаж ниже суммы займа (руб). Суммируется.",
    Metrics.INTEREST_SHARE.value: "Доля НЮЗ по полученным % и пени. Берём из файла; агрегация — среднее.",
})

PERCENT_METRICS = {
    Metrics.MARKUP_PCT.value,
    Metrics.ILLIQUID_BY_COUNT_PCT.value,
    Metrics.ILLIQUID_BY_VALUE_PCT.value,
    Metrics.YIELD.value,
    Metrics.RISK_SHARE.value,
    Metrics.ISSUE_SHARE.value,
    Metrics.DEBT_SHARE.value,
    Metrics.REDEEMED_SHARE_PCT.value,
}
PERCENT_METRICS |= {
    Metrics.PLAN_ISSUE_PCT.value,
    Metrics.PLAN_PENALTIES_PCT.value,
    Metrics.PLAN_REVENUE_PCT.value,
    Metrics.INTEREST_SHARE.value,
}

def is_percent_metric(name: str) -> bool:
    if not name:
        return False
    s = str(name)
    low = s.lower()
    return (
        s in PERCENT_METRICS
        or "доля" in low
        or ("наценк" in low and s in {Metrics.MARKUP_PCT.value, Metrics.CALC_MARKUP_PCT.value})
    )

def agg_of_metric(metric: str) -> str:
    return "sum" if metric in METRICS_SUM else ("last" if metric in METRICS_LAST else "mean")

def format_rub(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.0f} ₽".replace(",", " ")

def fmt_pct(v: float | None, digits: int = 2) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.{digits}f}%"

def fmt_days(v: float | None, digits: int = 2) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.{digits}f} дн."

def y_fmt_for_metric(m: str) -> tuple[str, str]:
    """Возвращает (plotly tickformat, suffix_for_hover)"""
    if ("руб" in m) or (m == Metrics.DEBT.value):
        return ",.0f", " ₽"
    if m == Metrics.DEBT_UNITS.value:
        return ",.0f", " шт"
    if is_percent_metric(m):
        return ".2f", "%"
    if "дней" in m:
        return ".2f", " дн."
    return ",.2f", ""

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.CategoricalIndex):
        df.columns = df.columns.astype(str)
    df.columns.name = None
    return df

def detect_category(raw_text: str) -> str:
    """Определяем, к чему относится строка: НЮЗ / ЮЗ / Общее (без явной метки)."""
    s = (raw_text or "").lower()
    # Признаки НЮЗ / ЮЗ (учитываем возможные опечатки и пробелы)
    has_nuz = re.search(r"\bн\s*ю\s*з\b|нюз", s) is not None
    has_yuz = re.search(r"\bю\s*з\b|юз", s) is not None
    if has_nuz and not has_yuz:
        return "НЮЗ"
    if has_yuz and not has_nuz:
        return "ЮЗ"
    return "Общее"

def normalize_metric_name(name: str) -> str:
    if name is None: return ""
    s = str(name).strip()
    t = re.sub(r"[^\wа-яё]+", " ", s.lower(), flags=re.IGNORECASE).strip()

    # --- планы (% выполнения) ---
    if ("выполн" in t and "план" in t and "выдан" in t and "займ" in t):
        return Metrics.PLAN_ISSUE_PCT.value
    if ("выполн" in t and "план" in t and ("пени" in t or "процент" in t or "проц" in t) and "получ" in t):
        return Metrics.PLAN_PENALTIES_PCT.value
    if ("выполн" in t and "план" in t and "выручк" in t and ("распрод" in t or "нюз" in t)):
        return Metrics.PLAN_REVENUE_PCT.value

    # Быстрые маркеры:
    is_nuz = ("нюз" in t)
    is_yuz = (re.search(r"\bюз\b", t) is not None)

    # 🔁 СИНОНИМЫ из твоего списка → канон
    if is_nuz and ("выкуп" in t) and ("колич" in t) and ("за период" in t):
        return Metrics.REDEEMED_ITEMS_COUNT.value
    if is_nuz and (("сумм" in t) and ("выкуп" in t) and ("за период" in t)):
        return Metrics.REDEEMED_SUM.value
    if is_nuz and ("доля" in t) and ("выкуп" in t) and ("за период" in t):
        return Metrics.REDEEMED_SHARE_PCT.value
    if is_nuz and ("средн" in t) and ("ссуд" in t) and ("задолж" in t):
        return Metrics.DEBT_NO_SALE.value

    # --- только НЮЗ — деньги/шт ---
    if is_nuz and "ссуд" in t and "задолж" in t and "без" in t and ("распрод" in t or "нюз" in t) and "шт" not in t:
        return Metrics.DEBT_NO_SALE.value
    if is_nuz and "ссуд" in t and "задолж" in t and "шт" in t:
        return Metrics.DEBT_UNITS.value
    if is_nuz and "ссуд" in t and "задолж" in t:
        return Metrics.DEBT.value

    if is_nuz and "выруч" in t and ("распрод" in t or "продаж" in t):
        return Metrics.REVENUE.value

    if is_nuz and "выдан" in t and "займ" in t and ("шт" in t or "штук" in t):
        return Metrics.LOAN_ISSUE_UNITS.value
    if is_nuz and "выдан" in t and "займ" in t:
        return Metrics.LOAN_ISSUE.value

    if is_nuz and "получено" in t and ("пени" in t or "проц" in t):
        return Metrics.PENALTIES_RECEIVED.value
    if is_nuz and "получено" in t and ("наценк" in t or "оценк" in t) and ("распрод" in t or "продаж" in t):
        return Metrics.MARKUP_AMOUNT.value
    if is_nuz and ("получено" in t and "пени" in t and ("наценк" in t or "оценк" in t)):
        return Metrics.PENALTIES_PLUS_MARKUP.value

    if is_nuz and "наценк" in t:
        return Metrics.MARKUP_PCT.value

    if is_nuz and "ссуда" in t and "вышедших" in t and "аукцион" in t:
        return Metrics.LOAN_VALUE_OF_SOLD.value
    if is_nuz and "колич" in t and "вышедших" in t:
        return Metrics.AUCTIONED_ITEMS_COUNT.value

    # Клиенты/филиалы — без привязки к НЮЗ (как в исходниках)
    if "колич" in t and "ломбард" in t and "нов" in t:
        return Metrics.BRANCH_NEW_COUNT.value
    if "колич" in t and "ломбард" in t and "закрыт" in t:
        return Metrics.BRANCH_CLOSED_COUNT.value
    if "колич" in t and "ломбард" in t:
        return Metrics.BRANCH_COUNT.value

    # Операционные НЮЗ — выкупы/погашения (добавлены твои формулировки):
    if is_nuz and ("погаш" in t) and ("сумм" in t) and ("займ" in t):
        return Metrics.LOAN_REPAYMENT_SUM.value
    if is_nuz and ("колич" in t and "выкуп" in t):
        return Metrics.REDEEMED_ITEMS_COUNT.value

    # Рисковые/derived:
    if is_nuz and "ниже" in t and "займ" in t and ("шт" in t or "штук" in t):
        return Metrics.BELOW_LOAN_UNITS.value
    if is_nuz and "ниже" in t and "займ" in t:
        return Metrics.BELOW_LOAN.value
    if is_nuz and "убыток" in t and "ниже" in t and "займ" in t:
        return Metrics.LOSS_BELOW_LOAN.value

    # Средние:
    if is_nuz and (("средн" in t and "сумм" in t and "займ" in t) or ("avg" in t and "loan" in t)):
        return Metrics.AVG_LOAN.value
    if is_nuz and "средн" in t and "срок" in t and ("пер иод" in t or "период" in t or "за пер" in t) and "займ" in t:
        return Metrics.AVG_LOAN_TERM.value

    # Доли НЮЗ:
    if is_nuz and "доходност" in t:
        return Metrics.YIELD.value
    if is_nuz and "доля" in t and "выдач" in t:
        return Metrics.ISSUE_SHARE.value
    if is_nuz and "доля" in t and "ссудн" in t and "задолж" in t:
        return Metrics.DEBT_SHARE.value
    if is_nuz and "доля" in t and "нюз" in t and ("пени" in t or "проц" in t) and ("получ" in t):
        return Metrics.INTEREST_SHARE.value

    return s

def normalize_month_token(x) -> str | None:
    RUS_MONTHS = {
        "январь": "Январь", "янв": "Январь", "февраль": "Февраль", "фев": "Февраль",
        "март": "Март", "мар": "Март", "апрель": "Апрель", "апр": "Апрель",
        "май": "Май", "июнь": "Июнь", "июль": "Июль", "август": "Август", "авг": "Август",
        "сентябрь": "Сентябрь", "сен": "Сентябрь", "октябрь": "Октябрь", "окт": "Октябрь",
        "ноябрь": "Ноябрь", "ноя": "Ноябрь", "декабрь": "Декабрь", "дек": "Декабрь",
        "итого": "Итого", "итог": "Итого"
    }
    if x is None: return None
    s = str(x).strip().lower().replace(".", "")
    s = re.sub(r"[\d\sггод]+$", "", s).strip()
    return RUS_MONTHS.get(s)

@st.cache_data
def consistent_color_map(keys: Tuple[str, ...]) -> Dict[str, str]:
    pal = (qcolors.Plotly + qcolors.D3 + qcolors.Set3 + qcolors.Dark24 + qcolors.Light24)
    return {k: pal[i % len(pal)] for i, k in enumerate(sorted(map(str, keys)))}

def detect_month_header(df: pd.DataFrame, max_header_rows: int = 15) -> tuple[int, list[tuple[int, str]]] | None:
    for r in range(min(max_header_rows, len(df))):
        row = df.iloc[r, :].tolist()
        month_cols = [(j, m) for j, val in enumerate(row) if (m := normalize_month_token(val)) in ORDER_WITH_TOTAL]
        if len({m for _, m in month_cols if m in ORDER}) >= 3:
            cleaned = []
            last_m = None
            for j, m in sorted(month_cols, key=lambda x: x[0]):
                if m != last_m:
                    cleaned.append((j, m))
                    last_m = m
            return r, cleaned
    return None

def guess_data_sheet(xl: pd.ExcelFile) -> str:
    if "TDSheet" in xl.sheet_names: return "TDSheet"
    best, best_score = None, -1
    for sh in xl.sheet_names:
        try:
            tmp = xl.parse(sh, header=None, nrows=15)
            det = detect_month_header(tmp)
            sc = len(det[1]) if det else 0
            if sc > best_score: best, best_score = sh, sc
        except Exception:
            continue
    return best or xl.sheet_names[0]

def _canonical_region_from_file(stem: str, df_head: pd.DataFrame) -> str:
    # 1) Пытаемся вытащить заголовок "Итого <Регион>" из тела файла
    try:
        c0 = df_head.iloc[1, 0]
        if isinstance(c0, str) and c0.strip().lower().startswith("итого"):
            reg = re.sub(r"^\s*итого\s+", "", c0.strip(), flags=re.IGNORECASE)
            return re.sub(r"\s{2,}", " ", reg).strip(" _-·.")
    except Exception:
        pass

    # 2) Чистим имя файла
    s = stem
    # убираем служебные слова
    s = re.sub(r"(?i)\b(итого|подразделени[яе]|расширенн\w*|данн\w*)\b", "", s)
    # убираем любые годы 20xx
    s = re.sub(r"\b20\d{2}\b", "", s)
    # убираем диапазоны месяцев вида "1-8", "01_08", "1–8"
    s = re.sub(r"\b\d{1,2}\s*[-–—_]\s*\d{1,2}\b", "", s)
    # убираем одиночные числовые хвосты
    s = re.sub(r"[ _\-–—]*\d+\b", "", s)
    # приводим пробелы и обрезаем мусор
    s = re.sub(r"\s{2,}", " ", s).strip(" _-·.")
    # кастомные нормализации
    if re.match(r"(?i)^(кк|краснодар)", s): s = "Краснодарский край"
    if re.fullmatch(r"(?i)санкт(?:-|\s*)петербург|санкт", s): s = "Санкт-Петербург"
    return s or stem

@st.cache_data(show_spinner="Читаю и разбираю файлы…")
def parse_excel(file_bytes: bytes, region_name: str, file_year: int | None = None) -> pd.DataFrame:
    def coerce_number(x) -> float:
        if x is None or (isinstance(x, float) and np.isnan(x)): return np.nan
        s = str(x).strip()
        if s in {"", "-", "—", "NA", "NaN", "nan"}: return np.nan
        if s.endswith("%"):
            try:
                s = s[:-1].strip().replace("\u00A0","").replace(" ","").replace(",", ".")
                return float(s)
            except (ValueError, TypeError):
                return np.nan
        s = s.replace("\u00A0","").replace(" ","")
        if s.count(",") == 1 and s.count(".") > 1: s = s.replace(".", "").replace(",", ".")
        elif s.count(",") > 1 and s.count(".") <= 1: s = s.replace(",", "")
        elif s.count(",") == 1 and s.count(".") == 0: s = s.replace(",", ".")
        try: return float(s)
        except (ValueError, TypeError): return np.nan

    xl = pd.ExcelFile(BytesIO(file_bytes))
    sheet = guess_data_sheet(xl)
    df = xl.parse(sheet, header=None)

    head = df.head(5)
    canonical_region = _canonical_region_from_file(Path(region_name).stem, head)
    first_c0 = next((str(x).strip() for x in df.iloc[:,0].dropna().tolist() if str(x).strip()), "")
    is_totals_file = bool(re.match(r"(?i)^\s*итого\b", first_c0))

    det = detect_month_header(df)
    if not det:
        raise ValueError(f"Не нашли строку месяцев на листе '{sheet}'.")
    header_row, month_cols = det
    month_map = {j: m for j, m in sorted(month_cols, key=lambda x: x[0])}
    month_indices = list(month_map.keys())
    first_month_col = min(month_indices)

    rows = []
    current_branch = ""
    last_cat = "Общее"
    for r in range(header_row + 1, len(df)):
        cell0 = df.iat[r, 0] if df.shape[1] > 0 else None
        if isinstance(cell0, str) and cell0.strip():
            current_branch = cell0.strip()

        metric_cell = None
        for c in range(first_month_col - 1, -1, -1):
            val = df.iat[r, c]
            if isinstance(val, str) and val.strip():
                metric_cell = val.strip()
                break
        if not metric_cell:
            continue

        metric_name = normalize_metric_name(metric_cell)

        # ⛔️ Пропускаем строки, если метрика не из белого списка
        if metric_name not in ACCEPTED_METRICS_CANONICAL:
            continue

        # НОВОЕ: собираем текст левых ячеек строки (все до первой колонки месяцев)
        left_cells = []
        for c in range(0, first_month_col):
            val = df.iat[r, c]
            if isinstance(val, str) and val.strip():
                left_cells.append(val.strip())
        left_blob = " ".join(left_cells).lower()

        # Базовая категория по названию метрики
        cat = detect_category(metric_cell)

        # Если метка не найдена в самой метрике — пытаемся определить по контексту строки
        if cat == "Общее":
            # 1) заголовок подразделения
            cat = detect_category(current_branch)
            # 2) текст в левой части строки
            if cat == "Общее":
                if re.search(r"\bн\s*ю\s*з\b|нюз", left_blob):
                    cat = "НЮЗ"
                elif re.search(r"\bю\s*з\b|юз", left_blob):
                    cat = "ЮЗ"
                else:
                    cat = last_cat  # 3) наследуем последнюю явную метку в блоке

        # Обновляем «липкую» метку, если нашли явную
        if cat in {"НЮЗ", "ЮЗ"}:
            last_cat = cat

        code_match = re.search(r"№\s*(\d+)", str(current_branch))
        code = code_match.group(1) if code_match else ""

        vals = [coerce_number(df.iat[r, j]) for j in month_indices]
        arr = np.array(vals, dtype=float)
        mask_fact = (~np.isnan(arr)) & (np.abs(arr) > 1e-12)
        if not mask_fact.any():
            continue

        first, last = int(np.where(mask_fact)[0][0]), int(np.where(mask_fact)[0][-1])
        for pos, j in enumerate(month_indices):
            v = vals[pos]
            if np.isnan(v) or ((pos < first or pos > last) and (abs(v) <= 1e-12)):
                continue
            rows.append({
                "Регион": str(canonical_region),
                "ИсточникФайла": "TOTALS_FILE" if is_totals_file else "BRANCHES_FILE",
                "Код": code,
                "Подразделение": str(current_branch),
                "Показатель": metric_name,
                "Месяц": month_map[j],
                "Значение": float(v),
                "Категория": cat,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Данные не распознаны.")

    out["Год"] = int(file_year) if file_year else pd.NA
    out["Месяц"] = pd.Categorical(out["Месяц"].astype(str), categories=ORDER_WITH_TOTAL, ordered=True)
    for c in ["Регион", "Подразделение", "Показатель", "Код", "ИсточникФайла", "Категория"]:
        out[c] = out[c].astype("string")
    return out

def apply_economic_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    # В упрощенном режиме эта функция не должна ничего делать, так как мы не создаем derived метрики
    if SIMPLE_MODE:
        return df

    out = df.copy()
    def div(a,b,scale=1.0):
        return (pd.to_numeric(out.get(a), errors="coerce") /
                pd.to_numeric(out.get(b), errors="coerce").replace(0, np.nan)) * scale

    if Metrics.AVG_LOAN.value not in out and {Metrics.LOAN_ISSUE.value, Metrics.LOAN_ISSUE_UNITS.value} <= set(out.columns):
        out[Metrics.AVG_LOAN.value] = div(Metrics.LOAN_ISSUE.value, Metrics.LOAN_ISSUE_UNITS.value)

    if Metrics.MARKUP_PCT.value not in out and {Metrics.MARKUP_AMOUNT.value, Metrics.REVENUE.value} <= set(out.columns):
        out[Metrics.MARKUP_PCT.value] = div(Metrics.MARKUP_AMOUNT.value, Metrics.REVENUE.value, 100.0)

    if Metrics.RISK_SHARE.value not in out and {Metrics.BELOW_LOAN.value, Metrics.REVENUE.value} <= set(out.columns):
        out[Metrics.RISK_SHARE.value] = div(Metrics.BELOW_LOAN.value, Metrics.REVENUE.value, 100.0)

    if Metrics.YIELD.value not in out and {Metrics.PENALTIES_RECEIVED.value, Metrics.LOAN_ISSUE.value} <= set(out.columns):
        out[Metrics.YIELD.value] = div(Metrics.PENALTIES_RECEIVED.value, Metrics.LOAN_ISSUE.value, 100.0)

    return out

def normalize_percent_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    clean = x.dropna()
    if not clean.empty and (clean.abs() <= 2).all(): x = x * 100.0
    return x

def strip_totals_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~df["Подразделение"].str.contains(r"^\s*итого\b", case=False, na=False)
    return df.loc[mask]

@st.cache_data
def get_aggregated_data(df_raw: pd.DataFrame, regions: Tuple[str, ...], months: Tuple[str, ...]) -> pd.DataFrame:
    df = strip_totals_rows(df_raw)
    sub = df[df["Регион"].isin(regions) & df["Месяц"].isin(months)]
    if sub.empty: return pd.DataFrame()
    all_entities_df = (sub[["Регион", "Подразделение"]].drop_duplicates().sort_values(by=["Регион", "Подразделение"]).set_index(["Регион", "Подразделение"]))

    # --- суммы ---
    df_sum = (sub[sub["Показатель"].isin(METRICS_SUM)]
              .groupby(["Регион","Подразделение","Показатель"], observed=True)["Значение"]
              .sum().unstack())
    df_sum = _flatten_columns(df_sum)
    result = all_entities_df.join(df_sum, how="left")

    # --- last (снимки на конец периода) ---
    if METRICS_LAST:
        snap = sub[sub["Показатель"].isin(METRICS_LAST)].copy()
        if not snap.empty:
            # Важно: 'Месяц' — упорядоченная категориальная, сортируем и берём последнее
            snap['Месяц'] = pd.Categorical(snap['Месяц'].astype(str), categories=ORDER, ordered=True)
            snap = snap.sort_values(["Регион","Подразделение","Показатель","Месяц"])
            df_last = (snap.groupby(["Регион","Подразделение","Показатель"], observed=True)
                            .tail(1)
                            .set_index(["Регион","Подразделение","Показатель"])["Значение"]
                            .unstack())
            df_last = _flatten_columns(df_last)
            result = result.join(df_last, how="left")

    # --- средние ---
    metrics_to_average = METRICS_MEAN - ({Metrics.RISK_SHARE.value} if not SIMPLE_MODE else set())

    if metrics_to_average:
        df_mean = (sub[sub["Показатель"].isin(metrics_to_average)]
                   .groupby(["Регион","Подразделение","Показатель"], observed=True)["Значение"]
                   .mean().unstack())
        df_mean = _flatten_columns(df_mean)
        for metric in metrics_to_average:
            if metric in df_mean.columns:
                result[metric] = df_mean[metric]

    result = apply_economic_derivatives(result) # Не будет ничего делать в SIMPLE_MODE

    return result.reset_index()


@st.cache_data
def get_monthly_pivoted_data(df_raw: pd.DataFrame, regions: Tuple[str, ...], months: Tuple[str, ...], raw_only: bool = False) -> pd.DataFrame:
    df = strip_totals_rows(df_raw)
    sub = df[df["Регион"].isin(regions) & df["Месяц"].isin(months)]
    if sub.empty: return pd.DataFrame()
    pivot = sub.pivot_table(index=["Регион","Подразделение","Месяц"], columns="Показатель", values="Значение", aggfunc="sum")
    pivot = _flatten_columns(pivot).reset_index()

    if not raw_only:
        pivot = apply_economic_derivatives(pivot)

    for col in [Metrics.ILLIQUID_BY_COUNT_PCT.value, Metrics.ILLIQUID_BY_VALUE_PCT.value, Metrics.YIELD.value, Metrics.MARKUP_PCT.value]:
        if col in pivot.columns: pivot[col] = normalize_percent_series(pivot[col])
    return pivot

@st.cache_data
def month_totals_matrix(df_raw: pd.DataFrame, regions: Tuple[str, ...], metric: str) -> pd.DataFrame:
    """
    Возвращает матрицу Регион × Месяц с помесячными значениями из строк «Итого по месяцу».
    Никаких сумм по филиалам — только то, что лежит в файле в строках «Итого».
    """
    dfm = get_monthly_totals_from_file(df_raw, regions, metric)
    if dfm.empty:
        return pd.DataFrame(columns=["Регион","Месяц","Значение"])
    return dfm.copy()

def _postprocess_monthly_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Регион","Месяц","Значение"])
    out = df.copy()
    keep = [c for c in out.columns if c in {"Регион","Месяц","Значение"}]
    out = out[keep]
    out["Регион"] = out["Регион"].astype("string")
    out["Месяц"] = out["Месяц"].astype(str)
    out = out[out["Месяц"].isin(ORDER)]
    out = (out.groupby(["Регион","Месяц"], as_index=False)["Значение"].sum())
    return out

@st.cache_data
def aggregate_for_dynamics_cached(df_raw: pd.DataFrame, regions: Tuple[str, ...], months: Tuple[str, ...], metric: str) -> pd.DataFrame:
    # В упрощенном режиме всегда берем данные из "Итого"
    from_file = get_monthly_totals_from_file(df_raw, regions, metric)
    if not from_file.empty:
        out = from_file[from_file["Месяц"].astype(str).isin(months)]
        if not out.empty:
            return _postprocess_monthly_df(out.rename(columns={"Значение": "Значение"}))
    return pd.DataFrame(columns=["Регион","Месяц","Значение"])


def number_column_config(title: str, money=False, percent=False, days=False):
    if money: return st.column_config.NumberColumn(f"{title}, ₽", help="Денежная сумма", format="%.0f")
    if percent: return st.column_config.NumberColumn(f"{title}, %", help="Проценты/доли", format="%.2f%%")
    if days: return st.column_config.NumberColumn(f"{title}, дн.", help="Дни", format="%.2f")
    return st.column_config.NumberColumn(title, format="%.2f")

def default_column_config(df: pd.DataFrame) -> dict:
    cfg = {}
    for c in df.columns:
        s = str(c)
        is_money = "руб" in s
        is_percent = s.endswith("(%)") or "наценк" in s.lower() or "доля" in s.lower() or s == Metrics.YIELD.value
        is_days = "дней" in s
        if pd.api.types.is_numeric_dtype(df[c]):
            cfg[s] = number_column_config(s, money=is_money, percent=is_percent, days=is_days)
    return cfg

def kpi_block(df_all: pd.DataFrame, regions: list[str], months_range: list[str], all_available_months: list[str], strict_mode: bool):
    st.markdown("## 📊 Ключевые показатели (KPI) <span class='badge'>Как в файле</span>", unsafe_allow_html=True)
    st.caption("Период считаем строго из строк «Итого по месяцу»: суммы — суммируем, проценты и средние — усредняем, снимки — берём последний месяц.")

    def _render_metric_if_value(col, title, value, *, kind="money", delta=None, delta_color="normal"):
        if value is None or pd.isna(value) or (isinstance(value, (int,float)) and abs(value) < 1e-12):
            return  # не рисуем карточку
        if kind == "money":
            txt = format_rub(value)
        elif kind == "pct":
            txt = fmt_pct(value)
        elif kind == "days":
            txt = fmt_days(value)
        else:
            txt = f"{value:,.0f}".replace(",", " ")
        col.metric(title, txt, delta=delta, delta_color=delta_color)

    sub_df = df_all[(df_all["Регион"].isin(regions)) & (df_all["Месяц"].astype(str).isin(months_range))]
    
    # какие метрики показываем в KPI-таблице по регионам
    KPI_SET_MONEY = [Metrics.REVENUE.value, Metrics.LOAN_ISSUE.value]
    KPI_SET_RATE  = [Metrics.MARKUP_PCT.value, Metrics.YIELD.value]
    KPI_COLUMNS   = KPI_SET_MONEY + KPI_SET_RATE  # можно расширить

    mode_view = st.radio("Отображение", ["По регионам", "Совокупно (по выборке)"], horizontal=True)

    if mode_view == "По регионам":
        regs_sorted = sorted(map(str, sub_df["Регион"].unique()))
        rows = []
        for reg in regs_sorted:
            r = {"Регион": reg}
            for m in KPI_COLUMNS:
                # для KPI по регионам для снимков берём среднее за период
                r[m] = period_value_from_itogo_for_region(df_all, reg, m, months_range, snapshots_mode="mean")
            rows.append(r)

        kpi_table = pd.DataFrame(rows).set_index("Регион")
        # сортировка по выручке
        sort_col = Metrics.REVENUE.value if Metrics.REVENUE.value in kpi_table.columns else kpi_table.columns[0]
        kpi_table = kpi_table.sort_values(by=sort_col, ascending=False)

        # конфиг числовых колонок (₽/%, без знаков после запятой для денег)
        cfg = {}
        for c in kpi_table.columns:
            if "руб" in c:
                cfg[c] = st.column_config.NumberColumn(f"{c}", format="%.0f")
            elif is_percent_metric(c):
                cfg[c] = st.column_config.NumberColumn(f"{c}", format="%.2f%%")
            elif "дней" in c:
                cfg[c] = st.column_config.NumberColumn(f"{c}", format="%.2f")
            else:
                cfg[c] = st.column_config.NumberColumn(f"{c}", format="%.0f")

        st.dataframe(kpi_table, use_container_width=True, column_config=cfg)
        st.caption("Период: средние значения для %-метрик и метрик-снимков; денежные/шт — суммируются за период.")
        return  # выходим, карточки ниже не рисуем в этом режиме

    if sub_df.empty:
        st.info("Нет данных по выбранным фильтрам.")
        return

    def pv(metric):
        return period_value_from_itogo(df_all, regions, metric, months_range)

    lbl_cur = f"{months_range[0]}–{months_range[-1]}" if len(months_range) > 1 else months_range[0]

    v_rev = pv(Metrics.REVENUE.value)
    v_issue = pv(Metrics.LOAN_ISSUE.value)
    v_markup_pct = pv(Metrics.MARKUP_PCT.value)
    v_yield = pv(Metrics.YIELD.value)
    v_avg_term = pv(Metrics.AVG_LOAN_TERM.value)
    v_avg_loan = pv(Metrics.AVG_LOAN.value)
    v_branches = pv(Metrics.BRANCH_COUNT.value)
    v_redeemed = pv(Metrics.REDEEMED_ITEMS_COUNT.value)

    cA, cB, cC, cD = st.columns(4)
    _render_metric_if_value(cA, f"Выручка ({lbl_cur})", v_rev, kind="money")
    _render_metric_if_value(cB, f"Выдано займов ({lbl_cur})", v_issue, kind="money")
    _render_metric_if_value(cC, f"Процент наценки ({lbl_cur})", v_markup_pct, kind="pct")
    _render_metric_if_value(cD, f"Доходность ({lbl_cur})", v_yield, kind="pct")

    st.markdown("<br>", unsafe_allow_html=True)
    cE, cF, cG, cH = st.columns(4)
    _render_metric_if_value(cE, f"Количество ломбардов ({lbl_cur})", v_branches, kind="num")

    if len(regions) == 1:
        _render_metric_if_value(cF, f"Средний срок займа ({lbl_cur})", v_avg_term, kind="days")
    else:
        per_reg = period_values_by_region_from_itogo(df_all, regions, Metrics.AVG_LOAN_TERM.value, months_range)
        if per_reg:
            with cF:
                st.markdown("**Средний срок займа (дней)**")
                st.dataframe(
                    pd.DataFrame(sorted(per_reg.items()), columns=["Регион", "дн."]).set_index("Регион"),
                    use_container_width=True,
                    column_config={"дн.": st.column_config.NumberColumn("дн.", format="%.2f")}
                )

    if len(regions) == 1:
        _render_metric_if_value(cG, f"Средняя сумма займа ({lbl_cur})", v_avg_loan, kind="money")
    else:
        per_reg_avg_loan = period_values_by_region_from_itogo(df_all, regions, Metrics.AVG_LOAN.value, months_range)
        if per_reg_avg_loan:
            with cG:
                st.markdown("**Средняя сумма займа (руб)**")
                st.dataframe(
                    pd.DataFrame(sorted(per_reg_avg_loan.items()), columns=["Регион", "₽"]).set_index("Регион"),
                    use_container_width=True,
                    column_config={"₽": st.column_config.NumberColumn("₽", format="%.0f")}
                )

    _render_metric_if_value(cH, f"Выкупленные залоги ({lbl_cur})", v_redeemed, kind="num")

    with st.expander("🧮 Как посчитано (правила агрегации)"):
        raw_metric_names = set(sub_df["Показатель"].dropna().unique())
        rows = []
        for m in sorted(raw_metric_names):
            rows.append({"Показатель": m, "Правило": aggregation_rule(m)})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("Нет показателей для отображения правил.")

def summary_block(agg_data, df_all, regions, months_range, all_available_months, strict_mode):
    st.subheader("📋 Сводка по регионам за период")
    st.caption("Все значения — ровно из строк «Итого по месяцу» в загруженных файлах. Никаких формул.")

    sub = df_all[df_all["Регион"].isin(regions)]
    raw_metrics = sorted(sub["Показатель"].dropna().unique())

    # соберём таблицу: строки — регионы; столбцы — метрики
    rows = []
    for reg in sorted(map(str, sub["Регион"].unique())):
        row = {"Регион": reg}
        for m in raw_metrics:
            row[m] = period_value_from_itogo_for_region(df_all, reg, m, months_range, snapshots_mode="mean")
        rows.append(row)

    if not rows:
        st.info("Нет данных для сводки по регионам.")
        return

    region_summary = pd.DataFrame(rows).set_index("Регион")

    # сортировка по выручке, если есть
    if Metrics.REVENUE.value in region_summary.columns:
        region_summary = region_summary.sort_values(by=Metrics.REVENUE.value, ascending=False)

    st.dataframe(
        region_summary,
        width="stretch",
        column_config=default_column_config(region_summary)
    )

def nuz_active_branches(df_all: pd.DataFrame,
                        regions: list[str] | Tuple[str, ...],
                        months: list[str]) -> pd.DataFrame:
    """Возвращает DataFrame с колонками Регион/Подразделение только для филиалов,
    где есть ненулевые значения по НЮЗ-метрикам в выбранном окне."""
    sub = strip_totals_rows(df_all)
    sub = sub[(sub["Регион"].isin(regions)) & (sub["Месяц"].astype(str).isin(months))]
    if sub.empty:
        return pd.DataFrame(columns=["Регион","Подразделение"])
    nuz = sub[sub["Показатель"].isin(NUZ_ACTIVITY_METRICS)].copy()
    nuz["Значение"] = pd.to_numeric(nuz["Значение"], errors="coerce").fillna(0).abs()
    act = (nuz.groupby(["Регион","Подразделение"], observed=True)["Значение"]
              .sum().reset_index())
    return act[act["Значение"] > 0][["Регион","Подразделение"]]

def leaderboard_block(df_all: pd.DataFrame, regions: list[str], available_months: list[str]):
    st.subheader("🏆 Лидеры и аутсайдеры")
    st.caption("Рейтинг филиалов по выбранной метрике из файла за период.")

    if df_all.empty or not available_months:
        st.info("Нет данных для отображения.")
        return

    only_nuz = st.checkbox(
        "Показывать только филиалы с активностью НЮЗ", value=True,
        help="Оставляет филиалы, где в выбранном периоде есть выдачи/выручка/шт по НЮЗ"
    )

    last_quarter = available_months[max(0, len(available_months)-3):]
    start_m, end_m = st.select_slider("Выберите период для рейтинга:", options=available_months, value=(last_quarter[0], last_quarter[-1]))
    leaderboard_months = ORDER[ORDER.index(start_m): ORDER.index(end_m) + 1]

    agg_data = get_aggregated_data(df_all, tuple(regions), tuple(leaderboard_months))
    if agg_data.empty:
        st.warning("Нет данных за выбранный период.")
        return

    if only_nuz:
        activity_cols = [Metrics.REVENUE.value, Metrics.LOAN_ISSUE.value, Metrics.LOAN_ISSUE_UNITS.value]
        have = [c for c in activity_cols if c in agg_data.columns and c in df_all["Показатель"].unique()]
        if have:
            mask = (agg_data[have].fillna(0).sum(axis=1) > 0)
            agg_data = agg_data[mask]
    if agg_data.empty:
        st.warning("После фильтрации по активности данных не осталось.")
        return

    # Сформируем пул допустимых метрик только из тех, что есть в файле
    raw_metric_names = set(df_all["Показатель"].dropna().unique())
    numeric_cols = [c for c in agg_data.columns if pd.api.types.is_numeric_dtype(agg_data[c]) and c != "Код"]
    metric_options = sorted([c for c in numeric_cols if c in raw_metric_names])

    if not metric_options:
        st.warning("В исходных файлах не найдено числовых метрик для рейтинга.")
        return

    chosen_metric = st.selectbox(
        "Показатель",
        options=metric_options,
        index=metric_options.index(Metrics.REVENUE.value) if Metrics.REVENUE.value in metric_options else 0
    )
    st.caption(METRIC_HELP.get(chosen_metric, ""))

    # определим правило «чем больше — тем лучше»
    if chosen_metric in METRICS_BIGGER_IS_BETTER:
        ascending = False
        title_best, title_worst = "✅ Топ-5 лучших", "❌ Топ-5 худших"
    elif chosen_metric in METRICS_SMALLER_IS_BETTER:
        ascending = True
        title_best, title_worst = "✅ Топ-5 лучших (меньше = лучше)", "❌ Топ-5 худших (больше = хуже)"
    else:
        ascending = False
        title_best, title_worst = "🔝 Топ-5 наибольших значений", "🔚 Топ-5 наименьших значений"

    sorted_data = agg_data.dropna(subset=[chosen_metric]).sort_values(by=chosen_metric, ascending=ascending)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{title_best} по _{chosen_metric}_**")
        st.dataframe(sorted_data.head(5)[["Подразделение","Регион",chosen_metric]], width="stretch", column_config=default_column_config(sorted_data))
    with c2:
        st.markdown(f"**{title_worst} по _{chosen_metric}_**")
        worst5 = sorted_data.tail(5)
        worst5 = worst5.iloc[::-1].copy()
        st.dataframe(worst5[["Подразделение","Регион",chosen_metric]], width="stretch", column_config=default_column_config(sorted_data))


def comparison_block(df_all: pd.DataFrame, regions: list[str], available_months: list[str]):
    st.subheader("⚖️ Сравнение периодов")
    st.caption("Сравнение филиалов по метрикам из файла за два разных периода.")
    if df_all.empty or not available_months: st.info("Нет данных для сравнения."); return
    c1, c2 = st.columns(2)
    with c1: start_a, end_a = st.select_slider("Период A (базовый):", options=available_months, value=(available_months[0], available_months[0]))
    with c2: start_b, end_b = st.select_slider("Период B (сравниваемый):", options=available_months, value=(available_months[-1], available_months[-1]))
    months_a = ORDER[ORDER.index(start_a): ORDER.index(end_a)+1]
    months_b = ORDER[ORDER.index(start_b): ORDER.index(end_b)+1]
    data_a = get_aggregated_data(df_all, tuple(regions), tuple(months_a))
    data_b = get_aggregated_data(df_all, tuple(regions), tuple(months_b))
    if data_a.empty or data_b.empty: st.warning("Нет данных для одного или обоих периодов."); return
    comparison_df = pd.merge(data_a, data_b, on=["Регион","Подразделение"], how="outer", suffixes=("_A","_B"))

    raw_metric_names = set(df_all["Показатель"].dropna().unique())
    all_metrics = sorted([c for c in data_a.columns if pd.api.types.is_numeric_dtype(data_a[c]) and c != "Код"])
    metric_options = [m for m in all_metrics if m in raw_metric_names]
    if not metric_options:
        st.warning("Нет метрик из файла для сравнения.")
        return

    chosen_metric = st.selectbox("Показатель для анализа:", options=metric_options, index=0, help=METRIC_HELP.get(metric_options[0], ""))
    col_a, col_b = f"{chosen_metric}_A", f"{chosen_metric}_B"
    if col_a not in comparison_df.columns: comparison_df[col_a] = np.nan
    if col_b not in comparison_df.columns: comparison_df[col_b] = np.nan
    comparison_df["Абсолютное изменение"] = comparison_df[col_b] - comparison_df[col_a]
    comparison_df["Относительное изменение, %"] = (comparison_df["Абсолютное изменение"] / comparison_df[col_a].replace(0, np.nan)) * 100
    is_money, is_percent, is_days = "руб" in chosen_metric, "%" in chosen_metric or "наценк" in chosen_metric.lower() or "доля" in chosen_metric.lower() or chosen_metric == Metrics.YIELD.value, "дней" in chosen_metric
    cfg = {
        col_a: number_column_config(f"{chosen_metric} (A: {start_a}-{end_a})", money=is_money, percent=is_percent, days=is_days),
        col_b: number_column_config(f"{chosen_metric} (B: {start_b}-{end_b})", money=is_money, percent=is_percent, days=is_days),
        "Абсолютное изменение": number_column_config("Изм. (абс.)", money=is_money and not is_percent and not is_days, percent=False, days=False),
        "Относительное изменение, %": st.column_config.NumberColumn("Изм. (%)", format="%.1f%%"),
    }
    st.dataframe(comparison_df[["Подразделение","Регион",col_a,col_b,"Абсолютное изменение","Относительное изменение, %"]].sort_values("Абсолютное изменение", ascending=False).dropna(subset=["Абсолютное изменение"]), width="stretch", column_config=cfg)
    st.info("**На что обратить внимание:** Ищите строки, где относительное изменение существенно отличается от нуля. Если какой-то филиал показал значительный рост доли ниже займа или падение доходности, это требует изучения причин (возможно, ухудшение качества залогов или изменение поведения клиентов).")

def dynamics_block(df_all: pd.DataFrame, regions: list[str], months_range: list[str], color_map: Dict[str, str]):
    st.subheader("📈 Динамика по регионам")

    raw_metric_names = sorted(set(df_all["Показатель"].dropna().unique()))
    if not raw_metric_names:
        st.warning("В файлах не найдено метрик для построения динамики.")
        return

    default_metrics = [m for m in [Metrics.REVENUE.value, Metrics.LOAN_ISSUE.value, Metrics.MARKUP_PCT.value] if m in raw_metric_names]
    metrics = st.multiselect("Показатели", options=raw_metric_names, default=default_metrics[:3])

    if not metrics:
        st.info("Выберите метрики.");
        return

    c1, c2, c3 = st.columns(3)
    only_actual, show_trend, fast_plot = c1.checkbox("Только фактические месяцы", True), c2.checkbox("Линия тренда", True), c3.checkbox("Облегчить отрисовку", False)

    for met in metrics:
        gp = get_monthly_totals_from_file(df_all, tuple(regions), met)
        gp = gp[gp["Месяц"].astype(str).isin(months_range)]
        if gp.empty:
            st.info(f"Нет данных по «{met}».");
            continue

        x_domain = months_range if not only_actual else sorted_months_safe(gp["Месяц"])
        if not x_domain:
            st.info(f"Нет фактических месяцев для «{met}»."); continue
        fig = go.Figure()

        # порядок регионов: как выбран пользователем в сайдбаре (если хотите — алфавит)
        region_order = [r for r in regions if r in gp["Регион"].astype(str).unique()]
        # резерв: добавим те, которых нет в выбранном списке (на случай фильтров)
        region_order += [r for r in sorted(gp["Регион"].astype(str).unique()) if r not in region_order]

        # чтобы легенда не переворачивала порядок
        fig.update_layout(
            legend=dict(traceorder="normal"),
            hovermode="x unified",
        )

        tickfmt, suf = y_fmt_for_metric(met)
        def fmt_hover(v):
            if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
            if tickfmt == ",.0f": return f"{v:,.0f}{suf}".replace(",", " ")
            if tickfmt == ".2f": return f"{v:.2f}{suf}"
            return f"{v:,.2f}{suf}".replace(",", " ")

        hovertemplate = (
            "<b>Регион: %{customdata[0]}</b><br>"
            "Месяц: %{x}<br>"
            "Значение: %{customdata[1]}<extra></extra>"
        )

        any_drawn = False
        for rank, reg in enumerate(region_order):
            t = gp[gp["Регион"].astype(str) == reg].groupby("Месяц", as_index=False)["Значение"].sum()
            series = t.set_index("Месяц")["Значение"].reindex(x_domain)
            if series.isna().all(): 
                continue
            any_drawn = True

            series_vals = series.values.astype(float)
            hover_vals  = [fmt_hover(v) for v in series_vals]

            fig.add_trace(go.Scatter(
                x=series.index, y=series_vals,
                mode="lines" if fast_plot else "lines+markers",
                name=reg,
                connectgaps=False,
                line=dict(color=color_map.get(reg)),
                legendgroup=reg,
                legendrank=rank,                  # ⬅️ порядок в легенде/ховере
                customdata=np.column_stack([np.full(len(series), reg), hover_vals]),
                hovertemplate=hovertemplate
            ))

            if (not fast_plot) and show_trend:
                y_vals = series.values.astype(float); mask = ~np.isnan(y_vals)
                if mask.sum() >= 2:
                    x_pos = np.arange(len(series.index))
                    k, b = np.polyfit(x_pos[mask], y_vals[mask], 1)
                    fig.add_trace(go.Scatter(
                        x=series.index, y=k * x_pos + b,
                        mode="lines",
                        name=f"{reg} · тренд",
                        line=dict(dash="dot", width=2, color=color_map.get(reg)),
                        showlegend=False,
                        hoverinfo="skip",           # ⬅️ тренды в ховер не попадают
                        legendgroup=reg,
                        legendrank=rank
                    ))
        if not any_drawn:
            st.info(f"Для «{met}» данные есть, но после выравнивания по календарю все серии пустые (разные месяцы у источников). Выберите «Только фактические месяцы» или сузьте период.")
            continue

        rule = aggregation_rule(met)
        rule_text = 'Сумма' if rule=='sum' else 'Среднее' if rule=='mean' else 'Последний месяц'
        subtitle = f"Источник: строки «Итого по месяцу». Агрегация за период: {rule_text}."
        fig.update_layout(title={'text': f"{met}<br><sup>{subtitle}</sup>", 'x':0}, hovermode="x unified", margin=dict(t=70,l=0,r=0,b=0))
        fig.update_yaxes(tickformat=tickfmt, ticksuffix=suf.strip(), title_text=suf.strip() or None)

        st.plotly_chart(fig, use_container_width=True)
        with st.expander(f"Данные для графика «{met}»"):
            st.dataframe(gp.pivot_table(index="Месяц", columns="Регион", values="Значение", aggfunc="sum").reindex(x_domain), use_container_width=True)


def dynamics_compare_block(df_a: pd.DataFrame, df_b: pd.DataFrame,
                           regions: list[str], months_range: list[str],
                           color_map: Dict[str, str], year_a: int, year_b: int):
    st.subheader(f"📈 Динамика по регионам: {year_b} vs {year_a}")

    raw_metric_names = sorted(set(pd.concat([df_a, df_b])["Показатель"].dropna().unique()))

    default_selection = [m for m in [Metrics.REVENUE.value, Metrics.LOAN_ISSUE.value, Metrics.MARKUP_PCT.value] if m in raw_metric_names]

    metrics = st.multiselect("Показатели", options=raw_metric_names,
                             default=default_selection, key="dyn_cmp_metrics")
    c1, c2 = st.columns(2)
    only_actual = c1.checkbox("Только фактические месяцы", True, key="dyn_cmp_actual")
    show_trend = c2.checkbox("Линия тренда", False, key="dyn_cmp_trend")
    if not metrics or not months_range:
        st.info("Выберите метрики и период."); return

    for met in metrics:
        rule = aggregation_rule(met)
        rule_text = 'Сумма' if rule=='sum' else 'Среднее' if rule=='mean' else 'Последний месяц'
        st.caption(f"Данные — из строк «Итого по месяцу» исходных файлов. Агрегация за период: **{rule_text}**.")

        gp_a = get_monthly_totals_from_file(df_a, tuple(regions), met)
        gp_b = get_monthly_totals_from_file(df_b, tuple(regions), met)
        if gp_a.empty and gp_b.empty:
            st.info(f"Нет данных по «{met}»."); continue

        xA = sorted_months_safe(gp_a["Месяц"]) if not gp_a.empty else []
        xB = sorted_months_safe(gp_b["Месяц"]) if not gp_b.empty else []
        x_domain = [m for m in months_range if (m in xA or m in xB)] if only_actual else months_range
        if not x_domain:
            st.info(f"Нет фактических месяцев для «{met}»."); continue

        fig = go.Figure()
        fig.update_layout(
            hovermode="x unified",
            legend=dict(
                traceorder="normal",   # не переворачивать
                tracegroupgap=10       # визуально разделять регионы
            )
        )

        def clean_region_label(reg: str) -> str:
            s = str(reg)
            s = re.sub(r"\b20\d{2}\b", "", s)         # выкинуть «2024/2025» из имени региона
            s = re.sub(r"\b\d{1,2}\s*-\s*\d{1,2}\b", "", s)  # «1-8»
            s = re.sub(r"[_\.\-–—]", " ", s)
            s = re.sub(r"\s{2,}", " ", s).strip()
            return s

        all_regs = set(gp_a["Регион"].astype(str)).union(set(gp_b["Регион"].astype(str)))
        region_order = [r for r in regions if r in all_regs] + [r for r in sorted(all_regs) if r not in regions]
        label_map = {r: clean_region_label(r) for r in all_regs}

        year_order = [year_a, year_b]          # порядок и в легенде, и в ховере
        dash_map   = {year_a: "dot", year_b: "solid"}  # «база» = пунктир, «сравнение» = сплошная

        tickfmt, suf = y_fmt_for_metric(met)
        def fmt_hover(v):
            if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
            if tickfmt == ",.0f": return f"{v:,.0f}{suf}".replace(",", " ")
            if tickfmt == ".2f": return f"{v:.2f}{suf}"
            return f"{v:,.2f}{suf}".replace(",", " ")

        hovertemplate = (
            "<b>%{customdata[0]}</b><br>"
            "Год: %{customdata[1]}<br>"
            "Месяц: %{x}<br>"
            "Значение: %{customdata[2]}<extra></extra>"
        )

        for r_rank, reg in enumerate(region_order):
            for y in year_order:
                gp = gp_a if y == year_a else gp_b
                if gp.empty: 
                    continue
                t = gp[gp["Регион"].astype(str) == reg].groupby("Месяц", as_index=False)["Значение"].sum()
                s = t.set_index("Месяц")["Значение"].reindex(x_domain)
                if s.isna().all():
                    continue

                vals = s.values.astype(float)
                fig.add_trace(go.Scatter(
                    x=s.index,
                    y=vals,
                    mode="lines+markers",
                    name=f"{label_map.get(reg, reg)} · {y}",
                    line=dict(color=color_map.get(reg), dash=dash_map[y]),
                    legendgroup=label_map.get(reg, reg),     # группируем легендой по региону
                    legendrank=r_rank * 10 + (0 if y == year_a else 1),  # стабильный порядок
                    customdata=np.column_stack([
                        np.full(len(s), label_map.get(reg, reg)),
                        np.full(len(s), y),
                        [fmt_hover(v) for v in vals]
                    ]),
                    hovertemplate=hovertemplate
                ))

                if show_trend:
                    mask = ~np.isnan(vals)
                    if mask.sum() >= 2:
                        xp = np.arange(len(vals))
                        k, b = np.polyfit(xp[mask], vals[mask], 1)
                        fig.add_trace(go.Scatter(
                            x=s.index, y=k * xp + b,
                            mode="lines",
                            line=dict(color=color_map.get(reg), dash="dash"),
                            name=f"{label_map.get(reg, reg)} · тренд · {y}",
                            showlegend=False,
                            hoverinfo="skip",                # тренд не попадает в ховер
                            legendgroup=label_map.get(reg, reg),
                            legendrank=r_rank * 10 + (0 if y == year_a else 1)
                        ))
        subtitle = f"Источник: строки «Итого по месяцу». Агрегация за период: {rule_text}."
        fig.update_layout(title={'text': f"{met}<br><sup>{subtitle}</sup>", 'x': 0},
                          hovermode="x unified", margin=dict(t=70, l=0, r=0, b=0))
        fig.update_yaxes(tickformat=tickfmt, ticksuffix=suf.strip(), title_text=suf.strip() or None)
        st.plotly_chart(fig, use_container_width=True)

def _aggregate_period(df_year: pd.DataFrame, regions: list[str], metric: str, months: list[str]) -> dict[str, float]:
    """Агрегируем строго из строк 'Итого по месяцу' по правилу метрики."""
    dfm = get_monthly_totals_from_file(df_year, tuple(regions), metric)
    if dfm.empty:
        return {}
    part = dfm[dfm["Месяц"].astype(str).isin(months)].copy()
    if part.empty:
        return {}
    rule = aggregation_rule(metric)
    out = {}
    for reg, g in part.groupby("Регион"):
        vals = pd.to_numeric(g["Значение"], errors="coerce").dropna()
        if vals.empty:
            continue
        if rule == "sum":
            out[str(reg)] = float(vals.sum())
        elif rule == "mean":
            out[str(reg)] = float(vals.mean())
        elif rule == "last":
            out[str(reg)] = float(vals.iloc[-1])
        else:
            out[str(reg)] = float(vals.mean())
    return out

def yoy_summary_block(df_a: pd.DataFrame, df_b: pd.DataFrame, regions: list[str], months_range: list[str], years: tuple[int,int]):
    st.subheader("📊 Итоги по метрикам: год B vs год A")
    raw_metric_names = sorted(set(pd.concat([df_a, df_b])["Показатель"].dropna().unique()))
    if not raw_metric_names:
        st.info("Нет метрик для сравнения.")
        return
    metric = st.selectbox("Показатель", options=raw_metric_names,
                          index=(raw_metric_names.index(Metrics.REVENUE.value) if Metrics.REVENUE.value in raw_metric_names else 0))
    yA, yB = years
    a = _aggregate_period(df_a, regions, metric, months_range)
    b = _aggregate_period(df_b, regions, metric, months_range)

    regs = sorted(set(a) | set(b))
    rows = []
    for r in regs:
        vA = a.get(r, np.nan)
        vB = b.get(r, np.nan)
        d  = vB - vA if pd.notna(vA) and pd.notna(vB) else np.nan
        dp = (d / vA * 100.0) if pd.notna(d) and pd.notna(vA) and vA != 0 else np.nan
        rows.append({"Регион": r, f"{metric} · {yA}": vA, f"{metric} · {yB}": vB,
                     "Δ (абс.)": d, "Δ (%)": dp})

    if not rows:
        st.info("Нет данных для выбранного периода.")
        return

    df = pd.DataFrame(rows).set_index("Регион")
    is_money  = "руб" in metric
    is_pct    = is_percent_metric(metric)
    is_days   = "дней" in metric
    cfg = {
        f"{metric} · {yA}": number_column_config(f"{metric} · {yA}", money=is_money, percent=is_pct, days=is_days),
        f"{metric} · {yB}": number_column_config(f"{metric} · {yB}", money=is_money, percent=is_pct, days=is_days),
        "Δ (абс.)": number_column_config("Δ (абс.)", money=is_money and not is_pct and not is_days),
        "Δ (%)": st.column_config.NumberColumn("Δ (%)", format="%.1f%%"),
    }
    # сортируем по абсолютной дельте по убыванию
    df = df.sort_values(by="Δ (абс.)", ascending=False)
    st.dataframe(df, use_container_width=True, column_config=cfg)

    st.markdown("—")
    st.markdown("**Δ по месяцам (год B − год A)**")
    # берём помесячные «Итого по месяцу» и строим разницу
    mA = month_totals_matrix(df_a, tuple(regions), metric)
    mB = month_totals_matrix(df_b, tuple(regions), metric)
    if mA.empty and mB.empty:
        st.info("Нет данных для теплокарты дельт.")
        return
    # оставляем только выбранные месяцы
    mA = mA[mA["Месяц"].astype(str).isin(months_range)]
    mB = mB[mB["Месяц"].astype(str).isin(months_range)]
    pA = mA.pivot_table(index="Регион", columns="Месяц", values="Значение", aggfunc="first")
    pB = mB.pivot_table(index="Регион", columns="Месяц", values="Значение", aggfunc="first")
    delta = (pB - pA).reindex(columns=months_range)
    if delta.empty:
        st.info("Нет пересечения месяцев для расчёта дельт.")
    else:
        fig = px.imshow(delta, text_auto=".1f", aspect="auto", color_continuous_scale="RdBu", origin="upper",
                        title=f"Δ {metric}: {yB} − {yA}")
        # понятный ховер
        if "руб" in metric:
            fig.update_traces(hovertemplate="Регион: %{y}<br>Месяц: %{x}<br>Δ: %{z:,.0f} ₽<extra></extra>".replace(",", " "))
        elif is_percent_metric(metric):
            fig.update_traces(hovertemplate="Регион: %{y}<br>Месяц: %{x}<br>Δ: %{z:.2f}%<extra></extra>")
        elif "дней" in metric:
            fig.update_traces(hovertemplate="Регион: %{y}<br>Месяц: %{x}<br>Δ: %{z:.2f} дн.<extra></extra>")
        else:
            fig.update_traces(hovertemplate="Регион: %{y}<br>Месяц: %{x}<br>Δ: %{z:,.0f}<extra></extra>".replace(",", " "))
        fig.update_layout(margin=dict(t=40, l=0, r=0, b=0), coloraxis_colorbar_title="Δ")
        st.plotly_chart(fig, use_container_width=True)

def treemap_heatmap_block(df_all: pd.DataFrame, regions: list[str], months_range: list[str], color_map: Dict[str, str]):
    st.subheader("🗺️ Структура и распределение по месяцам")
    st.caption("Treemap — вклад филиалов; теплокарта — помесячные значения. Используются только метрики из файлов.")

    sub = strip_totals_rows(df_all)
    sub = sub[(sub["Регион"].isin(regions)) & (sub["Месяц"].astype(str).isin(months_range))]
    if sub.empty:
        st.info("Нет данных."); return

    raw_metric_names = sorted(set(df_all["Показатель"].dropna().unique()))
    if not raw_metric_names:
        st.warning("Нет метрик для отображения.")
        return

    # ==== Структура (Treemap) ====
    st.markdown("**Структура по подразделениям (за выбранный месяц)**")
    metric = st.selectbox(
        "Метрика для площади",
        options=raw_metric_names,
        index=(raw_metric_names.index(Metrics.REVENUE.value) if Metrics.REVENUE.value in raw_metric_names else 0),
        help=METRIC_HELP.get(Metrics.REVENUE.value, "")
    )

    months_present = sorted_months_safe(sub["Месяц"])
    if not months_present:
        st.info("Нет данных за период."); return
    month_for_tree = st.selectbox("Месяц для структуры", options=months_present, index=len(months_present)-1)

    tree_base = sub[(sub["Показатель"] == metric) & (sub["Месяц"].astype(str) == month_for_tree)]
    tree_data = (tree_base.groupby(["Регион","Подразделение"], observed=True)["Значение"]
                       .sum().reset_index().rename(columns={"Значение": "Size"}))

    if not tree_data.empty and pd.to_numeric(tree_data["Size"], errors="coerce").fillna(0).abs().sum() > 0:
        fig_t = px.treemap(
            tree_data, path=[px.Constant("Все"), "Регион", "Подразделение"],
            values="Size", color="Регион", color_discrete_map=color_map
        )
        if "руб" in metric:
            fig_t.update_traces(texttemplate="%{label}<br>%{value:,.0f}".replace(",", " "),
                                hovertemplate="%{label}<br>Значение: %{value:,.0f} ₽".replace(",", " "))
        elif is_percent_metric(metric):
            fig_t.update_traces(texttemplate="%{label}<br>%{value:.2f}%",
                                hovertemplate="%{label}<br>Значение: %{value:.2f}%")
        else:
            fig_t.update_traces(texttemplate="%{label}<br>%{value:,.0f}".replace(",", " "),
                                hovertemplate="%{label}<br>Значение: %{value:,.0f}".replace(",", " "))
        fig_t.update_layout(margin=dict(t=40,l=0,r=0,b=0), title=f"Структура: {metric} · {month_for_tree}")
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("Недостаточно данных для структуры.")

    st.divider()

    # ==== Теплокарта ====
    st.markdown("**Распределение по месяцам (тепловая карта)**")
    st.caption("Используем значения из строк «Итого по месяцу». Для метрик-снимков (задолженность, кол-во ломбардов) ничего не суммируем — показываем месячный снимок.")

    heat_metric = st.selectbox(
        "Метрика для теплокарты",
        options=raw_metric_names,
        index=raw_metric_names.index(metric) if metric in raw_metric_names else 0,
        help=METRIC_HELP.get(metric, "")
    )
    by_subdiv = st.checkbox(
        "Показывать по подразделениям (в строках)",
        value=False,
        help="По умолчанию — региональные «Итого по месяцу». В режиме по подразделениям для метрик-снимков берём последнее значение месяца для каждого филиала."
    )

    if by_subdiv:
        df_loc = sub[sub["Показатель"] == heat_metric].copy()
        df_loc["RowLabel"] = df_loc["Регион"].astype(str) + " · " + df_loc["Подразделение"].astype(str)
        df_loc['Месяц'] = pd.Categorical(df_loc['Месяц'].astype(str), categories=ORDER, ordered=True)
        
        df_loc['__prio__'] = np.where(df_loc.get("ИсточникФайла", pd.Series(index=df_loc.index, dtype=object)).eq("TOTALS_FILE"), 1, 2)
        df_loc.sort_values(["RowLabel", "Месяц", "__prio__"], inplace=True)

        # правило агрегации
        rule = agg_of_metric(heat_metric)
        if heat_metric in METRICS_LAST or rule == "last":
            # снимок: никаких сумм — оставляем одно (первое после сортировки по приоритету) значение на (филиал, месяц)
            df_loc = df_loc.drop_duplicates(["RowLabel", "Месяц"], keep="first")
            aggfunc = "first"
        else:
            aggfunc = "sum" if rule == "sum" else "mean"

        hm = df_loc.pivot_table(index="RowLabel", columns="Месяц", values="Значение", aggfunc=aggfunc)
    else:
        # региональный уровень: уже берём ровно то, что в «Итого по месяцу»
        mat = month_totals_matrix(df_all, tuple(regions), heat_metric)
        hm = mat.pivot_table(index="Регион", columns="Месяц", values="Значение", aggfunc="first")

    if 'hm' in locals() and not hm.empty:
        hm = hm.reindex(columns=[m for m in months_range if m in hm.columns])
        hm = hm.loc[hm.mean(axis=1, numeric_only=True).sort_values(ascending=False).index]

        text_fmt = ".2f" if is_percent_metric(heat_metric) else ".0f"
        fig_h = px.imshow(hm, text_auto=text_fmt, aspect="auto", color_continuous_scale="RdYlGn",
                        title=f"Тепловая карта: {heat_metric}")

        # понятный ховер с единицами
        if "руб" in heat_metric:
            fig_h.update_traces(hovertemplate="Строка: %{y}<br>Месяц: %{x}<br>Значение: %{z:,.0f} ₽<extra></extra>".replace(",", " "))
        elif is_percent_metric(heat_metric):
            fig_h.update_traces(hovertemplate="Строка: %{y}<br>Месяц: %{x}<br>Значение: %{z:.2f}%<extra></extra>")
        elif "дней" in heat_metric:
            fig_h.update_traces(hovertemplate="Строка: %{y}<br>Месяц: %{x}<br>Значение: %{z:.2f} дн.<extra></extra>")
        else:
            fig_h.update_traces(hovertemplate="Строка: %{y}<br>Месяц: %{x}<br>Значение: %{z:,.0f}<extra></extra>".replace(",", " "))

        fig_h.update_layout(margin=dict(t=40,l=0,r=0,b=0), coloraxis_colorbar_title="Интенсивность")
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("Недостаточно данных для теплокарты.")


def scatter_plot_block(df_all: pd.DataFrame, monthly_data: pd.DataFrame, color_map: Dict[str, str]):
    st.subheader("🔬 Взаимосвязи (точечный график)")
    st.caption("Каждая точка — это одно подразделение в один из месяцев. Помогает найти скрытые зависимости между показателями.")
    if monthly_data.empty: st.info("Нет данных."); return
    with st.expander("Как это использовать?", expanded=False):
        st.info("""
            Попробуйте сравнить разные метрики. Например, выберите **X = Товар проданный ниже суммы займа НЮЗ (руб)** и **Y = Процент наценки НЮЗ**.
            Вы можете увидеть зависимость между этими показателями.
        """)
    available_months = sorted_months_safe(monthly_data['Месяц'])
    selected_months = st.multiselect("Месяцы:", options=available_months, default=available_months)
    if not selected_months: st.warning("Выберите хотя бы один месяц."); return
    plot_data = monthly_data[monthly_data['Месяц'].isin(selected_months)].copy()

    raw_metric_names = set(df_all["Показатель"].dropna().unique())
    metric_options = sorted([c for c in plot_data.columns if c in raw_metric_names])

    if len(metric_options) < 2: st.warning("Нужно как минимум 2 метрики из файла."); return
    c1, c2, c3 = st.columns(3)
    default_x = Metrics.BELOW_LOAN.value if Metrics.BELOW_LOAN.value in metric_options else metric_options[0]
    default_y = Metrics.MARKUP_PCT.value if Metrics.MARKUP_PCT.value in metric_options else metric_options[min(1, len(metric_options)-1)]
    default_size = Metrics.REVENUE.value if Metrics.REVENUE.value in metric_options else None
    x_axis = c1.selectbox("Ось X", options=metric_options, index=metric_options.index(default_x) if default_x in metric_options else 0, key="scatter_x")
    y_axis = c2.selectbox("Ось Y", options=metric_options, index=metric_options.index(default_y) if default_y in metric_options else 0, key="scatter_y")
    size_col = c3.selectbox("Размер точек", options=[None] + metric_options, index=([None] + metric_options).index(default_size) if default_size in metric_options else 0, key="scatter_size")

    if x_axis == y_axis: st.error("Выберите разные метрики для осей."); return
    df2 = plot_data.dropna(subset=[x_axis, y_axis]).copy()
    size_arg = "_size_" if size_col and pd.to_numeric(df2[size_col], errors="coerce").fillna(0).gt(0).any() else None
    if size_arg: df2[size_arg] = pd.to_numeric(df2[size_col], errors="coerce").fillna(0)
    df2['hover_text'] = df2['Подразделение'].astype(str) + " (" + df2['Месяц'].astype(str) + ")"
    fig = px.scatter(
        df2, x=x_axis, y=y_axis, color="Регион", size=size_arg, size_max=40,
        hover_name="hover_text", color_discrete_map=color_map,
        title=f"Точечный график: '{y_axis}' от '{x_axis}'"
    )
    def get_fmt(m):
        if "руб" in m: return "%{value:,.0f} ₽".replace(",", " ")
        if str(m).endswith("(%)") or "наценк" in m.lower() or "доля" in m.lower() or m == Metrics.YIELD.value: return "%{value:.2f}%"
        if "дней" in m: return "%{value:.2f} дн."
        return "%{value:,.2f}"
    fig.update_traces(hovertemplate=f"<b>%{{hovertext}}</b><br><br>{x_axis}: {get_fmt(x_axis).replace('value', 'x')}<br>{y_axis}: {get_fmt(y_axis).replace('value', 'y')}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)


def scatter_plot_block_years(df_all: pd.DataFrame, monthly_data: pd.DataFrame, color_map: Dict[str, str], year_a: int, year_b: int):
    st.subheader("🔬 Взаимосвязи (A vs B)")
    if monthly_data.empty: st.info("Нет данных."); return
    available_months = sorted_months_safe(monthly_data['Месяц'])
    selected_months = st.multiselect("Месяцы:", options=available_months, default=available_months, key="scm_months")
    if not selected_months: st.warning("Выберите хотя бы один месяц."); return
    df2 = monthly_data[monthly_data['Месяц'].isin(selected_months)].copy()

    raw_metric_names = set(df_all["Показатель"].dropna().unique())
    metric_options = sorted([m for m in df2.columns if m in raw_metric_names])

    if len(metric_options) < 2: st.warning("Нужно как минимум 2 метрики из файла."); return
    c1, c2 = st.columns(2)
    x_axis = c1.selectbox("Ось X", options=metric_options, index=0, key="scm_x")
    y_axis = c2.selectbox("Ось Y", options=metric_options, index=1 if len(metric_options)>1 else 0, key="scm_y")

    fig = px.scatter(df2, x=x_axis, y=y_axis, color="Год", symbol="Регион",
                     hover_name="Подразделение", title=f"'{y_axis}' от '{x_axis}' · {year_b} vs {year_a}")
    st.plotly_chart(fig, use_container_width=True)

def weighted_corr(df: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Рассчитывает взвешенную корреляционную матрицу."""
    valid_idx = weights.dropna().index.intersection(df.dropna().index)
    weights = weights.loc[valid_idx]
    df = df.loc[valid_idx]

    if weights.sum() <= 0 or (weights < 0).any() or len(df) < 2:
        return pd.DataFrame(index=df.columns, columns=df.columns)

    cov_matrix = np.cov(df.T, aweights=weights)
    std_devs = np.sqrt(np.diag(cov_matrix))

    with np.errstate(divide='ignore', invalid='ignore'):
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

    corr_matrix[np.isinf(corr_matrix)] = np.nan
    return pd.DataFrame(corr_matrix, index=df.columns, columns=df.columns)

def correlations_block(df_all: pd.DataFrame, monthly_data: pd.DataFrame):
    st.subheader("🔗 Корреляции (Pearson)")
    if monthly_data.empty:
        st.info("Нет данных."); return

    with st.expander("Как читать и использовать", expanded=True):
        st.info("""
            - **+1** — сильная прямая связь, **-1** — сильная обратная, **0** — связи нет.
            - Пул по всем регионам даёт «среднюю» картину; для нюансов используйте режим «По регионам».
            - **Взвешенная корреляция** придаёт большее значение подразделениям с высоким показателем веса (например, Выручкой).
            - Корреляция ≠ причинность.
        """)

    has_multiple_years = 'Год' in monthly_data.columns and monthly_data['Год'].nunique() > 1

    if has_multiple_years:
        analysis_scope = st.radio("Масштаб анализа", ["Объединённый", "Разделить по годам"], horizontal=True, key="corr_scope")
    else:
        analysis_scope = "Объединённый"

    mode = st.radio("Режим расчёта", ["Общий по выборке", "По каждому региону"], horizontal=True, key="corr_mode")
    use_weights = st.checkbox("Взвесить корреляции", help="Учитывать вес каждого наблюдения (строки) при расчёте. Например, по размеру выручки.", key="corr_weights")

    raw_metric_names = set(df_all["Показатель"].dropna().unique())

    weight_col = None
    if use_weights:
        numeric_cols = sorted([c for c in monthly_data.columns if pd.api.types.is_numeric_dtype(monthly_data[c]) and c != 'Год' and c in raw_metric_names])
        if not numeric_cols:
            st.warning("Нет метрик из файла для использования в качестве веса.")
            use_weights = False
        else:
            default_idx = numeric_cols.index(Metrics.REVENUE.value) if Metrics.REVENUE.value in numeric_cols else 0
            weight_col = st.selectbox("Вес:", options=numeric_cols, index=default_idx, key="corr_weight_col")


    def draw_corr(df: pd.DataFrame, title: str, weights_col_name: str | None = None):
        metrics_for_corr = sorted([c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != 'Год' and c in raw_metric_names])

        if weights_col_name and weights_col_name in metrics_for_corr:
            metrics_for_corr.remove(weights_col_name)

        if len(metrics_for_corr) < 2 or len(df) < 3:
            st.info(f"Недостаточно данных для расчёта ({title})."); return

        df_for_corr = df[metrics_for_corr]

        if weights_col_name:
            weights = df[weights_col_name]
            cm = weighted_corr(df_for_corr, weights)
            title += f" (взвешенно по '{weights_col_name}')"
        else:
            cm = df_for_corr.corr(method="pearson")

        fig = go.Figure(data=go.Heatmap(
            z=cm.values, x=cm.columns, y=cm.index,
            colorscale="RdBu", zmin=-1, zmax=1,
            text=cm.round(2), texttemplate="%{text}",
            colorbar=dict(tickformat=".2f", title="Корреляция")
        ))
        fig.update_layout(height=600, title=title, margin=dict(t=60,l=0,r=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    if analysis_scope == "Разделить по годам":
        years = sorted(monthly_data['Год'].dropna().unique())
        cols = st.columns(len(years))
        for i, year in enumerate(years):
            with cols[i]:
                df_year = monthly_data[monthly_data['Год'] == year]
                if mode == "Общий по выборке":
                    draw_corr(df_year, f"Корреляции за {year}", weights_col_name=weight_col)
                else:
                    regs = sorted(df_year["Регион"].dropna().astype(str).unique())
                    if not regs: st.info(f"Нет регионов для отображения за {year}."); continue
                    for reg in regs:
                        df_reg = df_year[df_year["Регион"].astype(str) == reg]
                        draw_corr(df_reg, f"{year} - {reg}", weights_col_name=weight_col)
    else:
        if mode == "Общий по выборке":
            draw_corr(monthly_data, "Корреляционная матрица", weights_col_name=weight_col)
        else:
            regs = sorted(monthly_data["Регион"].dropna().astype(str).unique())
            if not regs: st.info("Нет регионов для отображения."); return
            tabs = st.tabs(regs)
            for i, reg in enumerate(regs):
                with tabs[i]:
                    df_reg = monthly_data[monthly_data["Регион"].astype(str) == reg]
                    draw_corr(df_reg, f"Корреляции — регион: {reg}", weights_col_name=weight_col)

def monthly_totals_table(df_raw: pd.DataFrame, regions: list[str], months_range: list[str], all_available_months: list[str]) -> pd.DataFrame:
    df = strip_totals_rows(df_raw)
    if df.empty or not months_range:
        return pd.DataFrame()
    sub = df[(df["Регион"].isin(regions))].copy()
    if sub.empty:
        return pd.DataFrame()

    all_metrics = sorted(sub["Показатель"].dropna().unique().tolist())
    rows = []

    for metric in all_metrics:
        dfm = get_monthly_totals_from_file(df_raw, tuple(regions), metric)
        row = {"Показатель": metric}
        rule = aggregation_rule(metric)

        for m in months_range:
            vals = pd.to_numeric(dfm.loc[dfm["Месяц"].astype(str) == m, "Значение"], errors="coerce")
            row[m] = float(vals.sum()) if not vals.empty else np.nan

        avail = [m for m in months_range if pd.notna(row.get(m))]
        if not avail:
            row["Итого"] = np.nan
        else:
            series = pd.Series([row[m] for m in avail], index=avail)
            if rule == "sum":
                row["Итого"] = float(series.sum())
            elif rule == "mean":
                row["Итого"] = float(series.mean())
            elif rule == "last":
                row["Итого"] = float(series.iloc[-1])
            else:
                row["Итого"] = np.nan
        rows.append(row)

    dfw = pd.DataFrame(rows, columns=["Показатель"] + months_range + ["Итого"])
    def row_order(s): return s.map(lambda name: 0 if name in {Metrics.DEBT.value, Metrics.DEBT_UNITS.value} else (1 if "руб" in name or "шт" in name else 2))
    return dfw.sort_values(by="Показатель", key=row_order).reset_index(drop=True)

def provided_totals_from_files(df_all: pd.DataFrame, regions: list[str], months_range: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_tot = df_all[
        (df_all["Регион"].isin(regions)) &
        df_all["Подразделение"].str.contains(r"^\s*итого\b", case=False, na=False) &
        (df_all["Месяц"].astype(str).isin(months_range + ["Итого"]))
    ].copy()
    if df_tot.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_tot["__prio__"] = np.where(df_tot.get("ИсточникФайла", pd.Series(index=df_tot.index, dtype=object)).eq("TOTALS_FILE"), 1, 2)
    df_tot.sort_values(["Показатель","Месяц","__prio__"], inplace=True)
    best = df_tot.groupby(["Показатель","Месяц"], observed=True).first().reset_index()

    totals_row = best.pivot_table(index="Показатель", columns="Месяц", values="Значение", aggfunc="first").reset_index()
    cols_ordered = ["Показатель"] + [m for m in months_range if m in totals_row.columns] + (["Итого"] if "Итого" in totals_row.columns else [])
    totals_row = totals_row.reindex(columns=cols_ordered)

    totals_col = pd.DataFrame()
    if "Итого" in best["Месяц"].astype(str).unique():
        it_col = best[best["Месяц"].astype(str) == "Итого"][["Показатель", "Значение"]].rename(columns={"Значение": "Итого"})
        totals_col = it_col.groupby("Показатель")["Итого"].first().reset_index()

    return totals_row, totals_col

def reconciliation_block(df_all: pd.DataFrame, regions: list[str], months_range: list[str], all_available_months: list[str]):
    st.subheader("🧮 Сверка с «Итого» из файлов"); st.caption("Сравнение нашего расчета с данными из строк и столбцов 'Итого' в исходных файлах.")
    if df_all.empty or not months_range: st.info("Нет данных для сверки."); return
    st.info("""
        💡 **Важно: почему «Итого» по ссудной задолженности не совпадает**

        В нашем расчете «Итого» — это **среднее значение за период**. В файле Excel — это **сумма остатков за все месяцы**.
        Суммировать остатки некорректно, поэтому расхождение для этих метрик является ожидаемым и не считается ошибкой.
    """)
    ours = monthly_totals_table(df_all, regions, months_range, all_available_months)
    if ours.empty: st.info("Не удалось построить собственную таблицу для сверки."); return
    tot_row, tot_col = provided_totals_from_files(df_all, regions, months_range)
    colcfg = {c: st.column_config.NumberColumn(c, format="%.0f") for c in ours.columns if c != "Показатель"}
    colcfg["Показатель"] = st.column_config.TextColumn("Показатель")
    st.markdown("**1) Наш расчёт (месяцы + Итого)**"); st.dataframe(ours, width="stretch", column_config=colcfg)
    if not tot_row.empty:
        st.markdown("**2) «Итого по всем подразделениям» из файлов (строки 'Итого'):**"); st.dataframe(tot_row, width="stretch", column_config=colcfg)
        left, right = ours.set_index("Показатель"), tot_row.set_index("Показатель").reindex(ours["Показатель"])
        common_cols = [c for c in left.columns if c in right.columns]
        diff = (left[common_cols] - right[common_cols]).reset_index()
        st.markdown("**Δ Разница (наш расчёт − «Итого» строки):**"); st.dataframe(diff, width="stretch", column_config=colcfg)
    if not tot_col.empty:
        st.markdown("**3) Сверка со столбцом «Итого» в строках подразделений:**")
        left = ours[["Показатель","Итого"]].set_index("Показатель")
        right = tot_col.set_index("Показатель").reindex(left.index)
        diff2 = (left["Итого"] - right["Итого"]).reset_index()
        st.dataframe(diff2, width="stretch", column_config={"Показатель": st.column_config.TextColumn("Показатель"), "Итого": st.column_config.NumberColumn("Δ по «Итого»", format="%.0f")})
    if tot_row.empty and tot_col.empty: st.info("В загруженных файлах не обнаружено данных «Итого» для сверки.")

def month_check_block(df_all: pd.DataFrame, regions: list[str], months_range: list[str], all_available_months: list[str]):
    st.subheader("📅 Сводная таблица по месяцам")
    st.caption("Итоговые значения по всем выбранным регионам, рассчитанные по согласованной методике.")
    if df_all.empty or not months_range: st.info("Нет данных."); return
    table = monthly_totals_table(df_all, regions, months_range, all_available_months)
    if table.empty: st.info("Нет данных для таблицы."); return
    colcfg = {c: st.column_config.NumberColumn(c, format="%.0f") for c in table.columns if c != "Показатель"}
    colcfg["Показатель"] = st.column_config.TextColumn("Показатель")
    st.dataframe(table, width="stretch", column_config=colcfg)

def export_block(df_long: pd.DataFrame):
    st.subheader("📥 Экспорт данных"); st.caption("Длинный формат: Регион · Год · Подразделение · Показатель · Месяц · Значение.")
    csv_bytes = df_long.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Скачать объединённый датасет (CSV)", data=csv_bytes, file_name="NUZ_combined_Long.csv", mime="text/csv")

def info_block():
    st.header("ℹ️ Справка по методике")
    st.markdown("""
### Ключ расчёта
- **В упрощенном режиме все показатели берутся строго из файла.**
- **Суммы (руб, шт):** агрегируются как **сумма** месячных значений за период.
- **Проценты и доли (%):** агрегируются как **простое среднее** месячных значений за период.
- **Ссудная задолженность:** агрегируется как **последнее значение** за период (снимок на конец периода).
- **Количество ломбардов:** агрегируется как **последнее** значение за период (снимок на конец периода).

### Советы по анализу
- **Если доля продаж ниже займа растет**, ожидайте снижения наценки и, возможно, увеличения объема неликвида. Рекомендуем проверить политику оценки залогов в подразделениях с высоким значением этого показателя.
- **Высокая доходность (процентная) не всегда хорошо**, если при этом много залогов уходит в продажу. Ищите баланс между процентным доходом и качеством залогов: идеальная картина – умеренная доходность при низкой доле убыточных продаж.
- **Сравнивайте средний размер займа и процент выкупа изделий**. Слишком большие средние суммы могут означать, что филиал выдает займы под дорогие товары, которые клиентам сложнее выкупить, что повышает риск перехода в распродажу.
""")

def main():
    st.markdown(f"# 📊 Аналитический дашборд: НЮЗ  \n<span class='badge'>Версия {APP_VERSION}</span>", unsafe_allow_html=True)
    st.sidebar.header("📥 Загрузка данных")
    region_prefix = st.sidebar.text_input("Префикс региона", value="", help="Будет добавлен к имени файла как название региона.")
    uploads = st.sidebar.file_uploader("Загрузите файлы Excel", type=["xlsx","xls"], accept_multiple_files=True)
    if not uploads: st.info("⬅️ Загрузите один или несколько файлов Excel для старта."); st.stop()
    # strict_mode теперь не нужен как опция, он определяется глобальным флагом SIMPLE_MODE
    strict_mode = SIMPLE_MODE

    dfs, errors = [], []
    with st.spinner("Чтение и обработка файлов..."):
        for up in uploads:
            try:
                stem = Path(up.name).stem
                region_name = f"{region_prefix.strip()}: {stem}" if region_prefix.strip() else stem
                year_guess = guess_year_from_filename(up.name)

                key_year = f"year_for_{up.name}"
                if year_guess is None:
                    st.sidebar.warning(f"Не удалось определить год из имени: {up.name}")
                    st.sidebar.caption("Выберите год вручную.")
                    year_guess = st.sidebar.selectbox(
                        f"Год для файла: {up.name}",
                        options=[2023, 2024, 2025, 2026],
                        index=1,
                        key=key_year
                    )
                dfs.append(parse_excel(up.getvalue(), region_name, file_year=year_guess))
            except Exception as e:
                errors.append(f"**{up.name}**: {e}")

    if errors: st.error("Ошибки при чтении файлов:\n\n" + "\n\n".join(errors))
    if not dfs: st.stop()
    df_all = pd.concat(dfs, ignore_index=True)

    # доп. нормализация поля "Регион"
    df_all["Регион"] = (df_all["Регион"]
        .str.replace(r"\s{2,}", " ", regex=True)
        .str.replace(r"[·.]+$", "", regex=True)
        .str.strip()
        .astype("string")
    )

    df_all["Значение"] = pd.to_numeric(df_all["Значение"], errors="coerce")
    df_all["Год"] = pd.to_numeric(df_all["Год"], errors="coerce").astype("Int64")
    for c in ["Подразделение", "Показатель", "Код", "Месяц", "ИсточникФайла", "Категория"]:
        if c == "Месяц":
            df_all[c] = df_all[c].astype(pd.CategoricalDtype(categories=ORDER_WITH_TOTAL, ordered=True))
        else:
            df_all[c] = df_all[c].astype("string")

    years_all = sorted([int(y) for y in pd.Series(df_all["Год"].dropna().unique()).astype(int)])
    if not years_all:
        st.error("Не удалось определить год ни для одного из файлов. Проверьте названия файлов или выберите год вручную.")
        st.stop()

    mask_pct = df_all["Показатель"].apply(is_percent_metric)
    df_all.loc[mask_pct, "Значение"] = normalize_percent_series(df_all.loc[mask_pct, "Значение"])

    only_nuz = st.sidebar.toggle("Фильтровать только НЮЗ", value=True)
    if only_nuz and "Категория" in df_all.columns:
        df_nuz = df_all[df_all["Категория"].fillna("Общее") == "НЮЗ"]
        if df_nuz.empty:
            st.sidebar.warning("В файлах не найдено строк с меткой «НЮЗ». Показаны все категории.")
        else:
            df_all = df_nuz

    st.sidebar.markdown("### Режим анализа")
    mode_year = st.sidebar.radio("", options=["Один год", "Сравнение годов"], index=0, horizontal=True)

    if mode_year == "Один год":
        year_selected = st.sidebar.selectbox("Год", options=years_all, index=len(years_all)-1)
        df_scope = df_all[df_all["Год"] == year_selected].copy()
        months_in_data = sorted_months_safe(df_scope["Месяц"])
    else: # Сравнение годов
        c_y1, c_y2 = st.sidebar.columns(2)
        year_a = c_y1.selectbox("Год A (база)", options=years_all, index=max(0, len(years_all)-2), key="year_a")
        year_b = c_y2.selectbox("Год B (сравнение)", options=years_all, index=len(years_all)-1, key="year_b")
        df_a = df_all[df_all["Год"] == year_a].copy()
        df_b = df_all[df_all["Год"] == year_b].copy()
        months_a = sorted_months_safe(df_a["Месяц"])
        months_b = sorted_months_safe(df_b["Месяц"])
        months_in_data = [m for m in ORDER if m in months_a and m in months_b]

    if not months_in_data:
        st.error("В выбранном периоде (или пересечении периодов) нет данных с распознанными месяцами."); st.stop()

    st.sidebar.markdown("**Быстрый выбор периода**")
    if "global_period" not in st.session_state:
        st.session_state["global_period"] = (months_in_data[0], months_in_data[-1])

    preset = st.sidebar.radio("Быстрый выбор", options=["Весь период","Квартал","2 мес.","Текущий мес."], index=0, horizontal=True)
    if preset == "Весь период": st.session_state["global_period"] = (months_in_data[0], months_in_data[-1])
    elif preset == "Квартал":
        rng = months_in_data[-3:] if len(months_in_data) >= 3 else months_in_data
        st.session_state["global_period"] = (rng[0], rng[-1])
    elif preset == "2 мес.":
        rng = months_in_data[-2:] if len(months_in_data) >= 2 else months_in_data
        st.session_state["global_period"] = (rng[0], rng[-1])
    else:
        last1 = months_in_data[-1]
        st.session_state["global_period"] = (last1, last1)

    start_default, end_default = st.session_state.get("global_period", (months_in_data[0], months_in_data[-1]))
    if start_default not in months_in_data or end_default not in months_in_data:
        start_default, end_default = months_in_data[0], months_in_data[-1]

    start_m, end_m = st.sidebar.select_slider("Период", options=months_in_data, value=(start_default, end_default), key="period_slider")
    months_range = ORDER[ORDER.index(start_m): ORDER.index(end_m) + 1]

    if st.sidebar.button("Сбросить фильтры"):
        st.session_state.clear(); st.rerun()

    if st.sidebar.button("❌ Начать заново (сбросить файлы)"):
        st.cache_data.clear(); st.session_state.clear(); st.rerun()

    if mode_year == "Один год":
        regions_all = sorted(map(str, df_scope["Регион"].unique()))
        regions = st.sidebar.multiselect("Регионы", options=regions_all, default=regions_all)
        if not regions:
            st.warning("Выберите хотя бы один регион для анализа.")
            st.stop()
            
        st.markdown(f"**Анализ за {year_selected} год.** Активные фильтры: {len(regions)} из {len(regions_all)} регионов; Период: **{months_range[0]}** – **{months_range[-1]}**")
        st.divider()

        agg_data_global = get_aggregated_data(df_scope, tuple(regions), tuple(months_range))
        monthly_data_global = get_monthly_pivoted_data(df_scope, tuple(regions), tuple(months_range), raw_only=True)

        with st.expander("🔍 Статус и проверка данных", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Файлов", len(uploads)); c2.metric("Регионов", len(regions_all))
            c3.metric("Подразделений", strip_totals_rows(df_scope)["Подразделение"].nunique())
            c4.metric("Период в данных", f"{months_in_data[0]} – {months_in_data[-1]}")

        tab_list = ["📊 KPI","📋 Сводка","🏆 Лидеры","⚖️ Сравнение","📈 Динамика","🗺️ Структура","🔬 Взаимосвязи","🔗 Корреляции","📥 Экспорт","ℹ️ Справка"]
        tabs = st.tabs(tab_list)
        color_map = consistent_color_map(tuple(regions_all))

        with tabs[0]: kpi_block(df_scope, regions, months_range, months_in_data, strict_mode)
        with tabs[1]: summary_block(agg_data_global, df_scope, regions, months_range, months_in_data, strict_mode)
        with tabs[2]: leaderboard_block(df_scope, regions, months_in_data)
        with tabs[3]: comparison_block(df_scope, regions, months_in_data)
        with tabs[4]: dynamics_block(df_scope, regions, months_range, color_map)
        with tabs[5]: treemap_heatmap_block(df_scope, regions, months_range, color_map)
        with tabs[6]: scatter_plot_block(df_scope, monthly_data_global, color_map)
        with tabs[7]: correlations_block(df_scope, monthly_data_global)
        with tabs[8]: export_block(df_scope[df_scope["Регион"].isin(regions) & df_scope["Месяц"].isin(months_range)])
        with tabs[9]: info_block()

    else: # Режим сравнения годов
        regions_all = sorted(map(str, pd.concat([df_a, df_b])["Регион"].unique()))
        regions = st.sidebar.multiselect("Регионы", options=regions_all, default=regions_all)
        if not regions:
            st.warning("Выберите хотя бы один регион для анализа.")
            st.stop()

        st.markdown(f"**Сравнение {year_b} vs {year_a}.** Активные фильтры: {len(regions)} из {len(regions_all)} регионов; Период: **{months_range[0]}** – **{months_range[-1]}**")
        st.divider()

        monthly_a = get_monthly_pivoted_data(df_a, tuple(regions), tuple(months_range), raw_only=True)
        monthly_b = get_monthly_pivoted_data(df_b, tuple(regions), tuple(months_range), raw_only=True)
        monthly_ab = pd.concat([monthly_a.assign(Год=year_a), monthly_b.assign(Год=year_b)], ignore_index=True)
        color_map = consistent_color_map(tuple(regions_all))

        tab_list = ["📊 Итоги", "📈 Динамика", "🔬 Взаимосвязи", "🔗 Корреляции", "📥 Экспорт", "ℹ️ Справка"]
        tabs = st.tabs(tab_list)

        with tabs[0]:
            yoy_summary_block(df_a, df_b, regions, months_range, years=(year_a, year_b))
        with tabs[1]:
            dynamics_compare_block(df_a, df_b, regions, months_range, color_map, year_a, year_b)
        with tabs[2]:
            scatter_plot_block_years(df_all, monthly_ab, color_map, year_a, year_b)
        with tabs[3]:
            correlations_block(df_all, monthly_ab)
        with tabs[4]:
            export_block(pd.concat([df_a, df_b], ignore_index=True))
        with tabs[5]:
            info_block()


if __name__ == "__main__":
    main()