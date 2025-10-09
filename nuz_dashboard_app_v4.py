# Запуск: streamlit run nuz_dashboard_app_v4.py
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative as qcolors
from plotly.subplots import make_subplots
import streamlit as st
import requests

# A) Глобальные флаги
APP_VERSION = "v24.56-heatmap-fix"
# Режим: работать только с «Итого по месяцу» из файла, без формул/досчётов
SIMPLE_MODE = True
# Аналитика только по НЮЗ
NUZ_ONLY = True

# A) Вспомогательные функции для года
YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")

# Приближённые координаты ключевых регионов для карт Plotly (широта, долгота)
REGION_COORDS: Dict[str, Tuple[float, float]] = {
    "Москва": (55.7558, 37.6176),
    "Санкт-Петербург": (59.9391, 30.3158),
    "Новосибирск": (55.0084, 82.9357),
    "Екатеринбург": (56.8389, 60.6057),
    "Казань": (55.7903, 49.1347),
    "Нижний Новгород": (56.3269, 44.0059),
    "Краснодар": (45.0355, 38.9753),
    "Ростов-на-Дону": (47.2357, 39.7015),
    "Самара": (53.1959, 50.1008),
    "Челябинск": (55.1644, 61.4368),
    "Уфа": (54.7388, 55.9721),
    "Воронеж": (51.6608, 39.2003),
    "Пермь": (58.0105, 56.2502),
    "Красноярск": (56.0153, 92.8932),
    "Омск": (54.9885, 73.3242),
    "Иркутск": (52.2869, 104.3050),
    "Тюмень": (57.1530, 65.5343),
    "Сочи": (43.6028, 39.7342),
    "Хабаровск": (48.4808, 135.0928),
    "Владивосток": (43.1155, 131.8855),
}
REGION_COORDS_INDEX = {name.lower(): coords for name, coords in REGION_COORDS.items()}
REGION_GEOCODER_URL = "https://nominatim.openstreetmap.org/search"

def guess_year_from_filename(name: str) -> int | None:
    s = str(name).lower().replace("г.", " ").replace("г", " ")
    m = YEAR_RE.search(s)
    return int(m.group(1)) if m else None


def _resolve_region_coordinates_static(name: str) -> tuple[float, float] | None:
    if not name:
        return None
    raw = str(name).strip()
    candidates = {raw}
    if ":" in raw:
        candidates.add(raw.split(":", 1)[-1].strip())
    simplified = re.sub(r"\s*\(.*?\)", "", raw).strip()
    candidates.add(simplified)
    for candidate in candidates:
        key = candidate.lower()
        coords = REGION_COORDS_INDEX.get(key)
        if coords:
            return coords
        titled = candidate.title()
        coords = REGION_COORDS_INDEX.get(titled.lower())
        if coords:
            return coords
    return None


def _geocode_region(name: str) -> tuple[float, float] | None:
    if not name:
        return None
    try:
        headers = {"User-Agent": f"NUZ-Dashboard/{APP_VERSION}"}
        params = {"q": name, "format": "json", "limit": 1, "addressdetails": 0}
        resp = requests.get(REGION_GEOCODER_URL, params=params, headers=headers, timeout=8)
        if resp.status_code != 200:
            return None
        payload = resp.json()
        if not payload:
            return None
        lat = float(payload[0]["lat"])
        lon = float(payload[0]["lon"])
        return lat, lon
    except Exception:
        return None


def resolve_region_coordinates(name: str) -> tuple[float, float] | None:
    cached = _resolve_region_coordinates_static(name)
    if cached:
        return cached
    cache = st.session_state.setdefault("region_coords_dynamic", {})
    if name in cache:
        return cache[name]
    coords = _geocode_region(name)
    if coords:
        cache[name] = coords
        REGION_COORDS_INDEX[name.lower()] = coords
    return coords

# ------------------------- Конфигурация страницы -------------------------
st.set_page_config(page_title=f"НЮЗ — Дашборд {APP_VERSION}", layout="wide", page_icon="📊")
st.markdown("""
<style>
:root { --text-base: 15px; }
html, body, [class*="css"]  {
    font-size: var(--text-base);
    background: linear-gradient(180deg, #f5f7fb 0%, #eef1f6 35%, #e6ebf4 100%) !important;
    color: #0f172a;
}
section[data-testid="stSidebar"] {
    min-width: 330px !important;
    background: rgba(248, 250, 255, 0.82) !important;
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(255,255,255,0.6);
}
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
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.78);
    border-radius: 22px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.6);
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
    backdrop-filter: blur(20px);
}

div[data-testid="stDataFrame"] > div {
    border-radius: 18px;
    background: rgba(255,255,255,0.85);
    border: 1px solid rgba(226,232,240,0.7);
    box-shadow: 0 12px 30px rgba(15,23,42,0.08);
}

.stButton > button {
    border-radius: 14px;
    padding: 0.6rem 1.2rem;
    background: linear-gradient(135deg, #2563eb, #60a5fa);
    color: #fff;
    border: none;
    box-shadow: 0 10px 20px rgba(37, 99, 235, 0.25);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6);
}

.stRadio > div { gap: .6rem; }
.sidebar-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin: 1.4rem 0 0.5rem;
}
.sidebar-divider {
    margin: 1.2rem 0;
    height: 1px;
    background: linear-gradient(90deg, rgba(148,163,184,0), rgba(148,163,184,0.45), rgba(148,163,184,0));
    border: none;
}
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div {
    border-radius: 16px;
    background: rgba(255,255,255,0.65);
    border: 1px solid rgba(148,163,184,0.35);
    backdrop-filter: blur(12px);
}
section[data-testid="stSidebar"] .stRadio > div > label {
    background: rgba(255,255,255,0.55);
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    border: 1px solid rgba(148,163,184,0.25);
    color: #334155;
    font-weight: 600;
}
section[data-testid="stSidebar"] .stRadio > div > label[data-baseweb="radio"]:has(input[checked]) {
    background: linear-gradient(135deg, #2563eb, #60a5fa);
    color: white;
    border-color: transparent;
}
.hero {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 18px 22px;
    border-radius: 26px;
    background: rgba(255, 255, 255, 0.75);
    box-shadow: 0 22px 50px rgba(15, 23, 42, 0.12);
    backdrop-filter: blur(30px);
    border: 1px solid rgba(148, 163, 184, 0.20);
    margin-bottom: 1.2rem;
}
.hero__title {
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -.01em;
    color: #0f172a;
}
.hero__meta {
    color: #64748b;
    font-size: 0.92rem;
}
.hero-pill {
    display: inline-flex;
    align-items: center;
    width: fit-content;
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    background: linear-gradient(120deg, rgba(37,99,235,0.12), rgba(59,130,246,0.18));
    color: #1d4ed8;
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
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
    DEBT_NO_SALE_TOTAL = "Ссудная задолженность без распродажи (руб)"
    DEBT_NO_SALE_YUZ = "Ссудная задолженность без распродажи ЮЗ (руб)"
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
    Metrics.RISK_SHARE.value,
    Metrics.YIELD.value,
}
ACCEPTED_METRICS_CANONICAL |= {Metrics.DEBT.value}


def _normalize_metric_label(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw).lower()
    s = s.replace("ё", "е")
    s = re.sub(r"[\"'«»]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" :;.")


METRIC_ALIASES_RAW: Dict[str, List[str]] = {
    Metrics.REVENUE.value: [
        "выручка от распродажи нюз (руб)",
    ],
    Metrics.LOAN_ISSUE.value: [
        "выдано займов нюз (руб)",
    ],
    Metrics.LOAN_ISSUE_UNITS.value: [
        "выдано займов нюз (шт)",
    ],
    Metrics.PLAN_ISSUE_PCT.value: [
        "% выполнения плана выданных займов нюз",
    ],
    Metrics.PENALTIES_RECEIVED.value: [
        "получено % и пени нюз (руб)",
    ],
    Metrics.PLAN_PENALTIES_PCT.value: [
        "% выполнения плана по полученным % и пеням нюз",
    ],
    Metrics.PLAN_REVENUE_PCT.value: [
        "% выполнения плана по выручке от распродажи нюз",
    ],
    Metrics.MARKUP_AMOUNT.value: [
        "получено наценки от распродажи нюз (руб)",
    ],
    Metrics.PENALTIES_PLUS_MARKUP.value: [
        "получено % и пени + наценка на распродажу нюз (руб)",
    ],
    Metrics.MARKUP_PCT.value: [
        "процент наценки нюз",
    ],
    Metrics.AVG_LOAN.value: [
        "средняя сумма займа нюз (руб)",
    ],
    Metrics.AVG_LOAN_TERM.value: [
        "средний срок займа за период нюз (дней)",
        "средний срок займа за перод нюз (дней)",
    ],
    Metrics.LOAN_VALUE_OF_SOLD.value: [
        "ссуда вышедших изделий на аукцион нюз (руб)",
        "ссуда  вышедших изделий на аукцион нюз (руб)",
    ],
    Metrics.AUCTIONED_ITEMS_COUNT.value: [
        "количество вышедших изделий на аукцион нюз",
    ],
    Metrics.DEBT_NO_SALE.value: [
        "ссудная задолженность без распродажи нюз (руб)",
    ],
    Metrics.DEBT_UNITS.value: [
        "ссудная задолженность без распродажи нюз (шт)",
    ],
    Metrics.BELOW_LOAN.value: [
        "товар проданный ниже суммы займа нюз (руб)",
        "товар проданный ниже суммы займа нюз  (руб)",
    ],
    Metrics.BELOW_LOAN_UNITS.value: [
        "товар проданный ниже суммы займа нюз (шт)",
        "товар проданный ниже суммы займа нюз  (шт)",
    ],
    Metrics.LOSS_BELOW_LOAN.value: [
        "убыток от товара проданного ниже суммы займа нюз (руб)",
    ],
    Metrics.REDEEMED_ITEMS_COUNT.value: [
        "количество выкупленных залогов за период нюз (шт)",
        "выкуп заложенных за период количество нюз",
    ],
    Metrics.REDEEMED_SUM.value: [
        "сумма выкупленных за период нюз (руб)",
    ],
    Metrics.REDEEMED_SHARE_PCT.value: [
        "доля выкупов заложенных за период нюз (%)",
    ],
    Metrics.LOAN_REPAYMENT_SUM.value: [
        "сумма погашения суммы займа нюз (руб)",
    ],
    Metrics.ISSUE_SHARE.value: [
        "доля нюз по выдаче",
    ],
    Metrics.INTEREST_SHARE.value: [
        "доля нюз по полученным % и пени",
    ],
    Metrics.DEBT_SHARE.value: [
        "доля нюз по ссудной задолженности",
    ],
    Metrics.UNIQUE_CLIENTS.value: [
        "уникальные клиенты",
    ],
    Metrics.NEW_UNIQUE_CLIENTS.value: [
        "новые уникальные клиенты",
    ],
    Metrics.BRANCH_COUNT.value: [
        "количество ломбардов",
    ],
    Metrics.BRANCH_NEW_COUNT.value: [
        "количество новых ломбардов",
    ],
    Metrics.BRANCH_CLOSED_COUNT.value: [
        "количество закрытых ломбардов",
    ],
    Metrics.ILLIQUID_BY_COUNT_PCT.value: [
        "доля неликвида от количества (%)",
    ],
    Metrics.ILLIQUID_BY_VALUE_PCT.value: [
        "доля неликвида от оценки (%)",
    ],
    Metrics.YIELD.value: [
        "доходность",
    ],
}

METRIC_ALIAS_MAP: Dict[str, str] = {}
for canonical, variants in METRIC_ALIASES_RAW.items():
    for candidate in [canonical, *variants]:
        key = _normalize_metric_label(candidate)
        if not key:
            continue
        existing = METRIC_ALIAS_MAP.get(key)
        if existing and existing != canonical:
            continue
        METRIC_ALIAS_MAP[key] = canonical

METRIC_CATEGORY_OVERRIDES: Dict[str, str] = {
    Metrics.YIELD.value: "НЮЗ",
    Metrics.ILLIQUID_BY_COUNT_PCT.value: "НЮЗ",
    Metrics.ILLIQUID_BY_VALUE_PCT.value: "НЮЗ",
    Metrics.BRANCH_COUNT.value: "НЮЗ",
    Metrics.BRANCH_NEW_COUNT.value: "НЮЗ",
    Metrics.BRANCH_CLOSED_COUNT.value: "НЮЗ",
    Metrics.UNIQUE_CLIENTS.value: "НЮЗ",
    Metrics.NEW_UNIQUE_CLIENTS.value: "НЮЗ",
}

HIDDEN_METRICS = {
    Metrics.DEBT.value,
    Metrics.DEBT_NO_SALE.value,
    Metrics.DEBT_UNITS.value,
}


def append_risk_share_metric(df: pd.DataFrame) -> pd.DataFrame:
    needed = {Metrics.BELOW_LOAN.value, Metrics.REVENUE.value, Metrics.DEBT_NO_SALE.value}
    present = set(df["Показатель"].dropna().unique())
    if not (needed & present):
        return df

    cols = ["Регион", "Подразделение", "Категория", "Код", "Месяц", "Год"]
    subset = df[df["Показатель"].isin(needed)].copy()
    if subset.empty:
        return df
    subset["Значение"] = pd.to_numeric(subset["Значение"], errors="coerce")
    pivot = subset.pivot_table(index=cols, columns="Показатель", values="Значение", aggfunc="sum")
    if pivot.empty:
        return df

    derived_frames: List[pd.DataFrame] = []

    revenue = pivot.get(Metrics.REVENUE.value)
    below = pivot.get(Metrics.BELOW_LOAN.value)
    if revenue is not None and below is not None:
        denominator = revenue.replace(0, np.nan)
        ratio = (below / denominator) * 100.0
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        if not ratio.empty:
            risk_df = ratio.reset_index().rename(columns={0: "Значение"})
            risk_df["Показатель"] = Metrics.RISK_SHARE.value
            derived_frames.append(risk_df)

    if not derived_frames:
        return df

    derived = pd.concat(derived_frames, ignore_index=True)
    derived["ИсточникФайла"] = "DERIVED"
    order = ["Регион", "Подразделение", "Категория", "Код", "Показатель", "Месяц", "Значение", "Год", "ИсточникФайла"]
    for col in order:
        if col not in derived:
            derived[col] = pd.NA

    key_cols = ["Регион", "Подразделение", "Категория", "Код", "Показатель", "Месяц", "Год"]
    existing_keys = set(tuple(row) for row in df[df["Показатель"] == Metrics.RISK_SHARE.value][key_cols].itertuples(index=False, name=None))
    derived = derived[~derived[key_cols].apply(tuple, axis=1).isin(existing_keys)]
    return pd.concat([df, derived[order]], ignore_index=True)


ORDER = ["Январь","Февраль","Март","Апрель","Май","Июнь","Июль","Август","Сентябрь","Октябрь","Ноябрь","Декабрь"]
ORDER_WITH_TOTAL = ORDER + ["Итого"]

NUZ_ACTIVITY_METRICS = {
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
    Metrics.ILLIQUID_BY_COUNT_PCT.value,
    Metrics.ILLIQUID_BY_VALUE_PCT.value,
    Metrics.YIELD.value,
    Metrics.ISSUE_SHARE.value,
    Metrics.INTEREST_SHARE.value,
    Metrics.PLAN_ISSUE_PCT.value,
    Metrics.PLAN_PENALTIES_PCT.value,
    Metrics.PLAN_REVENUE_PCT.value,
}

FORECAST_METRICS = [
    Metrics.REVENUE.value,
    Metrics.LOAN_ISSUE.value,
    Metrics.PENALTIES_RECEIVED.value,
    Metrics.MARKUP_PCT.value,
    Metrics.PLAN_ISSUE_PCT.value,
    Metrics.PLAN_PENALTIES_PCT.value,
    Metrics.PLAN_REVENUE_PCT.value,
]

TAB_METRIC_SETS: Dict[str, List[str]] = {
    "issuance": [
        Metrics.LOAN_ISSUE.value,
        Metrics.LOAN_ISSUE_UNITS.value,
        Metrics.AVG_LOAN.value,
        Metrics.PLAN_ISSUE_PCT.value,
        Metrics.REDEEMED_ITEMS_COUNT.value,
        Metrics.REDEEMED_SUM.value,
    ],
    "interest": [
        Metrics.PENALTIES_RECEIVED.value,
        Metrics.YIELD.value,
        Metrics.LOAN_REPAYMENT_SUM.value,
        Metrics.PLAN_PENALTIES_PCT.value,
        Metrics.REDEEMED_SHARE_PCT.value,
    ],
    "sales": [
        Metrics.REVENUE.value,
        Metrics.MARKUP_PCT.value,
        Metrics.PENALTIES_PLUS_MARKUP.value,
        Metrics.BELOW_LOAN.value,
        Metrics.RISK_SHARE.value,
        Metrics.PLAN_REVENUE_PCT.value,
    ],
    "risk": [
        Metrics.RISK_SHARE.value,
        Metrics.ILLIQUID_BY_VALUE_PCT.value,
        Metrics.ILLIQUID_BY_COUNT_PCT.value,
        Metrics.BELOW_LOAN.value,
    ],
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
    if not base.empty:
        priority_map = {"RECALC_TOTAL": 0, "TOTALS_FILE": 1}
        src = base.get("ИсточникФайла", pd.Series(index=base.index, dtype=object))
        prio = src.map(priority_map).fillna(2).astype(int)
        base["__prio__"] = prio

        def _select_best(g: pd.DataFrame) -> pd.Series:
            priority_zero = g[g["__prio__"] == 0]
            if not priority_zero.empty:
                return priority_zero.iloc[0]
            priority_one = g[g["__prio__"] == 1]
            if not priority_one.empty:
                return priority_one.iloc[0]
            result = g.iloc[0].copy()
            numeric_values = pd.to_numeric(g["Значение"], errors="coerce")
            result["Значение"] = float(numeric_values.sum(skipna=True))
            return result

        return (base.groupby(["Регион", "Месяц"], observed=True)
                    .apply(_select_best)
                    .reset_index(drop=True)[["Регион", "Месяц", "Значение"]])

    # Fallback: суммируем по всем подразделениям
    subset = df_raw[
        df_raw["Регион"].isin(regions) &
        (df_raw["Показатель"] == metric) &
        (df_raw["Месяц"].astype(str) != "Итого")
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    agg_type = aggregation_rule(metric)

    def _aggregate_values(values: pd.Series) -> float:
        numeric = pd.to_numeric(values, errors="coerce")
        if agg_type == "mean":
            return float(numeric.mean(skipna=True))
        # для сумм и снимков берём сумму по подразделениям
        return float(numeric.sum(skipna=True))

    aggregated = (
        subset.groupby(["Регион", "Месяц"], observed=True)["Значение"]
        .apply(_aggregate_values)
        .reset_index()
    )
    return aggregated

@st.cache_data(show_spinner=False, max_entries=256)
def month_series_from_file(df_all, regions, metric, months):
    months_tuple = tuple(months)
    dfm = get_monthly_totals_from_file(df_all, tuple(regions), metric)
    if dfm.empty:
        return pd.Series(dtype=float)
    s = (dfm[dfm["Месяц"].astype(str).isin(months_tuple)]
            .groupby("Месяц", observed=True)["Значение"].sum())
    # строгая сортировка по календарю
    s = s.reindex([m for m in months_tuple if m in s.index])
    return s

@st.cache_data
def sorted_months_safe(_values) -> list[str]:
    """Без кеша: аккуратно приводим к строкам и сортируем по нашему ORDER."""
    if _values is None:
        return []
    s = pd.Series(_values)
    if isinstance(s.dtype, pd.CategoricalDtype):
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
    Metrics.DEBT.value,
    Metrics.DEBT_NO_SALE.value,
    Metrics.DEBT_UNITS.value,
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

@st.cache_data(show_spinner=False, max_entries=512)
def period_value_from_itogo(df_all: pd.DataFrame, regions: list[str], metric: str, months: list[str]) -> float | None:
    months_tuple = tuple(months)
    s = month_series_from_file(df_all, regions, metric, months_tuple)
    if s.empty:
        return None
    rule = aggregation_rule(metric)
    vals = pd.to_numeric(s, errors="coerce")
    result: float | None = None
    if rule == "sum":
        result = float(vals.sum())
    elif rule == "mean":
        result = float(vals.mean())
    elif rule == "last":
        # берём последнее доступное значение внутри выбранного окна
        last_months = sorted_months_safe(vals.dropna().index)
        if not last_months:
            result = None
        else:
            v = vals.get(last_months[-1], np.nan)
            result = float(v) if pd.notna(v) else None
    else:
        result = float(vals.mean())
    return _maybe_scale_percent(metric, result)

@st.cache_data(show_spinner=False, max_entries=1024)
def period_value_from_itogo_for_region(df_all: pd.DataFrame, region: str, metric: str,
                                       months: list[str], *, snapshots_mode: str = "last") -> float | None:
    months_tuple = tuple(months)
    # Берём ровно строки «Итого по месяцу» для региона
    dfm = get_monthly_totals_from_file(df_all, (region,), metric)
    if dfm.empty:
        return None
    part = dfm[dfm["Месяц"].astype(str).isin(months_tuple)]
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


@st.cache_data(show_spinner=False, max_entries=256)
def period_values_by_region_from_itogo(df_all, regions, metric, months) -> dict[str, float]:
    """
    Возвращает {Регион: значение за период} строго из строк «Итого по месяцу».
    Сумма/среднее/последний — как задано aggregation_rule(metric).
    """
    months_tuple = tuple(months)
    dfm = get_monthly_totals_from_file(df_all, tuple(regions), metric)
    if dfm.empty:
        return {}

    dfm = dfm[dfm["Месяц"].astype(str).isin(months_tuple)].copy()
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
            result = float(vals.sum())
        elif rule == "mean":
            result = float(vals.mean())
        elif rule == "last":
            result = float(vals.iloc[-1])
        else:
            result = float(vals.mean())  # дефолт
        out[str(reg)] = _maybe_scale_percent(metric, result)
    return out


MANDATORY_COLUMNS = {"Регион", "Подразделение", "Показатель", "Месяц", "Значение"}
COHORT_REQUIRED_METRICS = {Metrics.UNIQUE_CLIENTS.value, Metrics.NEW_UNIQUE_CLIENTS.value}
RISK_REQUIRED_METRICS = {Metrics.RISK_SHARE.value, Metrics.ILLIQUID_BY_VALUE_PCT.value}
SALES_REQUIRED_METRICS = {Metrics.REVENUE.value, Metrics.MARKUP_PCT.value}
TAB_METRIC_DEPENDENCIES: Dict[str, set[str]] = {
    "Риски": RISK_REQUIRED_METRICS,
    "Когорты": COHORT_REQUIRED_METRICS,
    "Распродажа": SALES_REQUIRED_METRICS,
}


def compute_health_report(df_current: pd.DataFrame, months_range: List[str]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    report["missing_columns"] = [col for col in MANDATORY_COLUMNS if col not in df_current.columns]
    available_metrics = set(df_current["Показатель"].dropna().unique())
    report["missing_key_metrics"] = [m for m in KEY_DECISION_METRICS if m not in available_metrics]
    report["tab_dependencies"] = {
        tab: sorted(metric for metric in metrics if metric not in available_metrics)
        for tab, metrics in TAB_METRIC_DEPENDENCIES.items()
    }
    present_months = sorted_months_safe(df_current.get("Месяц"))
    report["missing_months"] = [m for m in months_range if m not in present_months]
    regions = sorted(map(str, df_current.get("Регион", pd.Series(dtype=str)).dropna().unique()))
    missing_coords = [reg for reg in regions if _resolve_region_coordinates_static(reg) is None]
    report["missing_coordinates"] = missing_coords
    report["total_rows"] = int(len(df_current))
    return report


def render_health_check(ctx: PageContext) -> None:
    report = compute_health_report(ctx.df_current, ctx.months_range)
    issues_present = bool(
        report["missing_columns"] or report["missing_key_metrics"] or
        any(report["tab_dependencies"].values()) or report["missing_months"] or report["missing_coordinates"]
    )
    with st.expander("🩺 Health-check данных", expanded=issues_present):
        missing_cols = report["missing_columns"]
        if missing_cols:
            st.warning("Отсутствуют обязательные столбцы: " + ", ".join(missing_cols))
        else:
            st.markdown("- **Обязательные столбцы:** ✅ всё на месте")

        if report["missing_key_metrics"]:
            st.warning("Недостающие ключевые метрики: " + ", ".join(report["missing_key_metrics"]))
        else:
            st.markdown("- **Ключевые метрики:** ✅ в наличии")

        tab_messages = []
        for tab, missing in report["tab_dependencies"].items():
            if missing:
                tab_messages.append(f"{tab}: {', '.join(missing)}")
        if tab_messages:
            st.markdown("- **Что добавить для вкладок:**\n  - " + "\n  - ".join(tab_messages))
        else:
            st.markdown("- **Вкладки:** ✅ все разделы могут работать")

        if report["missing_months"]:
            st.markdown("- **Месяцы без данных:** " + ", ".join(report["missing_months"]))
        else:
            st.markdown("- **Месяцы:** ✅ покрывает выбранный период")

        if report["missing_coordinates"]:
            st.markdown("- **Координаты регионов:** требуется добавить для: " + ", ".join(report["missing_coordinates"]))
        else:
            st.markdown("- **Координаты регионов:** ✅ все распознаны")

        st.caption(f"Строк в текущем наборе: {report['total_rows']:,}".replace(",", " "))


def render_faq_block() -> None:
    with st.expander("📚 FAQ / Формулы", expanded=False):
        st.markdown(
            """
            - **Выручка от распродажи НЮЗ (руб)** — прямое значение из файла, используется как сумма для всех агрегатов.
            - **Процент наценки НЮЗ** — `Получено наценки / Выручка × 100`; помогает оценить маржу распродажи.
            - **Доля ниже займа, %** — `Товар проданный ниже суммы займа / Выручка × 100`; сигнализирует о доле убыточных продаж.
            - **Новые / Уникальные клиенты** — суммарные значения по филиалам; нужны для вкладки «Когорты» и AI-аналитики.
            - **Лидеры и сигналы** опираются на пороги из сайдбара: минимальную наценку, максимальный риск и лимит убытка.
            - Если вкладка не отображает данные, проверьте раздел Health-check: он подскажет, какие столбцы или метрики добавить.
            """
        )


def _top_regions_by_metric(df_source: pd.DataFrame, regions_all: List[str], months_range: List[str], metric: str, *, top_n: int = 5, ascending: bool = False) -> List[str]:
    values = period_values_by_region_from_itogo(df_source, regions_all, metric, months_range)
    if not values:
        return []
    filtered = [(reg, val) for reg, val in values.items() if val is not None and not pd.isna(val)]
    if not filtered:
        return []
    sorted_items = sorted(filtered, key=lambda kv: kv[1], reverse=not ascending)
    return [reg for reg, _ in sorted_items[:top_n]]

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
    Metrics.MARKUP_PCT.value: "Наценка на распродажу: отношение суммы полученной наценки к выручке от распродажи (Получено наценки / Выручка × 100). Показывает, насколько итоговая цена превышает оценочную стоимость; высокий % означает, что распродажа защищает маржу.",
    Metrics.RISK_SHARE.value: "Доля продаж ниже займа: вычисляется как 'Товар проданный ниже суммы займа НЮЗ (руб)' / 'Выручка от распродажи НЮЗ (руб)' × 100. Рост показателя означает рост убыточных реализаций и напрямую связан с риском.",
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
    Metrics.DEBT_NO_SALE.value: "Задолженность без распродажи НЮЗ на конец месяца (снимок). В периоде считаем среднее/последний, а не сумму.",
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


def _maybe_scale_percent(metric: str, value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return value
    if not is_percent_metric(metric):
        return value
    if abs(float(value)) <= 1.5:
        return float(value) * 100.0
    return float(value)

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


# --- AI анализ ---
AI_METRICS_FOCUS = [
    Metrics.REVENUE.value,
    Metrics.LOAN_ISSUE.value,
    Metrics.MARKUP_PCT.value,
    Metrics.YIELD.value,
    Metrics.RISK_SHARE.value,
    Metrics.ILLIQUID_BY_VALUE_PCT.value,
    Metrics.REDEEMED_SHARE_PCT.value,
    Metrics.BRANCH_COUNT.value,
]

AI_REGION_METRICS = [
    Metrics.REVENUE.value,
    Metrics.MARKUP_PCT.value,
    Metrics.RISK_SHARE.value,
    Metrics.ILLIQUID_BY_VALUE_PCT.value,
]


def _forecast_target_label(last_month: str) -> str:
    if last_month in ORDER:
        idx = ORDER.index(last_month)
        if idx + 1 < len(ORDER):
            return ORDER[idx + 1]
        return f"{ORDER[0]} (следующий год)"
    return "Следующий период"


def _format_value_for_metric(metric: str, value: float | None) -> str:
    if value is None or pd.isna(value):
        return "нет данных"
    if is_percent_metric(metric):
        return f"{value:.2f}%"
    if "руб" in metric:
        return f"{value:,.0f} руб".replace(",", " ")
    if "дней" in metric:
        return f"{value:.2f} дней"
    return f"{value:,.0f}".replace(",", " ")


def _format_metric_for_prompt(metric: str, value: float | None) -> str:
    val = _format_value_for_metric(metric, value)
    if val == "нет данных":
        return f"{metric}: нет данных"
    return f"{metric}: {val}"


def _format_pct_change_text(value: float | None) -> str:
    if value is None:
        return "—"
    arrow = "↑" if value > 0 else ("↓" if value < 0 else "↔")
    return f"{arrow} {value * 100:+.1f}%"


def _collect_period_metrics(df_source: pd.DataFrame, regions: List[str], months_range: List[str]) -> Dict[str, float | None]:
    data: Dict[str, float | None] = {}
    for metric in AI_METRICS_FOCUS:
        data[metric] = period_value_from_itogo(df_source, regions, metric, months_range)
    return data


def _collect_comparison_metrics(df_a: pd.DataFrame, df_b: pd.DataFrame, regions: List[str], months_range: List[str]) -> Tuple[Dict[str, float | None], Dict[str, float | None]]:
    return _collect_period_metrics(df_a, regions, months_range), _collect_period_metrics(df_b, regions, months_range)


@st.cache_data(show_spinner=False, max_entries=512)
def _monthly_series_for_metric(df_source: pd.DataFrame, regions: List[str], metric: str, months_range: List[str]) -> pd.Series:
    months_tuple = tuple(months_range)
    ser = month_series_from_file(df_source, tuple(regions), metric, months_tuple)
    if ser.empty:
        return ser
    ser = ser.reindex(months_tuple).fillna(0.0)
    ser = pd.to_numeric(ser, errors="coerce").fillna(0.0)
    return ser.astype(float)


def _format_monthly_series(metric: str, series: pd.Series) -> str:
    if series is None or series.empty:
        return ""
    parts = []
    for month, val in series.items():
        if val is None or pd.isna(val):
            continue
        parts.append(f"{month}={_format_value_for_metric(metric, val)}")
    return "; ".join(parts)


def _regional_snapshot_for_metric(df_source: pd.DataFrame, regions: List[str], months_range: List[str], metric: str, top_n: int = 3) -> str:
    values = period_values_by_region_from_itogo(df_source, regions, metric, months_range)
    if not values:
        return ""
    items: List[Tuple[str, float | None]] = []
    for reg, val in values.items():
        items.append((reg, None if val is None else float(val)))
    need_desc = metric in METRICS_SMALLER_IS_BETTER
    items.sort(key=lambda kv: (float('inf') if kv[1] is None else kv[1]), reverse=not need_desc)
    top = items[:top_n]
    bottom = []
    if len(items) > top_n:
        bottom_raw = items[-top_n:]
        bottom = bottom_raw if need_desc else list(reversed(bottom_raw))

    def fmt_entry(entry):
        reg, val = entry
        return f"{reg}: {_format_value_for_metric(metric, val)}"

    parts = []
    if top:
        parts.append("ТОП: " + ", ".join(fmt_entry(row) for row in top))
    if bottom:
        parts.append("Низ: " + ", ".join(fmt_entry(row) for row in bottom))
    return " | ".join(parts)


def _regional_context_block(df_source: pd.DataFrame, regions: List[str], months_range: List[str], metrics_subset: List[str]) -> str:
    lines = []
    for metric in metrics_subset:
        snap = _regional_snapshot_for_metric(df_source, regions, months_range, metric)
        if snap:
            lines.append(f"{metric}: {snap}")
    return "\n".join(lines)


def _build_metric_matrix(df_source: pd.DataFrame, regions: List[str], months_range: List[str], metrics: List[str]) -> pd.DataFrame:
    scoped: Dict[str, pd.Series] = {}
    for metric in metrics:
        series = _monthly_series_for_metric(df_source, tuple(regions), metric, months_range)
        if series is None or series.empty:
            continue
        scoped[metric] = pd.to_numeric(series, errors="coerce")
    if not scoped:
        return pd.DataFrame()
    matrix = pd.DataFrame(scoped)
    matrix = matrix.dropna(how="all")
    matrix = matrix.dropna(axis=1, how="all")
    return matrix


def _future_month_labels(last_month: str, horizon: int) -> List[str]:
    if last_month in ORDER:
        base_index = ORDER.index(last_month)
    else:
        base_index = len(ORDER) - 1
    labels: List[str] = []
    for step in range(1, horizon + 1):
        idx = (base_index + step) % len(ORDER)
        label = ORDER[idx]
        if base_index + step >= len(ORDER):
            label += " (след.)"
        labels.append(label)
    return labels


def _linear_trend_forecast(series: pd.Series, horizon: int = 3) -> Dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty or len(clean) < 2:
        return {}
    y = clean.values.astype(float)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residuals = y - fitted
    sigma = float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0
    forecast_vals: List[float] = []
    lower_vals: List[float] = []
    upper_vals: List[float] = []
    for step in range(1, horizon + 1):
        t = len(y) + step - 1
        value = intercept + slope * t
        if sigma == 0.0:
            band = 0.0
        else:
            band = 1.96 * sigma * np.sqrt(1 + step / max(len(y), 1))
        forecast_vals.append(float(value))
        lower_vals.append(float(value - band))
        upper_vals.append(float(value + band))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "sigma": sigma,
        "forecast": forecast_vals,
        "lower": lower_vals,
        "upper": upper_vals,
        "fitted": fitted,
        "residuals": residuals,
        "method": "linear",
        "seasonal": None,
        "description": "Линейный тренд по последним месяцам",
    }


def _seasonal_linear_forecast(series: pd.Series, horizon: int = 3) -> Dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty or len(clean) < 4:
        return {}
    y = clean.values.astype(float)
    labels = [str(idx) for idx in series.dropna().index]
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residuals = y - fitted
    if len(residuals) < 2:
        return {}
    seasonal_groups: Dict[int, List[float]] = defaultdict(list)
    for idx, label in enumerate(labels):
        month_idx = ORDER.index(label) if label in ORDER else idx % len(ORDER)
        seasonal_groups[month_idx].append(residuals[idx])
    if not seasonal_groups:
        return {}
    seasonal_adjustment = {k: float(np.mean(v)) for k, v in seasonal_groups.items() if v}
    sigma = float(residuals.std(ddof=1))
    future_labels = _future_month_labels(labels[-1], horizon)
    forecast_vals: List[float] = []
    lower_vals: List[float] = []
    upper_vals: List[float] = []
    for step, future_label in enumerate(future_labels, start=1):
        t = len(y) + step - 1
        base_value = intercept + slope * t
        month_key = future_label.split()[0]
        month_idx = ORDER.index(month_key) if month_key in ORDER else t % len(ORDER)
        seasonal = seasonal_adjustment.get(month_idx, 0.0)
        value = base_value + seasonal
        band = 0.0 if sigma == 0.0 else 1.96 * sigma * np.sqrt(1 + step / max(len(y), 1))
        forecast_vals.append(float(value))
        lower_vals.append(float(value - band))
        upper_vals.append(float(value + band))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "sigma": sigma,
        "forecast": forecast_vals,
        "lower": lower_vals,
        "upper": upper_vals,
        "fitted": fitted,
        "residuals": residuals,
        "seasonal": seasonal_adjustment,
        "method": "seasonal_linear",
        "description": "Линейный тренд с поправкой на сезонность по месяцам",
    }


def _choose_forecast_model(series: pd.Series, horizon: int = 3) -> Dict[str, Any]:
    linear = _linear_trend_forecast(series, horizon)
    seasonal = _seasonal_linear_forecast(series, horizon)
    if not linear and not seasonal:
        return {}
    if not seasonal:
        return linear
    if not linear:
        return seasonal
    def _sse(bundle):
        residuals = bundle.get("residuals")
        if residuals is None:
            return float("inf")
        return float(np.sum(np.square(residuals)))
    linear_sse = _sse(linear)
    seasonal_sse = _sse(seasonal)
    # если сезонность ощутимо лучше (на 5% и более), используем её
    if seasonal_sse < linear_sse * 0.95:
        return seasonal | {"selected_model": "seasonal", "sse": seasonal_sse, "baseline_sse": linear_sse}
    return linear | {"selected_model": "linear", "sse": linear_sse, "baseline_sse": seasonal_sse}


def _prepare_forecast(df_source: pd.DataFrame, regions: List[str], months_range: List[str], metric: str, horizon: int = 3) -> Dict[str, Any]:
    history = _monthly_series_for_metric(df_source, tuple(regions), metric, months_range)
    if history is None or history.empty:
        return {}
    history = pd.to_numeric(history, errors="coerce").dropna()
    if history.empty:
        return {}
    trend = _choose_forecast_model(history, horizon=horizon)
    if not trend:
        return {}
    future_labels = _future_month_labels(str(history.index[-1]), horizon)
    cleaned = {k: v for k, v in trend.items() if k not in {"residuals", "fitted"}}
    return {
        "history": history,
        "future_labels": future_labels,
        **cleaned,
    }


def _monthly_context_block(df_source: pd.DataFrame, regions: List[str], months_range: List[str], metrics_subset: List[str]) -> str:
    lines = []
    for metric in metrics_subset:
        series = _monthly_series_for_metric(df_source, regions, metric, months_range)
        text = _format_monthly_series(metric, series)
        if text:
            lines.append(f"{metric}: {text}")
    return "\n".join(lines)


def _render_insights(title: str, lines: List[str]) -> None:
    clean = [line for line in lines if line]
    if not clean:
        return
    bullets = "\n".join(f"- {line}" for line in clean)
    st.markdown(f"**{title}**\n{bullets}")


def _render_plan(title: str, lines: List[str]) -> None:
    clean = [line for line in lines if line]
    if not clean:
        return
    numbered = "\n".join(f"{idx}. {line}" for idx, line in enumerate(clean, start=1))
    st.markdown(f"**{title}**\n{numbered}")


def _sanitize_metric_label(label: str) -> str:
    return re.sub(r"\(.*?\)", "", str(label)).strip().lower()


def _unique_lines(lines: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for line in lines:
        if line and line not in seen:
            result.append(line)
            seen.add(line)
    return result


def _monthly_diagnostics(series: pd.Series, metric: str) -> List[str]:
    lines: List[str] = []
    s = series.dropna()
    if s.empty:
        return lines

    first_month = str(s.index[0])
    last_month = str(s.index[-1])
    start_val = _format_value_for_metric(metric, s.iloc[0])
    end_val = _format_value_for_metric(metric, s.iloc[-1])
    if first_month != last_month:
        direction = "вырос" if s.iloc[-1] > s.iloc[0] else ("снизился" if s.iloc[-1] < s.iloc[0] else "остался на уровне")
        lines.append(f"{metric}: {direction} с {start_val} в {first_month} до {end_val} в {last_month}.")
    else:
        lines.append(f"{metric}: {first_month} = {start_val}.")

    if len(s) >= 2:
        diff = s.diff().dropna()
        if not diff.empty:
            inc_month = diff.idxmax()
            inc_val = diff.loc[inc_month]
            if inc_val > 0:
                prev_idx = list(s.index).index(inc_month) - 1
                if prev_idx >= 0:
                    prev_month = s.index[prev_idx]
                    prev_val = _format_value_for_metric(metric, s.loc[prev_month])
                    lines.append(f"Наибольший прирост в {inc_month}: +{_format_delta(metric, inc_val)} (с {prev_val}).")
            dec_month = diff.idxmin()
            dec_val = diff.loc[dec_month]
            if dec_val < 0:
                prev_idx = list(s.index).index(dec_month) - 1
                if prev_idx >= 0:
                    prev_month = s.index[prev_idx]
                    prev_val = _format_value_for_metric(metric, s.loc[prev_month])
                    lines.append(f"Наибольшее падение в {dec_month}: {_format_delta(metric, dec_val)} (с {prev_val}).")

    extrema = {
        "max": s.idxmax() if not s.empty else None,
        "min": s.idxmin() if not s.empty else None,
    }
    if extrema["max"] is not None:
        lines.append(f"Пик: {extrema['max']} — {_format_value_for_metric(metric, s.loc[extrema['max']])}.")
    if extrema["min"] is not None and extrema["min"] != extrema["max"]:
        lines.append(f"Минимум: {extrema['min']} — {_format_value_for_metric(metric, s.loc[extrema['min']])}.")
    return lines


def _plot_metric_trend(metric: str, series: pd.Series, title: str) -> go.Figure:
    s = series.dropna()
    fig = go.Figure()
    if not s.empty:
        fig.add_trace(go.Scatter(
            x=list(map(str, s.index)),
            y=s.values,
            mode="lines+markers",
            fill="tozeroy",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=6)
        ))
    tickfmt, suf = y_fmt_for_metric(metric)
    fig.update_layout(
        title=title,
        height=320,
        margin=dict(t=60, l=20, r=20, b=40),
        hovermode="x unified",
        plot_bgcolor="rgba(255,255,255,0.85)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(tickformat=tickfmt, ticksuffix=suf.strip(), title=None)
    fig.update_xaxes(title=None, showgrid=False)
    return fig


def _render_metric_trend_section(
    title: str,
    metrics: List[str],
    df_source: pd.DataFrame,
    regions: List[str],
    months_range: List[str],
    *,
    widget_key: str
) -> None:
    available = []
    for metric in metrics:
        series = _monthly_series_for_metric(df_source, tuple(regions), metric, months_range)
        if not series.empty:
            available.append(metric)
    if not available:
        st.info("Нет данных для динамики.")
        return

    default_metric = available[0]
    chosen_metric = st.selectbox(
        title,
        options=available,
        index=0,
        key=f"trend_{widget_key}"
    )
    series = _monthly_series_for_metric(df_source, tuple(regions), chosen_metric, months_range)
    fig = _plot_metric_trend(chosen_metric, series, chosen_metric)
    st.plotly_chart(fig, use_container_width=True)
    monthly_lines = _monthly_diagnostics(series, chosen_metric)
    _render_insights("Месячная динамика", monthly_lines)


def _describe_metric_series(series: pd.Series, metric: str) -> str | None:
    if series is None:
        return None
    ser = pd.Series(series).dropna()
    if ser.empty:
        return None
    try:
        ser = ser.astype(float)
    except Exception:
        ser = pd.to_numeric(ser, errors="coerce").dropna()
    if ser.empty:
        return None
    is_small_better = metric in METRICS_SMALLER_IS_BETTER
    best_idx = ser.idxmin() if is_small_better else ser.idxmax()
    worst_idx = ser.idxmax() if is_small_better else ser.idxmin()
    best_val = _format_value_for_metric(metric, ser.loc[best_idx])
    best_name = " · ".join(map(str, best_idx)) if isinstance(best_idx, tuple) else str(best_idx)
    if best_idx == worst_idx:
        return f"{metric}: данные только по {best_name} ({best_val})."
    worst_val = _format_value_for_metric(metric, ser.loc[worst_idx])
    worst_name = " · ".join(map(str, worst_idx)) if isinstance(worst_idx, tuple) else str(worst_idx)
    if is_small_better:
        return f"{metric}: минимальный показатель у {best_name} ({best_val}); максимальный — у {worst_name} ({worst_val})."
    return f"{metric}: лидирует {best_name} ({best_val}); отстаёт {worst_name} ({worst_val})."


def _format_delta(metric: str, delta: float) -> str:
    if delta is None or pd.isna(delta):
        return "0"
    if is_percent_metric(metric):
        return f"{delta:+.2f} п.п."
    if "руб" in metric:
        return f"{delta:+,.0f} руб".replace(",", " ")
    if "дней" in metric:
        return f"{delta:+.1f} дн."
    return f"{delta:+,.1f}".replace(",", " ")


def _describe_deltas(deltas: List[tuple[str, float]], metric: str) -> str | None:
    meaningful = [(name, float(delta)) for name, delta in deltas if delta is not None and not pd.isna(delta)]
    if not meaningful:
        return None
    pos = [item for item in meaningful if item[1] > 0]
    neg = [item for item in meaningful if item[1] < 0]
    fragments = []
    if pos:
        leader = max(pos, key=lambda x: x[1])
        fragments.append(f"рост у {leader[0]} ({_format_delta(metric, leader[1])})")
    if neg:
        lagger = min(neg, key=lambda x: x[1])
        fragments.append(f"снижение у {lagger[0]} ({_format_delta(metric, lagger[1])})")
    if not fragments:
        # все изменения одной направленности
        top = max(meaningful, key=lambda x: abs(x[1]))
        fragments.append(f"изменение у {top[0]} ({_format_delta(metric, top[1])})")
    return f"{metric}: {', '.join(fragments)}"


DEFAULT_ACTION_TEMPLATES = {
    "high": "Зафиксируйте методику {name}: {metric} держится на уровне {value}. Опишите шаги и обучите соседние команды.",
    "low": "Разберите причину снижения в {name}: {metric} упал до {value}. Назначьте ответственного и план восстановления.",
    "delta_pos": "Рост у {name}: {metric} изменился на {delta}. Снимите драйверы и внедрите контроль устойчивости.",
    "delta_neg": "Просадка в {name}: {metric} изменился на {delta}. Проведите диагностику и задайте сроки коррекции.",
}


ACTION_TEMPLATES = {
    Metrics.REVENUE.value: {
        "high": "Развивайте флагман {name}: выручка {value}. Проверьте наличие дефицитных позиций и подготовьте план масштабирования практик.",
        "low": "Поднимите продажи в {name}: показатель просел до {value}. Проведите экспресс-аудит витрины, запланируйте акцию и усилие маркетинга.",
        "delta_pos": "Продажи в {name} выросли на {delta}. Зафиксируйте акции, трафик и мотивацию — сделайте чек-лист тиражирования.",
        "delta_neg": "Продажи в {name} упали на {delta}. Соберите команду, проверьте ассортимент и запустите программу восстановления.",
    },
    Metrics.MARKUP_PCT.value: {
        "high": "Используйте подход {name}: наценка {value}. Запишите алгоритм оценки и обучите других продавцов.",
        "low": "Повышайте наценку {name}: показатель {value}. Пересмотрите оценку залогов и сценарии продаж, выявите скидки.",
        "delta_pos": "Наценка {name} выросла на {delta}. Протестируйте возможность закрепить уровень и расширить ассортимент с высокой маржой.",
        "delta_neg": "Наценка {name} упала на {delta}. Проведите ревизию скидок, обновите правила уценки и проконтролируйте исполнение.",
    },
    Metrics.RISK_SHARE.value: {
        "high": "Контролируйте риск {name}: доля ниже займа {value}. Оцифруйте оценку и примените в других филиалах.",
        "low": "Сократите риск {name}: доля {value}. Проверьте оценку, работу с клиентами и запретите сделки с уязвимыми товарами.",
        "delta_pos": "Риск в {name} снижается на {delta}. Закрепите дисциплину оценки, проведите обучение и мониторинг.",
        "delta_neg": "Риск в {name} вырос на {delta}. Срочно проведите разбор сделок и пересмотрите лимиты и дисконт.",
    },
    Metrics.ILLIQUID_BY_VALUE_PCT.value: {
        "high": "Склад в норме у {name}: неликвид {value}. Используйте их подход к обороту и контролю остатков.",
        "low": "Снижай неликвид у {name}: доля {value}. Запустите распродажу, разберите причины зависания и перестройте закупки.",
        "delta_pos": "Неликвид {name} сократился на {delta}. Распишите успешные действия как стандарт.",
        "delta_neg": "Неликвид {name} вырос на {delta}. Проведите ревизию склада и составьте график распродажи.",
    },
    Metrics.YIELD.value: {
        "high": "Удержите доходность {name}: {value}. Убедитесь, что процесс взыскания устойчив и не ухудшает клиентский опыт.",
        "low": "Поднимайте доходность {name}: {value}. Проверьте ставки, работу с просрочкой и напоминания клиентам.",
        "delta_pos": "Доходность {name} выросла на {delta}. Закрепите практики работы с долгами и стимулируйте повторение.",
        "delta_neg": "Доходность {name} снизилась на {delta}. Проведите аудит просрочки и скорректируйте стратегию взыскания.",
    },
}


def _action_templates_for_metric(metric: str) -> Dict[str, str]:
    base = DEFAULT_ACTION_TEMPLATES.copy()
    custom = ACTION_TEMPLATES.get(metric, {})
    base.update(custom)
    return base


def _label_from_index(idx) -> str:
    if isinstance(idx, tuple):
        return " · ".join(str(x) for x in idx)
    return str(idx)


def _generate_actions_for_series(series: pd.Series | Dict, metric: str) -> List[str]:
    ser = pd.Series(series).dropna()
    if ser.empty:
        return []
    try:
        ser = ser.astype(float)
    except Exception:
        ser = pd.to_numeric(ser, errors="coerce").dropna()
    if ser.empty:
        return []
    ser.index = [_label_from_index(idx) for idx in ser.index]
    higher_better = metric not in METRICS_SMALLER_IS_BETTER
    best_idx = ser.idxmax() if higher_better else ser.idxmin()
    worst_idx = ser.idxmin() if higher_better else ser.idxmax()
    templates = _action_templates_for_metric(metric)
    lines = []
    best_value = _format_value_for_metric(metric, ser.loc[best_idx])
    lines.append(templates["high"].format(name=best_idx, value=best_value, metric=metric))
    if worst_idx != best_idx and worst_idx in ser.index:
        worst_value = _format_value_for_metric(metric, ser.loc[worst_idx])
        lines.append(templates["low"].format(name=worst_idx, value=worst_value, metric=metric))
    return lines


def _generate_actions_for_deltas(deltas: List[tuple[str, float]], metric: str) -> List[str]:
    meaningful = [(name, float(delta)) for name, delta in deltas if delta is not None and not pd.isna(delta) and abs(delta) > 1e-9]
    if not meaningful:
        return []
    templates = _action_templates_for_metric(metric)
    lines = []
    pos = [item for item in meaningful if item[1] > 0]
    neg = [item for item in meaningful if item[1] < 0]
    if pos:
        leader = max(pos, key=lambda x: x[1])
        lines.append(templates["delta_pos"].format(name=leader[0], delta=_format_delta(metric, leader[1]), metric=metric))
    if neg:
        lagger = min(neg, key=lambda x: x[1])
        lines.append(templates["delta_neg"].format(name=lagger[0], delta=_format_delta(metric, lagger[1]), metric=metric))
    return lines


KEY_DECISION_METRICS = [
    Metrics.LOAN_ISSUE.value,
    Metrics.PENALTIES_RECEIVED.value,
    Metrics.REVENUE.value,
]

SUPPORT_DECISION_METRICS = [
    Metrics.MARKUP_PCT.value,
    Metrics.RISK_SHARE.value,
    Metrics.ILLIQUID_BY_VALUE_PCT.value,
]


METRIC_GUIDE: Dict[str, Dict[str, Dict[str, List[str] | str]]] = {
    Metrics.LOAN_ISSUE.value: {
        "up": {
            "summary": "Выдачи выросли на {delta_pct_pct:.1f}% (с {start} до {current}).",
            "actions": [
                "Проверьте, что фондирование и кассы успевают за ростом.",
                "Зафиксируйте практики лидеров выдач и распространите их на отстающих."
            ],
        },
        "down": {
            "summary": "Выдачи упали на {delta_pct_pct:.1f}% (с {start} до {current}).",
            "actions": [
                "Разберите причину падения по филиалам, где просадка максимальная.",
                "Запустите стимулирующие акции или пересмотрите оценку залога, чтобы вернуть поток."
            ],
        },
        "flat": {
            "summary": "Выдачи держатся на уровне {current} относительно стартового {start}.",
            "actions": [
                "Проверьте локальные отклонения и задайте цели роста по ключевым филиалам.",
                "Подготовьте инициативы по увеличению выдач (маркетинг, партнёрства)."
            ],
        },
    },
    Metrics.PENALTIES_RECEIVED.value: {
        "up": {
            "summary": "Доход от процентов и пеней вырос на {delta_pct_pct:.1f}% (с {start} до {current}).",
            "actions": [
                "Сохраните дисциплину взыскания: убедитесь, что процессы работают стабильно.",
                "Проверьте, не растёт ли нагрузка на клиентов при увеличении ставок и сроков."
            ],
        },
        "down": {
            "summary": "Процентные доходы снизились на {delta_pct_pct:.1f}% (с {start} до {current}).",
            "actions": [
                "Проверьте, не ухудшилась ли дисциплина погашений и контроль просрочки.",
                "Запланируйте меры по восстановлению доходности: корректировка ставок, работа с дебиторкой."
            ],
        },
        "flat": {
            "summary": "Процентные поступления остаются около {current}.",
            "actions": [
                "Сравните регионы: где можно ускорить взыскание без роста просрочки?",
                "Подготовьте меры по повышению доходности портфеля (up-sell, пролонгация)."
            ],
        },
    },
    Metrics.REVENUE.value: {
        "up": {
            "summary": "Выручка от распродажи выросла на {delta_pct_pct:.1f}% (с {start} до {current}).",
            "actions": [
                "Убедитесь, что маржа не проседает — контролируйте наценку.",
                "Поддержите филиалы-лидеры распродажи и масштабируйте их подходы."
            ],
        },
        "down": {
            "summary": "Выручка от распродажи снизилась на {delta_pct_pct:.1f}% (с {start} до {current}).",
            "actions": [
                "Разберите ассортимент и причины падения спроса по ключевым регионам.",
                "Запустите кампанию по активизации продаж и проверьте мотивацию персонала."
            ],
        },
        "flat": {
            "summary": "Выручка держится около {current}; значимых изменений нет.",
            "actions": [
                "Ищите точки дополнительного дохода: спецпредложения, cross-sell.",
                "Проверьте, нет ли скрытых просадок по марже или риску в отдельных филиалах."
            ],
        },
    },
}


SCENARIO_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Ежемесячный контроль": {
        "delta_threshold": 0.03,
        "delta_strong": 0.08,
        "yoy_threshold": 0.05,
        "risk_high": 12.0,
        "illiquid_high": 30.0,
        "markup_drop": -0.03,
        "action_suffix": {
            "up": "Закрепите результат и поделитесь практиками на планёрке.",
            "down": "Назначьте ответственного за восстановление и контроль сроков.",
            "flat": "Определите инициативы для ускорения показателя."
        },
        "intensity": {
            "strong_up": " (значительный рост)",
            "strong_down": " (резкое падение)"
        }
    },
    "Антикризис": {
        "delta_threshold": 0.02,
        "delta_strong": 0.05,
        "yoy_threshold": 0.03,
        "risk_high": 10.0,
        "illiquid_high": 25.0,
        "markup_drop": -0.02,
        "action_suffix": {
            "up": "Удержите тренд: закрепите ключевые действия и добавьте контроль.",
            "down": "Срочно разработайте программу восстановления и согласуйте её с руководством.",
            "flat": "Ищите возможности для quick wins и устранения скрытых провалов."
        },
        "intensity": {
            "strong_up": " (резкий рост — используйте шанс)",
            "strong_down": " (критическое падение — нужен кризисный план)"
        }
    },
    "Планирование": {
        "delta_threshold": 0.04,
        "delta_strong": 0.10,
        "yoy_threshold": 0.07,
        "risk_high": 12.0,
        "illiquid_high": 28.0,
        "markup_drop": -0.04,
        "action_suffix": {
            "up": "Включите этот рост в планы и обеспечьте ресурсами масштабирование.",
            "down": "Закладывайте в план корректирующие мероприятия и дополнительные инвестиции.",
            "flat": "Определите точки ускорения и запланируйте пилоты для роста."
        },
        "intensity": {
            "strong_up": " (значимый рост — закладываем в стратегию)",
            "strong_down": " (существенное падение — пересматриваем план)"
        }
    }
}


SCENARIO_DESCRIPTIONS = {
    "Ежемесячный контроль": "Следим за отклонениями и быстро устраняем локальные проблемы.",
    "Антикризис": "Фокус на рисках и оперативном восстановлении ключевых показателей.",
    "Планирование": "Оцениваем потенциал роста и закладываем инициативы в планы." 
}


@dataclass
class PageContext:
    mode: str  # "single" или "compare"
    df_current: pd.DataFrame
    df_previous: pd.DataFrame | None
    agg_current: pd.DataFrame
    regions: List[str]
    months_range: List[str]
    months_available: List[str]
    scenario_name: str
    year_current: int
    year_previous: int | None
    color_map: Dict[str, str]
    strict_mode: bool
    thresholds: Dict[str, float] | None = None


def _calc_pct_change(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or abs(old) < 1e-9:
        return None
    return (new - old) / old


@st.cache_data(show_spinner=False, max_entries=256)
def compute_metric_stats(df_source: pd.DataFrame, regions: List[str], months_range: List[str], metric: str) -> Dict[str, Any]:
    series = _monthly_series_for_metric(df_source, regions, metric, months_range)
    if series is None:
        series = pd.Series(dtype=float)
    else:
        series = pd.to_numeric(series, errors="coerce")
    series = series.dropna()
    total = period_value_from_itogo(df_source, regions, metric, months_range)
    start = float(series.iloc[0]) if not series.empty else None
    current = float(series.iloc[-1]) if not series.empty else None
    delta_abs = None
    delta_pct = None
    if start is not None and current is not None:
        delta_abs = current - start
        if abs(start) > 1e-9:
            delta_pct = delta_abs / start

    current_month = str(series.index[-1]) if not series.empty else None
    previous_month = str(series.index[-2]) if len(series) >= 2 else None
    mom_pct = None
    if len(series) >= 2 and series.iloc[-2] != 0:
        mom_pct = _calc_pct_change(series.iloc[-1], series.iloc[-2])
    qoq_pct = None
    if len(series) >= 4 and series.iloc[-4] != 0:
        qoq_pct = _calc_pct_change(series.iloc[-1], series.iloc[-4])

    extrema = {}
    if not series.empty:
        if metric in METRICS_SMALLER_IS_BETTER:
            best_idx = series.idxmin(); worst_idx = series.idxmax()
        else:
            best_idx = series.idxmax(); worst_idx = series.idxmin()
        extrema = {
            "best": {"month": best_idx, "value": float(series.loc[best_idx])},
            "worst": {"month": worst_idx, "value": float(series.loc[worst_idx])}
        }

    # год-к-году на начало/конец
    if len(months_range) >= 1:
        try:
            prev_year = df_source["Год"].dropna().astype(int).unique()
        except Exception:
            prev_year = np.array([])
        prev_total = None
        for yr in prev_year:
            mask = df_source["Год"] == yr
            if mask.any():
                prev_total = period_value_from_itogo(df_source[mask], regions, metric, months_range)
                break
    else:
        prev_total = None
    yoy_pct = _calc_pct_change(total, prev_total) if prev_total is not None else None

    if metric in METRICS_LAST and total is not None:
        try:
            current = float(total)
        except (TypeError, ValueError):
            pass

    return {
        "series": series,
        "total": total,
        "start": start,
        "current": current,
        "delta_abs": delta_abs,
        "delta_pct": delta_pct,
        "prev_total": prev_total,
        "yoy_pct": yoy_pct,
        "extrema": extrema,
        "current_month": current_month,
        "previous_month": previous_month,
        "mom_pct": mom_pct,
        "qoq_pct": qoq_pct,
    }


def _interpret_metric(metric: str, stats: Dict[str, Any], scenario_conf: Dict[str, Any], period_start: str,
                      stats_map: Dict[str, Dict[str, Any]], months_range: List[str],
                      baseline_stats: Dict[str, Any] | None = None) -> tuple[str | None, List[str], List[str], str]:
    current = stats.get("current")
    start = stats.get("start")
    delta_abs = stats.get("delta_abs")
    delta_pct = stats.get("delta_pct")
    total = stats.get("total")
    baseline_total = baseline_stats.get("total") if baseline_stats else None
    yoy_pct = _calc_pct_change(total, baseline_total)

    current_str = _format_value_for_metric(metric, current) if current is not None else "нет данных"
    start_str = _format_value_for_metric(metric, start) if start is not None else current_str
    delta_abs_str = _format_delta(metric, delta_abs) if delta_abs is not None else "0"
    delta_pct_pct = (delta_pct * 100) if delta_pct is not None else 0.0
    yoy_pct_pct = (yoy_pct * 100) if yoy_pct is not None else None

    format_params = {
        "start": start_str,
        "current": current_str,
        "delta_abs": delta_abs_str,
        "delta_pct_pct": abs(delta_pct_pct),
        "period_start": period_start,
        "metric": metric,
    }

    status = evaluate_metric_status(
        metric,
        current,
        scenario_conf,
        delta_pct=delta_pct,
        yoy_pct=yoy_pct,
    )

    guide = METRIC_GUIDE.get(metric, {})
    base_status = status
    if status in {"risk_high", "margin_drop"}:
        base_status = "down"
    if status == "no_data":
        base_status = "flat"
    if status not in {"strong_up", "up", "strong_down", "down"}:
        # fall back to delta-based buckets for guide selection
        if delta_pct is not None:
            if delta_pct >= scenario_conf["delta_threshold"]:
                base_status = "up"
            elif delta_pct <= -scenario_conf["delta_threshold"]:
                base_status = "down"
            else:
                base_status = "flat"

    entry = guide.get(status) or guide.get(base_status) or guide.get("flat")
    summary: str | None = None
    actions: List[str] = []
    highlights: List[str] = []
    if entry:
        summary_template = entry.get("summary")
        if summary_template:
            summary = summary_template.format(**format_params)
            intensity = scenario_conf["intensity"].get(status)
            if intensity:
                summary += intensity
        action_templates = entry.get("actions", [])
        for template in action_templates:
            actions.append(template.format(**format_params))

    if delta_pct is None and summary is None:
        summary = f"{metric}: текущий уровень {current_str}."

    if delta_pct is not None and abs(delta_pct) < scenario_conf["delta_threshold"] and summary:
        summary += " (без существенных изменений)"

    if yoy_pct is not None and summary:
        if abs(yoy_pct) >= scenario_conf["yoy_threshold"]:
            summary += f" Год к году: {yoy_pct_pct:+.1f}%."

    scenario_suffix = scenario_conf["action_suffix"].get(base_status)
    if scenario_suffix:
        actions.append(scenario_suffix)

    extrema = stats.get("extrema") or {}
    best = extrema.get("best")
    worst = extrema.get("worst")
    if best and best.get("value") is not None:
        highlights.append(f"Максимум: {best['month']} — {_format_value_for_metric(metric, best['value'])}.")
    if worst and worst.get("value") is not None and (not best or worst['month'] != best['month']):
        highlights.append(f"Минимум: {worst['month']} — {_format_value_for_metric(metric, worst['value'])}.")

    if delta_pct is not None and months_range and start is not None and current is not None:
        first_month = months_range[0]
        last_month = months_range[-1]
        direction = "вырос" if delta_pct > 0 else ("снизился" if delta_pct < 0 else "без существенных изменений")
        highlights.append(
            f"{metric}: {direction} с {_format_value_for_metric(metric, start)} в {first_month} до {_format_value_for_metric(metric, current)} в {last_month}."
        )

    # Дополнительные контрмеры по поддерживающим метрикам
    markup_stats = stats_map.get(Metrics.MARKUP_PCT.value)
    if metric == Metrics.REVENUE.value and markup_stats:
        markup_delta = markup_stats.get("delta_pct")
        if markup_delta is not None and markup_delta < scenario_conf["markup_drop"]:
            summary = (summary or "") + " Наценка снижается — рост выручки съедает маржу."

    return summary, actions, highlights, status


def build_metric_recommendations(stats_map: Dict[str, Dict[str, Any]], scenario_name: str, months_range: List[str], baseline_map: Dict[str, Dict[str, Any]] | None = None) -> tuple[List[str], List[str]]:
    config = SCENARIO_CONFIGS[scenario_name]
    period_start = months_range[0] if months_range else "начало периода"
    summary_lines: List[str] = []
    action_lines: List[str] = []
    for metric in KEY_DECISION_METRICS:
        stats = stats_map.get(metric)
        if not stats:
            continue
        baseline_stats = baseline_map.get(metric) if baseline_map else None
        summary, actions, highlights, _ = _interpret_metric(metric, stats, config, period_start, stats_map, months_range, baseline_stats)
        if summary:
            summary_lines.append(summary)
        action_lines.extend(actions)
        summary_lines.extend(highlights)

    risk_stats = stats_map.get(Metrics.RISK_SHARE.value)
    if risk_stats and risk_stats.get("current") is not None:
        risk_val = float(risk_stats["current"])
        if risk_val > config["risk_high"]:
            summary_lines.append(f"Риск: доля ниже займа {risk_val:.1f}% — выше порога {config['risk_high']}%.")
            action_lines.append("Проведите экспресс-аудит оценки залогов и ограничьте выдачи по проблемным категориям.")
        else:
            baseline_risk = baseline_map.get(Metrics.RISK_SHARE.value)["current"] if baseline_map and baseline_map.get(Metrics.RISK_SHARE.value) else None
            if baseline_risk and risk_val < float(baseline_risk) - 1:
                summary_lines.append(f"Риск снижается: доля ниже займа {risk_val:.1f}% (было {float(baseline_risk):.1f}%).")

    illiquid_stats = stats_map.get(Metrics.ILLIQUID_BY_VALUE_PCT.value)
    if illiquid_stats and illiquid_stats.get("current") is not None:
        illiquid_val = float(illiquid_stats["current"])
        if illiquid_val > config["illiquid_high"]:
            summary_lines.append(f"Неликвид: {illiquid_val:.1f}% — выше допустимого {config['illiquid_high']}%.")
            action_lines.append("Запустите программу распродажи неликвида и пересмотрите закупочную политику.")

    markup_stats = stats_map.get(Metrics.MARKUP_PCT.value)
    if markup_stats and markup_stats.get("current") is not None and markup_stats.get("delta_pct") is not None:
        if markup_stats["delta_pct"] < config["markup_drop"]:
            markup_summary = f"Маржинальность падает до {_format_value_for_metric(Metrics.MARKUP_PCT.value, markup_stats['current'])}."
            if all("марж" not in line.lower() for line in summary_lines):
                summary_lines.append(markup_summary)
            action_lines.append("Проверьте скидки, контроль оценок и мотивацию продавцов.")

    return summary_lines, action_lines


STATUS_SEVERITY = {
    "risk_high": 0,
    "strong_down": 1,
    "margin_drop": 2,
    "down": 3,
    "flat": 4,
    "up": 5,
    "strong_up": 6,
    "no_data": 7,
}

STATUS_LABELS = {
    "risk_high": "🚨 Высокий риск",
    "strong_down": "🔴 Резкое падение",
    "margin_drop": "🟠 Падение маржи",
    "down": "🟠 Снижение",
    "flat": "⚪️ Стабильно",
    "up": "🟢 Рост",
    "strong_up": "🟢 Рост+",
    "no_data": "⚙️ Недостаточно данных",
}

STATUS_ICONS = {
    "risk_high": "🚨",
    "strong_down": "🔴",
    "margin_drop": "🟠",
    "down": "🟠",
    "flat": "⚪️",
    "up": "🟢",
    "strong_up": "🟢",
    "no_data": "⚙️",
}

STATUS_COLORS = {
    "risk_high": "#b91c1c",
    "strong_down": "#dc2626",
    "margin_drop": "#f97316",
    "down": "#f59e0b",
    "flat": "#6b7280",
    "up": "#10b981",
    "strong_up": "#047857",
    "no_data": "#94a3b8",
}


def evaluate_metric_status(
    metric: str,
    value: float | None,
    config: Dict[str, Any],
    *,
    delta_pct: float | None = None,
    yoy_pct: float | None = None,
) -> str:
    if value is None or pd.isna(value):
        return "no_data"

    if metric == Metrics.RISK_SHARE.value and value > config["risk_high"]:
        return "risk_high"

    if metric == Metrics.ILLIQUID_BY_VALUE_PCT.value and value > config["illiquid_high"]:
        return "risk_high"

    if metric == Metrics.MARKUP_PCT.value:
        if delta_pct is not None and delta_pct < config["markup_drop"]:
            return "margin_drop"
        if yoy_pct is not None and yoy_pct < config["markup_drop"]:
            return "margin_drop"

    candidate = "flat"

    def classify(delta: float) -> str:
        if delta >= config["delta_strong"]:
            return "strong_up"
        if delta >= config["delta_threshold"]:
            return "up"
        if delta <= -config["delta_strong"]:
            return "strong_down"
        if delta <= -config["delta_threshold"]:
            return "down"
        return "flat"

    if delta_pct is not None:
        candidate = classify(delta_pct)

    if candidate in {"flat", "up", "strong_up"} and yoy_pct is not None:
        yoy_candidate = classify(yoy_pct)
        if STATUS_SEVERITY.get(yoy_candidate, 4) < STATUS_SEVERITY.get(candidate, 4):
            candidate = yoy_candidate

    return candidate


def compute_alert_streak(metric: str, stats: Dict[str, Any], config: Dict[str, Any]) -> int:
    series = stats.get("series")
    if series is None or series.empty:
        return 0
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0
    streak = 0
    for idx in range(len(values) - 1, -1, -1):
        current = values.iloc[idx]
        prev = values.iloc[idx - 1] if idx > 0 else None
        delta_pct = None
        if prev is not None and prev != 0:
            delta_pct = (current - prev) / prev
        status = evaluate_metric_status(
            metric,
            current,
            config,
            delta_pct=delta_pct,
            yoy_pct=None,
        )
        if STATUS_SEVERITY.get(status, 4) > STATUS_SEVERITY["down"]:
            break
        streak += 1
    return streak


def forecast_next_value(stats: Dict[str, Any]) -> float | None:
    series = stats.get("series")
    if series is None or series.empty:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    if len(values) == 1:
        return float(values.iloc[-1])
    recent = values.iloc[-3:] if len(values) >= 3 else values
    deltas = recent.diff().dropna()
    if deltas.empty:
        delta = 0.0
    else:
        delta = float(deltas.mean())
    return float(values.iloc[-1] + delta)


def _format_forecast(metric: str, value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    if is_percent_metric(metric):
        return f"{value:.2f}%"
    if "руб" in metric:
        return f"{value:,.0f} ₽".replace(",", " ")
    if "дней" in metric:
        return f"{value:.1f} дн."
    return f"{value:,.0f}".replace(",", " ")


def build_metric_dashboard(
    stats_current: Dict[str, Dict[str, Any]],
    stats_previous: Dict[str, Dict[str, Any]] | None,
    scenario_name: str,
    months_range: List[str],
) -> tuple[pd.DataFrame, List[str], List[Dict[str, Any]]]:
    scenario_conf = SCENARIO_CONFIGS[scenario_name]
    metrics_sequence = list(stats_current.keys())
    period_start = months_range[0] if months_range else "начало периода"
    rows: List[Dict[str, Any]] = []
    priority_actions: List[str] = []
    alerts: List[Dict[str, Any]] = []

    for metric in metrics_sequence:
        stats = stats_current.get(metric)
        if not stats:
            continue
        baseline_stats = stats_previous.get(metric) if stats_previous else None
        summary, actions, highlights, status = _interpret_metric(
            metric,
            stats,
            scenario_conf,
            period_start,
            stats_current,
            months_range,
            baseline_stats,
        )

        label = STATUS_LABELS.get(status, STATUS_LABELS["flat"])
        severity = STATUS_SEVERITY.get(status, STATUS_SEVERITY["flat"])

        delta_abs = stats.get("delta_abs")
        delta_pct = stats.get("delta_pct")
        if delta_abs is not None:
            delta_text = _format_delta(metric, float(delta_abs))
            if delta_pct is not None:
                delta_text += f" ({delta_pct * 100:+.1f}%)"
        else:
            delta_text = "—"

        yoy_pct = stats.get("yoy_pct")
        yoy_text = f"{yoy_pct * 100:+.1f}%" if yoy_pct is not None else "—"

        current_val = stats.get("current")
        current_text = _format_value_for_metric(metric, current_val)
        comment = summary or (highlights[0] if highlights else "—")
        recommendation = actions[0] if actions else "—"

        streak = compute_alert_streak(metric, stats, scenario_conf)
        streak_text = str(streak) if streak > 0 else "—"

        forecast_val = forecast_next_value(stats)
        forecast_text = _format_forecast(metric, forecast_val)

        rows.append(
            {
                "Показатель": metric,
                "Статус": label,
                "Текущий уровень": current_text,
                "Δ с начала периода": delta_text,
                "Δ к прошлому году": yoy_text,
                "Стрик тревоги (мес.)": streak_text,
                "Прогноз следующего периода": forecast_text,
                "Комментарий": comment,
                "Рекомендация": recommendation,
                "__severity": severity,
                "__status_code": status,
            }
        )

        if recommendation != "—" and severity <= STATUS_SEVERITY.get("down", 3):
            icon = STATUS_ICONS.get(status, "•")
            if streak > 1:
                priority_actions.append(f"{icon} {metric}: {recommendation} (стрик {streak})")
            else:
                priority_actions.append(f"{icon} {metric}: {recommendation}")

        series = stats.get("series")
        series_numeric = None
        if series is not None:
            try:
                series_numeric = pd.to_numeric(series, errors="coerce").dropna()
            except Exception:
                series_numeric = None

        alerts.append(
            {
                "metric": metric,
                "status": status,
                "severity": severity,
                "label": label,
                "icon": STATUS_ICONS.get(status, "•"),
                "current_text": current_text,
                "delta_text": delta_text,
                "yoy_text": yoy_text,
                "streak": streak,
                "forecast_text": forecast_text,
                "comment": comment,
                "recommendation": recommendation,
                "series": series_numeric,
            }
        )

    if not rows:
        return pd.DataFrame(), [], alerts

    df_board = pd.DataFrame(rows)
    df_board = df_board.sort_values(["__severity", "Показатель"]).reset_index(drop=True)
    df_board = df_board.drop(columns=["__severity", "__status_code"])
    column_order = [
        "Показатель",
        "Статус",
        "Текущий уровень",
        "Δ с начала периода",
        "Δ к прошлому году",
        "Стрик тревоги (мес.)",
        "Прогноз следующего периода",
        "Комментарий",
        "Рекомендация",
    ]
    df_board = df_board[column_order]
    df_board = df_board.fillna("—")
    alerts.sort(key=lambda item: (item["severity"], item["metric"]))
    return df_board, priority_actions, alerts


def render_executive_summary(
    ctx: PageContext,
    stats_current: Dict[str, Dict[str, Any]],
    stats_previous: Dict[str, Dict[str, Any]] | None,
) -> None:
    st.markdown("### 🧭 Executive summary")
    board, priority_actions, alerts = build_metric_dashboard(stats_current, stats_previous, ctx.scenario_name, ctx.months_range)
    if alerts:
        render_severity_ribbon(alerts)
        render_alert_cards(alerts, max_cards=3)
    if board.empty:
        st.info("Недостаточно данных для сводки по метрикам.")
    else:
        st.markdown("#### 📊 Таблица сигналов")
        st.dataframe(board, use_container_width=True, hide_index=True)
    if priority_actions:
        _render_plan("Приоритетные шаги", priority_actions[:5])
    else:
        st.caption("Сигналы не обнаружены — метрики в допустимых пределах.")

def render_tab_summary(
    ctx: PageContext,
    metrics: List[str],
    *,
    title: str,
) -> None:
    stats_current = {
        metric: compute_metric_stats(ctx.df_current, ctx.regions, ctx.months_range, metric)
        for metric in metrics
    }
    stats_previous = None
    if ctx.mode == "compare" and ctx.df_previous is not None:
        stats_previous = {
            metric: compute_metric_stats(ctx.df_previous, ctx.regions, ctx.months_range, metric)
            for metric in metrics
        }
    st.markdown(title)
    board, priority_actions, alerts = build_metric_dashboard(
        stats_current,
        stats_previous,
        ctx.scenario_name,
        ctx.months_range,
    )
    if alerts:
        render_severity_ribbon(alerts, max_items=3)
        render_alert_cards(alerts, max_cards=2)
    if board.empty:
        st.info("Недостаточно данных для автоматической сводки.")
    else:
        st.dataframe(board, use_container_width=True, hide_index=True)
    if priority_actions:
        _render_plan("В первую очередь", priority_actions[:5])
    else:
        st.caption("Тревожных сигналов по этим показателям нет.")


def _render_alert_sparkline(alert: Dict[str, Any], *, chart_key: str | None = None) -> None:
    series = alert.get("series")
    if series is None or len(series) < 2:
        return
    data = series.tail(6)
    try:
        x_values = [str(x) for x in data.index]
        y_values = data.values.astype(float)
    except Exception:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines+markers",
        line=dict(color="#2563eb", width=2),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.15)",
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=110,
        margin=dict(l=0, r=0, t=20, b=10),
        showlegend=False,
        template="plotly_white",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    if chart_key is None:
        metric_slug = _normalize_metric_label(alert.get("metric", "metric"))
        chart_key = f"spark_{metric_slug}_{hash(tuple(x_values)) & 0xFFFF}"
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=chart_key)


def render_severity_ribbon(alerts: List[Dict[str, Any]], *, max_items: int = 4) -> None:
    if not alerts:
        return
    critical = [a for a in alerts if a["severity"] <= STATUS_SEVERITY["down"]]
    shortlisted = critical or alerts
    shortlisted = sorted(shortlisted, key=lambda a: (a["severity"], a["metric"]))[:max_items]
    cols = st.columns(len(shortlisted))
    for col, alert in zip(cols, shortlisted):
        color = STATUS_COLORS.get(alert["status"], "#6b7280")
        chip = (
            f"<div style='background:{color};color:white;padding:6px 12px;"
            "border-radius:999px;font-weight:600;text-align:center;'>"
            f"{alert['icon']} {alert['metric']}</div>"
        )
        col.markdown(chip, unsafe_allow_html=True)
        subtitle = alert["comment"]
        if (not subtitle) or subtitle == "—":
            subtitle = alert["recommendation"]
        if subtitle and subtitle != "—":
            if len(subtitle) > 140:
                subtitle = subtitle[:137] + "…"
            col.caption(subtitle)


def render_alert_cards(alerts: List[Dict[str, Any]], *, max_cards: int = 3) -> None:
    if not alerts:
        st.caption("Тревожных сигналов пока нет.")
        return
    critical = [a for a in alerts if a["severity"] <= STATUS_SEVERITY["down"]]
    cards = critical[:max_cards] if critical else alerts[:max_cards]
    for idx, alert in enumerate(cards):
        color = STATUS_COLORS.get(alert["status"], "#6b7280")
        with st.container():
            st.markdown(
                f"<div style='padding:14px 18px;border:1px solid rgba(148,163,184,0.35);"
                f"border-radius:16px;margin-bottom:8px;background:rgba(148,163,184,0.05);'>"
                f"<h4 style='margin:0;color:{color};'>{alert['icon']} {alert['metric']}</h4>",
                unsafe_allow_html=True,
            )

            metrics_cols = st.columns(4)
            metrics_cols[0].markdown(
                f"**Текущий уровень**<br>{alert['current_text']}",
                unsafe_allow_html=True,
            )
            metrics_cols[1].markdown(
                f"**Δ периода**<br>{alert['delta_text']}",
                unsafe_allow_html=True,
            )
            metrics_cols[2].markdown(
                f"**Δ к прошлому году**<br>{alert['yoy_text']}",
                unsafe_allow_html=True,
            )
            metrics_cols[3].markdown(
                f"**Прогноз**<br>{alert['forecast_text']}",
                unsafe_allow_html=True,
            )

            if alert.get("streak", 0) > 1:
                st.markdown(
                    f"<span style='color:{color}; font-weight:600;'>⚠️ Уже {alert['streak']} мес. подряд превышение порога.</span>",
                    unsafe_allow_html=True,
                )

            if alert["comment"] and alert["comment"] != "—":
                st.markdown(f"**Что видно:** {alert['comment']}")
            if alert["recommendation"] and alert["recommendation"] != "—":
                st.markdown(f"**Что сделать:** {alert['recommendation']}")
            if alert["forecast_text"] and alert["forecast_text"] != "—":
                st.caption(f"Если тренд сохранится, в следующем периоде ожидаем {alert['forecast_text']}.")

            spark_key = f"spark_card_{uuid4()}"
            _render_alert_sparkline(alert, chart_key=spark_key)

            st.markdown("</div>", unsafe_allow_html=True)


def render_correlation_block(df_source: pd.DataFrame, regions: List[str], months_range: List[str], *, default_metrics: List[str]) -> None:
    st.subheader("📈 Корреляции показателей")
    available = sorted({m for m in df_source["Показатель"].dropna().unique() if m in ACCEPTED_METRICS_CANONICAL and m not in HIDDEN_METRICS})
    if not available:
        st.info("Недостаточно данных для построения корреляций.")
        return
    defaults = [m for m in default_metrics if m in available] or available[: min(len(available), 4)]
    selected = st.multiselect(
        "Метрики для анализа взаимосвязей",
        options=available,
        default=defaults,
        help="Отметьте показатели, чтобы увидеть матрицу корреляций."
    )
    if not selected:
        st.info("Выберите хотя бы одну метрику.")
        return
    matrix = _build_metric_matrix(df_source, regions, months_range, selected)
    if matrix.empty or matrix.shape[1] < 2:
        st.info("Для выбранного набора метрик недостаточно данных.")
        return
    corr = matrix.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title="r"),
    ))
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40), height=420)
    st.plotly_chart(fig, use_container_width=True, key="corr_heatmap")

    pairs: List[Tuple[str, str, float]] = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.isna(val):
                continue
            pairs.append((cols[i], cols[j], float(val)))
    if not pairs:
        return
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    top_pairs = pairs[:5]
    bullets = []
    for left, right, value in top_pairs:
        relation = "прямая" if value > 0 else "обратная"
        bullets.append(f"- **{left} ↔ {right}**: {value:+.2f} ({relation} связь)")
    st.markdown("**Сильнейшие связи:**\n" + "\n".join(bullets))


def render_revenue_waterfall(ctx: PageContext) -> None:
    st.subheader("💧 Вклад регионов в изменение выручки")
    if len(ctx.months_range) < 2:
        st.info("Выберите минимум два месяца, чтобы показать динамику.")
        return
    start_month, end_month = ctx.months_range[0], ctx.months_range[-1]
    subset = ctx.df_current[
        (ctx.df_current["Регион"].isin(ctx.regions)) &
        (ctx.df_current["Показатель"] == Metrics.REVENUE.value) &
        (ctx.df_current["Месяц"].astype(str).isin([start_month, end_month]))
    ]
    if subset.empty:
        subset = ctx.df_current[
            (ctx.df_current["Регион"].isin(ctx.regions)) &
            (ctx.df_current["Показатель"] == Metrics.REVENUE.value) &
            (ctx.df_current["Месяц"].astype(str).isin(ctx.months_range))
        ]
        if subset.empty:
            st.info("Нет данных по выручке для построения диаграммы.")
            return
    grouped = (subset.groupby(["Регион", "Месяц"], observed=True)["Значение"]
                     .sum()
                     .unstack(fill_value=0))
    available_months = [m for m in ctx.months_range if m in grouped.columns]
    if len(available_months) < 2:
        fallback_months = sorted_months_safe(grouped.columns)
        if len(fallback_months) < 2:
            st.info("Недостаточно данных по выбранным месяцам.")
            return
        available_months = [fallback_months[0], fallback_months[-1]]
    start_month, end_month = available_months[0], available_months[-1]
    start_series = grouped.get(start_month, pd.Series(dtype=float)).fillna(0.0)
    end_series = grouped.get(end_month, pd.Series(dtype=float)).fillna(0.0)
    if start_series.empty or end_series.empty:
        st.info("Недостаточно данных по выбранным месяцам.")
        return
    start_total = float(start_series.sum())
    end_total = float(end_series.sum())
    delta = (end_series - start_series).sort_values(key=lambda x: -x.abs())
    if np.isclose(start_total, end_total) and delta.abs().sum() < 1e-6:
        st.caption("Суммарная выручка не изменилась — водопад окажется плоским.")
        return
    measures = ["absolute"] + ["relative"] * len(delta) + ["total"]
    x_axis = ["Начало"] + delta.index.tolist() + ["Конец"]
    y_axis = [start_total] + delta.values.tolist() + [end_total]
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=x_axis,
        y=y_axis,
        connector={"line": {"color": "rgba(148,163,184,0.35)"}},
        decreasing={"marker": {"color": "#ef4444"}},
        increasing={"marker": {"color": "#10b981"}},
        totals={"marker": {"color": "#2563eb"}}
    ))
    fig.update_layout(height=420, margin=dict(l=40, r=40, t=40, b=40), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="revenue_waterfall")
    st.caption(f"Старт: {format_rub(start_total)} → Конец: {format_rub(end_total)}.")


def sales_intelligence_block(ctx: PageContext, thresholds: Dict[str, float] | None = None) -> None:
    st.subheader("🧠 Интеллект продаж по регионам")
    st.caption("Сводный взгляд на выручку, маржу и риск по регионам: сверху таблица и рекомендации, ниже — карта наценка ↔ риск.")
    revenue_map = period_values_by_region_from_itogo(ctx.df_current, ctx.regions, Metrics.REVENUE.value, ctx.months_range)
    if not revenue_map:
        st.info("Нет данных по выручке для выбранного окна.")
        return
    markup_map = period_values_by_region_from_itogo(ctx.df_current, ctx.regions, Metrics.MARKUP_PCT.value, ctx.months_range)
    risk_map = period_values_by_region_from_itogo(ctx.df_current, ctx.regions, Metrics.RISK_SHARE.value, ctx.months_range)

    rows: list[dict[str, float | str]] = []
    for region, value in revenue_map.items():
        if value is None or pd.isna(value):
            continue
        rows.append({
            "Регион": region,
            "Выручка, ₽": float(value),
            "Наценка, %": float(markup_map.get(region)) if markup_map and markup_map.get(region) not in (None, np.nan) else np.nan,
            "Риск, %": float(risk_map.get(region)) if risk_map and risk_map.get(region) not in (None, np.nan) else np.nan,
        })
    if not rows:
        st.info("Нет регионов с данными по выручке.")
        return

    df = pd.DataFrame(rows).sort_values("Выручка, ₽", ascending=False)
    total_revenue = float(df["Выручка, ₽"].sum())
    if total_revenue > 0:
        df["Доля, %"] = (df["Выручка, ₽"] / total_revenue) * 100
    if df["Наценка, %"].notna().any():
        mean_markup = float(df["Наценка, %"].dropna().mean())
        df["Δ наценки к средн."] = df["Наценка, %"] - mean_markup
    if thresholds:
        df["Сигнал"] = ""
        min_markup = thresholds.get("min_markup")
        max_risk = thresholds.get("max_risk")
        for idx, row in df.iterrows():
            notes: List[str] = []
            if min_markup is not None and not pd.isna(row.get("Наценка, %")) and row["Наценка, %"] < min_markup:
                notes.append("⬇︎ наценка")
            if max_risk is not None and not pd.isna(row.get("Риск, %")) and row["Риск, %"] > max_risk:
                notes.append("⚠️ риск")
            if notes:
                df.at[idx, "Сигнал"] = ", ".join(notes)
    else:
        df["Сигнал"] = ""
    column_config = {
        "Выручка, ₽": st.column_config.NumberColumn("Выручка, ₽", format="%.0f"),
        "Наценка, %": st.column_config.NumberColumn("Наценка, %", format="%.2f"),
        "Риск, %": st.column_config.NumberColumn("Риск, %", format="%.2f"),
    }
    if "Доля, %" in df.columns:
        column_config["Доля, %"] = st.column_config.NumberColumn("Доля от итога, %", format="%.1f%%")
    if "Δ наценки к средн." in df.columns:
        column_config["Δ наценки к средн."] = st.column_config.NumberColumn("Δ наценки к средн., п.п.", format="%.2f")
    if "Сигнал" in df.columns:
        column_config["Сигнал"] = st.column_config.TextColumn("Сигнал")
    st.dataframe(df, use_container_width=True, hide_index=True, column_config=column_config)

    insights: list[str] = []
    top_row = df.iloc[0]
    if "Доля, %" in df.columns:
        insights.append(f"Лидер по выручке — {top_row['Регион']}: {format_rub(top_row['Выручка, ₽'])} ({top_row['Доля, %']:.1f}% от суммарной выручки).")
    else:
        insights.append(f"Лидер по выручке — {top_row['Регион']}: {format_rub(top_row['Выручка, ₽'])}.")
    if df["Наценка, %"].notna().any():
        best_markup = df.sort_values("Наценка, %", ascending=False).iloc[0]
        delta_markup = best_markup.get("Δ наценки к средн.")
        extra = "" if pd.isna(delta_markup) else f" (Δ к среднему {delta_markup:+.2f} п.п.)"
        insights.append(f"Максимальная наценка — {best_markup['Регион']}: {fmt_pct(best_markup['Наценка, %'])}{extra}.")
    if df["Риск, %"].notna().any():
        highest_risk = df.sort_values("Риск, %", ascending=False).iloc[0]
        insights.append(f"Самый высокий риск продаж ниже займа у {highest_risk['Регион']}: {fmt_pct(highest_risk['Риск, %'])}.")
    _render_insights("Главные наблюдения", insights)

    flagged = df[df["Сигнал"].astype(str).str.len() > 0]
    if not flagged.empty:
        st.markdown("**Сигналы порогов:**\n" + "\n".join(f"- {row['Регион']}: {row['Сигнал']}" for _, row in flagged.iterrows()))

    revenue_series = df.set_index("Регион")["Выручка, ₽"]
    action_lines = _generate_actions_for_series(revenue_series, Metrics.REVENUE.value)
    _render_plan("Рекомендации по выручке", action_lines[:4])

    scatter_df = df.dropna(subset=["Наценка, %", "Риск, %"]).copy()
    if not scatter_df.empty:
        scatter = px.scatter(
            scatter_df,
            x="Наценка, %",
            y="Риск, %",
            size=scatter_df["Выручка, ₽"].clip(lower=0.0),
            color="Регион",
            hover_data={
                "Регион": True,
                "Выручка, ₽": ':,.0f',
                "Наценка, %": ':.2f',
                "Риск, %": ':.2f',
                "Доля, %": ':.1f' if "Доля, %" in scatter_df else False,
            },
            labels={"Наценка, %": "Наценка, %", "Риск, %": "Риск ниже займа, %"},
            title="Региональная карта: наценка vs риск",
        )
        scatter.update_layout(height=360, margin=dict(l=40, r=40, t=60, b=40))
        st.plotly_chart(scatter, use_container_width=True, key="sales_risk_markup")


def render_scenario_simulator(ctx: PageContext) -> None:
    st.subheader("🧪 Симулятор «что если»")
    st.caption("Простая линейная модель: выставите изменения и посмотрите, как могут измениться ключевые KPI.")
    col_issue, col_markup, col_risk = st.columns(3)
    delta_issue = col_issue.slider("Выдачи займов", -40, 40, 0, step=2, format="%d%%")
    delta_markup = col_markup.slider("Процент наценки", -30, 30, 0, step=1, format="%d%%")
    delta_risk = col_risk.slider("Доля продаж ниже займа", -30, 30, 0, step=1, format="%d%%")

    def _safe(value: float | None) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    base_issue = _safe(period_value_from_itogo(ctx.df_current, ctx.regions, Metrics.LOAN_ISSUE.value, ctx.months_range))
    base_revenue = _safe(period_value_from_itogo(ctx.df_current, ctx.regions, Metrics.REVENUE.value, ctx.months_range))
    base_markup = _safe(period_value_from_itogo(ctx.df_current, ctx.regions, Metrics.MARKUP_PCT.value, ctx.months_range))
    base_risk_share = _safe(period_value_from_itogo(ctx.df_current, ctx.regions, Metrics.RISK_SHARE.value, ctx.months_range))
    base_below = _safe(period_value_from_itogo(ctx.df_current, ctx.regions, Metrics.BELOW_LOAN.value, ctx.months_range))

    factor_issue = 1 + delta_issue / 100
    factor_markup = 1 + delta_markup / 100
    factor_risk = 1 + delta_risk / 100

    new_issue = base_issue * factor_issue
    new_revenue = base_revenue * factor_issue * factor_markup
    new_markup = base_markup * factor_markup
    new_risk_share = max(0.0, base_risk_share * factor_risk)
    new_below = base_below * factor_issue * factor_risk

    m1, m2, m3 = st.columns(3)
    m1.metric("Выдано займов (руб)", format_rub(new_issue), delta=format_rub(new_issue - base_issue))
    m2.metric("Выручка (руб)", format_rub(new_revenue), delta=format_rub(new_revenue - base_revenue))
    m3.metric("Наценка / Риск", f"{fmt_pct(new_markup)} / {fmt_pct(new_risk_share)}")

    st.caption("Оценка ориентировочная. Для детального прогноза используйте вкладку «Прогноз» и индивидуальные сценарии.")

    values_base = {
        "Выдачи": base_issue,
        "Выручка": base_revenue,
        "Наценка (%)": base_markup,
        "Риск (%)": base_risk_share,
        "Убыток ниже займа": base_below,
    }
    values_new = {
        "Выдачи": new_issue,
        "Выручка": new_revenue,
        "Наценка (%)": new_markup,
        "Риск (%)": new_risk_share,
        "Убыток ниже займа": new_below,
    }

    comparison = make_subplots(specs=[[{"secondary_y": True}]])
    money_labels = ["Выдачи", "Выручка", "Убыток ниже займа"]
    money_base = [values_base[label] for label in money_labels]
    money_new = [values_new[label] for label in money_labels]
    comparison.add_trace(
        go.Bar(
            x=money_labels,
            y=money_base,
            name="База (руб)",
            marker_color="rgba(148, 163, 184, 0.6)",
        ),
        secondary_y=False,
    )
    comparison.add_trace(
        go.Bar(
            x=money_labels,
            y=money_new,
            name="Сценарий (руб)",
            marker_color="#2563eb",
        ),
        secondary_y=False,
    )

    perc_labels = ["Наценка (%)", "Риск (%)"]
    perc_base = [values_base[label] for label in perc_labels]
    perc_new = [values_new[label] for label in perc_labels]
    comparison.add_trace(
        go.Scatter(
            x=perc_labels,
            y=perc_base,
            name="Наценка/Риск (база)",
            mode="lines+markers",
            line=dict(color="#22c55e", width=3),
        ),
        secondary_y=True,
    )
    comparison.add_trace(
        go.Scatter(
            x=perc_labels,
            y=perc_new,
            name="Наценка/Риск (сценарий)",
            mode="lines+markers",
            line=dict(color="#f97316", width=3, dash="dot"),
        ),
        secondary_y=True,
    )

    comparison.update_layout(
        height=360,
        margin=dict(l=30, r=40, t=30, b=40),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        title_text="Денежные показатели и проценты в сценарии"
    )
    comparison.update_yaxes(title_text="руб", secondary_y=False)
    comparison.update_yaxes(title_text="%", secondary_y=True)
    st.plotly_chart(comparison, use_container_width=True, key="what_if_combined")

    monthly_revenue = _monthly_series_for_metric(ctx.df_current, tuple(ctx.regions), Metrics.REVENUE.value, ctx.months_range)
    monthly_issue = _monthly_series_for_metric(ctx.df_current, tuple(ctx.regions), Metrics.LOAN_ISSUE.value, ctx.months_range)
    monthly_markup = _monthly_series_for_metric(ctx.df_current, tuple(ctx.regions), Metrics.MARKUP_PCT.value, ctx.months_range)
    monthly_risk = _monthly_series_for_metric(ctx.df_current, tuple(ctx.regions), Metrics.RISK_SHARE.value, ctx.months_range)

    scenario_timeline = pd.DataFrame(index=monthly_revenue.index.astype(str))
    scenario_timeline["Выручка (база)"] = pd.to_numeric(monthly_revenue, errors="coerce")
    scenario_timeline["Выручка (сценарий)"] = scenario_timeline["Выручка (база)"] * factor_issue * factor_markup
    scenario_timeline["Риск (база)"] = pd.to_numeric(monthly_risk, errors="coerce")
    scenario_timeline["Риск (сценарий)"] = scenario_timeline["Риск (база)"] * factor_risk
    scenario_timeline = scenario_timeline.dropna(how="all")

    if not scenario_timeline.empty:
        trend_fig = make_subplots(specs=[[{"secondary_y": True}]])
        trend_fig.add_trace(
            go.Scatter(
                x=scenario_timeline.index,
                y=scenario_timeline["Выручка (база)"],
                name="Выручка (база)",
                line=dict(color="#0ea5e9", width=3),
                mode="lines+markers"
            ),
            secondary_y=False,
        )
        trend_fig.add_trace(
            go.Scatter(
                x=scenario_timeline.index,
                y=scenario_timeline["Выручка (сценарий)"],
                name="Выручка (сценарий)",
                line=dict(color="#2563eb", width=3, dash="dot"),
                mode="lines+markers"
            ),
            secondary_y=False,
        )
        if scenario_timeline["Риск (база)"].notna().any():
            trend_fig.add_trace(
                go.Scatter(
                    x=scenario_timeline.index,
                    y=scenario_timeline["Риск (база)"],
                    name="Риск (база)",
                    line=dict(color="#f97316", width=2),
                    mode="lines+markers"
                ),
                secondary_y=True,
            )
        if scenario_timeline["Риск (сценарий)"].notna().any():
            trend_fig.add_trace(
                go.Scatter(
                    x=scenario_timeline.index,
                    y=scenario_timeline["Риск (сценарий)"],
                    name="Риск (сценарий)",
                    line=dict(color="#fb923c", width=2, dash="dot"),
                    mode="lines+markers"
                ),
                secondary_y=True,
            )
        trend_fig.update_layout(
            height=360,
            margin=dict(l=30, r=20, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
        )
        trend_fig.update_yaxes(title_text="Выручка (руб)", secondary_y=False)
        trend_fig.update_yaxes(title_text="Доля ниже займа, %", secondary_y=True)
        st.plotly_chart(trend_fig, use_container_width=True, key="what_if_trend")

    delta_text = []
    if base_issue:
        delta_text.append(f"Выдачи {delta_issue:+d}% → {format_rub(new_issue)}")
    if base_revenue:
        delta_text.append(f"Выручка {format_rub(new_revenue - base_revenue)}")
    delta_text.append(f"Наценка {fmt_pct(new_markup)} | Риск {fmt_pct(new_risk_share)}")
    if base_below:
        delta_text.append(f"Убыток от продаж ниже займа {format_rub(new_below - base_below)}")
    st.markdown("**Что меняется:** " + "; ".join(delta_text))


def render_margin_capacity_planner(ctx: PageContext, widget_prefix: str = "margin_planner") -> None:
    st.subheader("🎯 Планировщик маржи и скидок")
    base_markup = period_value_from_itogo(ctx.df_current, ctx.regions, Metrics.MARKUP_PCT.value, ctx.months_range)
    base_revenue = period_value_from_itogo(ctx.df_current, ctx.regions, Metrics.REVENUE.value, ctx.months_range)
    base_loss = period_value_from_itogo(ctx.df_current, ctx.regions, Metrics.BELOW_LOAN.value, ctx.months_range)
    base_loss_units = period_value_from_itogo(ctx.df_current, ctx.regions, Metrics.BELOW_LOAN_UNITS.value, ctx.months_range)
    base_markup_amount = period_value_from_itogo(ctx.df_current, ctx.regions, Metrics.MARKUP_AMOUNT.value, ctx.months_range)

    default_markup = 45.0
    if base_markup is not None and not pd.isna(base_markup):
        default_markup = float(np.clip(base_markup, 0.0, 150.0))

    if base_loss is not None and not pd.isna(base_loss) and base_loss > 0:
        default_loss_budget = float(max(0.5, round((base_loss / 1_000_000) * 1.1, 1)))
    else:
        default_loss_budget = 5.0

    col_target, col_budget = st.columns(2)
    target_markup = col_target.slider(
        "Целевой процент наценки",
        min_value=0.0,
        max_value=150.0,
        value=default_markup,
        step=0.5,
        help="Какой средний процент наценки вы хотите удержать по выбранному периоду.",
        key=f"{widget_prefix}_target_markup",
    )
    loss_budget_mln = col_budget.number_input(
        "Допустимый бюджет убыточных продаж, млн ₽",
        min_value=0.0,
        max_value=500.0,
        value=default_loss_budget,
        step=0.5,
        help="Лимит потерь на распродажах ниже займа, включая текущий результат.",
        key=f"{widget_prefix}_loss_budget",
    )
    loss_budget = loss_budget_mln * 1_000_000

    markup_gap_pct = None
    if base_markup is not None and not pd.isna(base_markup):
        markup_gap_pct = target_markup - base_markup

    target_markup_amount = None
    markup_amount_gap = None
    if base_markup_amount is not None and base_markup is not None and base_markup not in (0, 0.0) and not pd.isna(base_markup_amount):
        target_markup_amount = base_markup_amount * (target_markup / base_markup if base_markup else 1.0)
    elif base_revenue is not None and not pd.isna(base_revenue):
        target_markup_amount = base_revenue * (target_markup / 100)
    if target_markup_amount is not None and base_markup_amount is not None and not pd.isna(base_markup_amount):
        markup_amount_gap = target_markup_amount - base_markup_amount

    loss_gap = None
    if base_loss is not None and not pd.isna(base_loss):
        loss_gap = loss_budget - base_loss

    avg_loss_per_unit = None
    if base_loss_units and base_loss_units > 0 and base_loss is not None and not pd.isna(base_loss) and base_loss > 0:
        avg_loss_per_unit = base_loss / base_loss_units

    allowed_extra_units = None
    if loss_gap is not None and loss_gap > 0 and avg_loss_per_unit:
        allowed_extra_units = loss_gap / avg_loss_per_unit

    col_status_1, col_status_2, col_status_3 = st.columns(3)
    delta_markup_label = "—"
    if markup_gap_pct is not None:
        delta_markup_label = f"{markup_gap_pct:+.1f} п.п."
    col_status_1.metric("Целевая наценка", fmt_pct(target_markup), delta=delta_markup_label)

    col_status_2.metric(
        "Маржа за период (новая)",
        format_rub(target_markup_amount),
        delta="—" if markup_amount_gap is None else format_rub(markup_amount_gap)
    )

    loss_delta_label = "—"
    if loss_gap is not None and base_loss is not None and not pd.isna(base_loss):
        loss_delta_label = f"{(loss_budget - base_loss) / 1_000_000:+.2f} млн ₽"
    col_status_3.metric(
        "Лимит убыточных продаж",
        f"{loss_budget_mln:.2f} млн ₽",
        delta=loss_delta_label
    )

    insights: list[str] = []
    if markup_gap_pct is not None:
        if markup_gap_pct > 0:
            if markup_amount_gap is not None:
                insights.append(f"Нужно добрать {format_rub(markup_amount_gap)} маржи, чтобы выйти на {fmt_pct(target_markup)}.")
            else:
                insights.append(f"Поднять наценку до {fmt_pct(target_markup)} (+{markup_gap_pct:.1f} п.п.).")
        elif markup_gap_pct < 0:
            insights.append(f"Есть запас в {abs(markup_gap_pct):.1f} п.п. по наценке — можно упростить условия распродажи.")
        else:
            insights.append("Текущая наценка уже соответствует цели.")
    if loss_gap is not None:
        if loss_gap > 0:
            msg = f"Можно ещё допустить {format_rub(loss_gap)} продаж ниже займа"
            if allowed_extra_units is not None and allowed_extra_units > 0:
                msg += f" (~{int(max(1, np.floor(allowed_extra_units)))} поз.)."
            else:
                msg += "."
            insights.append(msg)
        elif loss_gap < 0:
            insights.append(f"Бюджет убытков превышен на {format_rub(abs(loss_gap))} — увеличьте наценку или сократите скидки.")
        else:
            insights.append("Лимит убыточных продаж выбран вровень с текущим уровнем.")
    if avg_loss_per_unit:
        insights.append(f"Средний убыток на позицию сейчас {format_rub(avg_loss_per_unit)}.")

    if insights:
        bullets = "\n".join(f"- {line}" for line in insights)
        st.markdown(f"**Выводы:**\n{bullets}")
    else:
        st.caption("Недостаточно данных для расчёта рекомендаций — загрузите показатели наценки и убытков.")


def risk_alerts_block(ctx: PageContext) -> dict[str, float | None]:
    st.subheader("🔔 Сигналы риска")
    thresholds = ctx.thresholds or {}
    default_risk = float(thresholds.get("max_risk", 25.0))
    default_markup = float(thresholds.get("min_markup", 45.0))
    default_loss = float(thresholds.get("loss_cap", 5.0))
    col_risk, col_markup, col_loss = st.columns(3)
    risk_threshold = col_risk.number_input(
        "Порог доли ниже займа, %",
        min_value=0.0,
        max_value=100.0,
        value=default_risk,
        step=1.0,
        help="Сигнал, если доля продаж ниже займа превышает этот уровень."
    )
    markup_floor = col_markup.number_input(
        "Минимальная наценка, %",
        min_value=0.0,
        max_value=200.0,
        value=default_markup,
        step=1.0,
        help="Сигнал, если средняя наценка опускается ниже заданного порога."
    )
    loss_cap_mln = col_loss.number_input(
        "Лимит убытка ниже займа, млн ₽",
        min_value=0.0,
        max_value=500.0,
        value=default_loss,
        step=0.5,
        help="Сигнал, если суммарный убыток от продаж ниже займа превышает бюджет."
    )
    loss_cap = loss_cap_mln * 1_000_000

    current_risk = period_value_from_itogo(ctx.df_current, ctx.regions, Metrics.RISK_SHARE.value, ctx.months_range)
    current_markup = period_value_from_itogo(ctx.df_current, ctx.regions, Metrics.MARKUP_PCT.value, ctx.months_range)
    current_loss = period_value_from_itogo(ctx.df_current, ctx.regions, Metrics.BELOW_LOAN.value, ctx.months_range)

    def _delta_text(delta: float | None, unit: str) -> str | None:
        if delta is None or pd.isna(delta):
            return None
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.1f} {unit}"

    with st.container():
        col_r, col_m, col_l = st.columns(3)
        risk_delta = None
        if current_risk is not None and pd.notna(current_risk):
            risk_delta = risk_threshold - current_risk
        markup_delta = None
        if current_markup is not None and pd.notna(current_markup):
            markup_delta = current_markup - markup_floor
        loss_delta = None
        if current_loss is not None and pd.notna(current_loss):
            loss_delta = loss_cap - current_loss

        col_r.metric(
            "Доля продаж ниже займа",
            fmt_pct(current_risk),
            delta=_delta_text(risk_delta, "п.п.") or "—"
        )
        col_r.caption(f"Порог: {fmt_pct(risk_threshold)}")
        if risk_delta is not None and risk_delta < 0:
            col_r.error("Порог превышен — требуется реакция.")

        col_m.metric(
            "Средняя наценка",
            fmt_pct(current_markup),
            delta=_delta_text(markup_delta, "п.п.") or "—"
        )
        col_m.caption(f"Минимум: {fmt_pct(markup_floor)}")
        if markup_delta is not None and markup_delta < 0:
            col_m.warning("Наценка ниже целевого уровня.")

        loss_value_display = None if current_loss is None else current_loss / 1_000_000
        loss_delta_display = None if loss_delta is None else loss_delta / 1_000_000
        loss_delta_text = _delta_text(loss_delta_display, "млн") or "—"
        col_l.metric(
            "Убыток ниже займа (млн ₽)",
            "—" if loss_value_display is None else f"{loss_value_display:.2f}",
            delta=loss_delta_text
        )
        col_l.caption(f"Лимит: {loss_cap_mln:.1f} млн ₽")
        if loss_delta is not None and loss_delta < 0:
            col_l.error("Бюджет убытков превышен.")

    bullet_points: list[str] = []
    if current_risk is not None and not pd.isna(current_risk):
        flag = "✅" if risk_delta is None or risk_delta >= 0 else "⚠️"
        bullet_points.append(f"{flag} Риск: {fmt_pct(current_risk)} при пороге {fmt_pct(risk_threshold)}.")
    if current_markup is not None and not pd.isna(current_markup):
        flag = "✅" if markup_delta is None or markup_delta >= 0 else "⚠️"
        bullet_points.append(f"{flag} Наценка: {fmt_pct(current_markup)} vs минимум {fmt_pct(markup_floor)}.")
    if current_loss is not None and not pd.isna(current_loss):
        flag = "✅" if loss_delta is None or loss_delta >= 0 else "⚠️"
        bullet_points.append(
            f"{flag} Убыток: {format_rub(current_loss)} при лимите {format_rub(loss_cap)}."
        )
    if bullet_points:
        st.markdown("**Статус:**<br>" + "<br>".join(bullet_points), unsafe_allow_html=True)

    st.session_state["thresholds_config"] = {
        "min_markup": float(markup_floor),
        "max_risk": float(risk_threshold),
        "loss_cap": float(loss_cap_mln),
    }

    return {
        "risk_threshold": risk_threshold,
        "markup_floor": markup_floor,
        "loss_cap": loss_cap,
    }


def risk_markup_heatmap_block(ctx: PageContext) -> None:
    st.subheader("🔥 Теплокарта «риск ↔ наценка»")
    risk_df = get_monthly_totals_from_file(ctx.df_current, tuple(ctx.regions), Metrics.RISK_SHARE.value)
    markup_df = get_monthly_totals_from_file(ctx.df_current, tuple(ctx.regions), Metrics.MARKUP_PCT.value)
    revenue_df = get_monthly_totals_from_file(ctx.df_current, tuple(ctx.regions), Metrics.REVENUE.value)
    if risk_df.empty or markup_df.empty:
        st.info("Недостаточно данных для построения теплокарты.")
        return

    merged = (
        risk_df.rename(columns={"Значение": "Risk"})
        .merge(markup_df.rename(columns={"Значение": "Markup"}), on=["Регион", "Месяц"], how="inner")
    )
    if revenue_df is not None and not revenue_df.empty:
        merged = merged.merge(
            revenue_df.rename(columns={"Значение": "Revenue"}),
            on=["Регион", "Месяц"],
            how="left"
        )
    merged = merged[merged["Месяц"].astype(str).isin(ctx.months_range)]
    merged["Risk"] = pd.to_numeric(merged["Risk"], errors="coerce")
    merged["Markup"] = pd.to_numeric(merged["Markup"], errors="coerce")
    merged["Revenue"] = pd.to_numeric(merged.get("Revenue"), errors="coerce")
    merged = merged.dropna(subset=["Risk", "Markup"])
    if merged.empty:
        st.info("Нет точек с одновременными значениями риска и наценки.")
        return

    risk_min, risk_max = merged["Risk"].min(), merged["Risk"].max()
    markup_min, markup_max = merged["Markup"].min(), merged["Markup"].max()
    if np.isclose(risk_min, risk_max) or np.isclose(markup_min, markup_max):
        st.info("Разброс значений недостаточен для теплокарты — показываю точечный график.")
        scatter = px.scatter(
            merged,
            x="Markup",
            y="Risk",
            size=merged["Revenue"].clip(lower=0.0) if "Revenue" in merged else None,
            color="Регион",
            hover_data=["Месяц", "Регион", "Revenue"],
            labels={"Markup": "Наценка, %", "Risk": "Риск, %"}
        )
        scatter.update_layout(height=380, margin=dict(l=40, r=40, t=30, b=40))
        st.plotly_chart(scatter, use_container_width=True, key="risk_markup_scatter_fallback")
        return

    risk_bins = np.linspace(risk_min, risk_max, num=7)
    markup_bins = np.linspace(markup_min, markup_max, num=7)
    merged["risk_bucket"] = pd.cut(merged["Risk"], bins=risk_bins, include_lowest=True)
    merged["markup_bucket"] = pd.cut(merged["Markup"], bins=markup_bins, include_lowest=True)

    weight = merged["Revenue"].fillna(0.0)
    if weight.abs().sum() == 0:
        weight = pd.Series(1.0, index=merged.index)

    heat = (
        merged.assign(weight=weight)
        .groupby(["risk_bucket", "markup_bucket"], observed=True)["weight"]
        .sum()
        .unstack(fill_value=0.0)
    )
    if heat.empty:
        st.info("Не удалось построить теплокарту.")
        return

    heat = heat.loc[:, heat.sum(axis=0) > 0.0]
    heat = heat.loc[heat.sum(axis=1) > 0.0]
    if heat.empty:
        st.info("Не удалось построить теплокарту.")
        return

    x_display = [f"{interval.left:.1f}–{interval.right:.1f}%" for interval in heat.columns]
    y_display = [f"{interval.left:.1f}–{interval.right:.1f}%" for interval in heat.index]

    fig = go.Figure(
        go.Heatmap(
            z=heat.values,
            x=x_display,
            y=y_display,
            colorscale=[
                [0.0, "#f8fafc"],
                [0.2, "#dbeafe"],
                [0.5, "#93c5fd"],
                [0.8, "#3b82f6"],
                [1.0, "#1d4ed8"],
            ],
            hovertemplate="Наценка: %{x}<br>Риск: %{y}<br>Вес: %{z:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=60, r=40, t=40, b=60),
        xaxis_title="Диапазон наценки, %",
        yaxis_title="Диапазон риска, %",
    )
    st.plotly_chart(fig, use_container_width=True, key="risk_markup_heatmap")
    st.caption("Яркость клетки отражает совокупную выручку (или количество точек), попавших в пару диапазонов.")


def risk_failure_forecast_block(ctx: PageContext, risk_threshold: float | None) -> None:
    st.subheader("📉 Прогноз провалов по риску")
    st.caption("Прогноз доли продаж ниже суммы займа на ближайшие месяцы. Используем линейный тренд по факту: линия — ожидаемый процент, область — 95% интервал. Если установлен порог, показываем, где прогноз его превышает.")
    forecast_bundle = _prepare_forecast(
        ctx.df_current,
        ctx.regions,
        ctx.months_range,
        Metrics.RISK_SHARE.value,
        horizon=4,
    )
    if not forecast_bundle:
        st.info("Недостаточно данных для расчёта прогноза.")
        return

    history = forecast_bundle["history"]
    future_labels = forecast_bundle["future_labels"]
    forecast_vals = forecast_bundle["forecast"]
    lower_vals = forecast_bundle["lower"]
    upper_vals = forecast_bundle["upper"]

    fig = go.Figure()
    hist_x = [str(x) for x in history.index]
    fig.add_trace(go.Scatter(
        x=hist_x,
        y=history.values,
        mode="lines+markers",
        name="Факт",
        line=dict(color="#2563eb", width=3),
    ))

    forecast_x = [hist_x[-1]] + future_labels
    forecast_y = [history.values[-1]] + forecast_vals
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=forecast_y,
        mode="lines+markers",
        name="Прогноз",
        line=dict(color="#7c3aed", width=2, dash="dot"),
    ))

    fig.add_trace(go.Scatter(
        x=future_labels + future_labels[::-1],
        y=upper_vals + lower_vals[::-1],
        fill="toself",
        fillcolor="rgba(124,58,237,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="95% интервал",
    ))

    if risk_threshold is not None:
        fig.add_hline(
            y=risk_threshold,
            line=dict(color="#ef4444", width=2, dash="dash"),
            annotation_text=f"Порог {risk_threshold:.1f}%",
            annotation_position="top left",
        )

    fig.update_layout(
        height=380,
        margin=dict(l=40, r=40, t=50, b=30),
        yaxis_title="Доля продаж ниже займа, %",
        xaxis_title="Месяц",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key="risk_failure_forecast")

    forecast_table = pd.DataFrame({
        "Месяц": future_labels,
        "Прогноз, %": [float(v) for v in forecast_vals],
        "Низ, %": [float(v) for v in lower_vals],
        "Верх, %": [float(v) for v in upper_vals],
    })
    if risk_threshold is not None:
        forecast_table["Сигнал"] = [
            "⚠️ Превышение" if (risk_threshold is not None and val > risk_threshold) else "✅ В норме"
            for val in forecast_table["Прогноз, %"]
        ]
    st.dataframe(
        forecast_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Прогноз, %": st.column_config.NumberColumn("Прогноз, %", format="%.2f"),
            "Низ, %": st.column_config.NumberColumn("Нижняя граница, %", format="%.2f"),
            "Верх, %": st.column_config.NumberColumn("Верхняя граница, %", format="%.2f"),
        },
    )
    st.caption("Прогноз построен на линейном тренде с сезонностью. Сигнал появляется, если прогноз превышает заданный порог.")


def render_risk_dependency(ctx: PageContext) -> None:
    st.subheader("⚖️ Риск и маржа")
    risk_series = _monthly_series_for_metric(ctx.df_current, tuple(ctx.regions), Metrics.RISK_SHARE.value, ctx.months_range)
    markup_series = _monthly_series_for_metric(ctx.df_current, tuple(ctx.regions), Metrics.MARKUP_PCT.value, ctx.months_range)
    below_series = _monthly_series_for_metric(ctx.df_current, tuple(ctx.regions), Metrics.BELOW_LOAN.value, ctx.months_range)
    revenue_series = _monthly_series_for_metric(ctx.df_current, tuple(ctx.regions), Metrics.REVENUE.value, ctx.months_range)
    data = pd.DataFrame({
        "Risk": pd.to_numeric(risk_series, errors="coerce"),
        "Markup": pd.to_numeric(markup_series, errors="coerce"),
        "Below": pd.to_numeric(below_series, errors="coerce"),
        "Revenue": pd.to_numeric(revenue_series, errors="coerce")
    })
    data = data.dropna(how="all")
    if data.empty or data["Risk"].dropna().empty:
        st.info("Недостаточно данных для анализа взаимосвязи.")
        return

    line_df = data.dropna(subset=["Risk", "Markup"])
    if not line_df.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=[str(x) for x in line_df.index],
                y=line_df["Risk"],
                mode="lines+markers",
                name="Доля ниже займа, %",
                line=dict(color="#f97316", width=3),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=[str(x) for x in line_df.index],
                y=line_df["Markup"],
                mode="lines+markers",
                name="Процент наценки",
                line=dict(color="#22c55e", width=2, dash="dot"),
            ),
            secondary_y=True,
        )
        if not data["Below"].dropna().empty:
            fig.add_trace(
                go.Bar(
                    x=[str(x) for x in data.index],
                    y=data["Below"].fillna(0.0),
                    name="Убыток ниже займа (руб)",
                    marker_color="rgba(239,68,68,0.35)",
                    opacity=0.6,
                ),
                secondary_y=False,
            )
        fig.update_yaxes(title_text="Доля ниже займа, %", secondary_y=False)
        fig.update_yaxes(title_text="Наценка, %", secondary_y=True)
        fig.update_layout(
            height=360,
            margin=dict(l=40, r=40, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig, use_container_width=True, key="risk_dual_axis")

    scatter_df = data.dropna(subset=["Risk", "Markup"])
    if not scatter_df.empty:
        revenue_max = float(scatter_df["Revenue"].fillna(0).abs().max() or 1.0)
        sizes = scatter_df["Revenue"].fillna(0).apply(lambda v: 12 + 30 * (abs(v) / revenue_max))
        colors = scatter_df["Below"].fillna(0.0)
        scatter = go.Figure(go.Scatter(
            x=scatter_df["Risk"],
            y=scatter_df["Markup"],
            mode="markers+text",
            text=[str(x) for x in scatter_df.index],
            textposition="top center",
            marker=dict(
                size=sizes,
                color=colors,
                colorscale="Reds",
                showscale=True,
                colorbar=dict(title="Убыток ниже займа, руб"),
            ),
            name="Месяцы",
        ))
        scatter.update_layout(
            height=360,
            margin=dict(l=40, r=40, t=20, b=40),
            xaxis_title="Доля продаж ниже займа, %",
            yaxis_title="Процент наценки, %",
        )
        st.plotly_chart(scatter, use_container_width=True, key="risk_scatter")
        corr = float(scatter_df["Risk"].corr(scatter_df["Markup"])) if scatter_df.shape[0] > 1 else float("nan")
        if not np.isnan(corr):
            st.caption(f"Корреляция между долей ниже займа и наценкой: {corr:+.2f}.")
    st.caption("Размер точки отражает выручку месяца, цвет — уровень убытка от продаж ниже займа.")


@st.cache_data(show_spinner=False, max_entries=256)
def _extract_region_month_metric(df_source: pd.DataFrame, regions: list[str], metric: str, months: list[str]) -> pd.DataFrame:
    frame = get_monthly_totals_from_file(df_source, tuple(regions), metric)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["Месяц"] = frame["Месяц"].astype(str)
    frame = frame[frame["Месяц"].isin(months)]
    if frame.empty:
        return frame
    frame["Значение"] = pd.to_numeric(frame["Значение"], errors="coerce")
    frame = frame.dropna(subset=["Значение"])
    if not frame.empty and is_percent_metric(metric):
        sample = frame["Значение"].abs()
        if sample.median(skipna=True) <= 1.5:
            frame["Значение"] = frame["Значение"] * 100.0
    frame["Регион"] = frame["Регион"].astype(str)
    frame["Месяц"] = pd.Categorical(frame["Месяц"], categories=ORDER, ordered=True)
    frame = frame.sort_values("Месяц")
    frame["Месяц"] = frame["Месяц"].astype(str)
    return frame


def render_region_band_chart(ctx: PageContext) -> None:
    st.subheader("🎀 Ленточный график выручки по регионам")
    st.caption("Лента отображает диапазон 10–90 перцентилей по выручке, тёмная линия — медиана. Так видно типичный коридор и выбросы по регионам.")
    df_metric = _extract_region_month_metric(ctx.df_current, ctx.regions, Metrics.REVENUE.value, ctx.months_range)
    if df_metric.empty:
        st.info("Нет данных по выручке для выбранных регионов.")
        return
    stats: Dict[str, Dict[str, float]] = {}
    for month, group in df_metric.groupby("Месяц"):
        values = group["Значение"].astype(float)
        if values.empty:
            continue
        stats[str(month)] = {
            "p10": float(np.percentile(values, 10)),
            "median": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
        }
    months = [m for m in ctx.months_range if m in stats]
    if not months:
        st.info("Недостаточно данных для выбранного периода.")
        return
    p10 = [stats[m]["p10"] for m in months]
    p90 = [stats[m]["p90"] for m in months]
    median = [stats[m]["median"] for m in months]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months,
        y=p90,
        mode="lines",
        name="90-й перцентиль",
        line=dict(color="#38bdf8", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=months,
        y=p10,
        mode="lines",
        name="10-й перцентиль",
        line=dict(color="#38bdf8", width=1.5),
        fill="tonexty",
        fillcolor="rgba(59,130,246,0.18)",
    ))
    fig.add_trace(go.Scatter(
        x=months,
        y=median,
        mode="lines+markers",
        name="Медиана",
        line=dict(color="#0f172a", width=3),
        marker=dict(size=6, color="#0ea5e9")
    ))
    fig.update_layout(
        height=360,
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified",
        yaxis_title="Выручка, ₽",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key="region_band_chart")
    st.caption("Тёмная линия — медианная выручка; полупрозрачная лента показывает диапазон 10–90 перцентилей по регионам.")


def render_markup_candlestick(ctx: PageContext) -> None:
    st.subheader("🕯️ Свечной профиль наценки")
    df_metric = _extract_region_month_metric(ctx.df_current, ctx.regions, Metrics.MARKUP_PCT.value, ctx.months_range)
    if df_metric.empty:
        st.info("Нет данных по наценке для построения свечного графика.")
        return
    stats: Dict[str, Dict[str, float]] = {}
    for month, group in df_metric.groupby("Месяц"):
        values = group["Значение"].astype(float)
        if values.empty:
            continue
        stats[str(month)] = {
            "low": float(values.min()),
            "q1": float(np.percentile(values, 25)),
            "q3": float(np.percentile(values, 75)),
            "high": float(values.max()),
        }
    months = [m for m in ctx.months_range if m in stats]
    if not months:
        st.info("Недостаточно данных по наценке для выбранного периода.")
        return
    fig = go.Figure(go.Candlestick(
        x=months,
        open=[stats[m]["q1"] for m in months],
        close=[stats[m]["q3"] for m in months],
        low=[stats[m]["low"] for m in months],
        high=[stats[m]["high"] for m in months],
        increasing_line_color="#22c55e",
        increasing_fillcolor="rgba(34,197,94,0.4)",
        decreasing_line_color="#ef4444",
        decreasing_fillcolor="rgba(239,68,68,0.35)",
        name="Диапазон наценки"
    ))
    fig.update_layout(
        height=360,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False,
        yaxis_title="Наценка, %",
        xaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True, key="markup_candlestick")
    st.caption("Свечи показывают разброс наценки по регионам: основания — квартиль 25/75, тени — минимум и максимум.")


def render_region_map_block(ctx: PageContext) -> None:
    st.subheader("🗺️ Интенсивность по регионам и подразделениям")
    map_metrics = [
        (Metrics.REVENUE.value, "Выручка, ₽"),
        (Metrics.RISK_SHARE.value, "Доля продаж ниже займа, %"),
        (Metrics.MARKUP_PCT.value, "Процент наценки, %"),
    ]
    view_mode = st.radio(
        "Формат отображения",
        options=["Лента лидеров", "Карта"],
        horizontal=True,
        key="region_map_mode"
    )
    metric_choice = st.selectbox(
        "Метрика",
        options=map_metrics,
        format_func=lambda item: item[1],
        key="region_map_metric"
    )
    metric_key = metric_choice[0]
    st.caption(METRIC_HELP.get(metric_key, ""))

    sub = strip_totals_rows(ctx.df_current)
    sub = sub[
        (sub["Регион"].isin(ctx.regions)) &
        (sub["Месяц"].astype(str).isin(ctx.months_range)) &
        (sub["Показатель"] == metric_key)
    ].copy()
    if sub.empty:
        st.info("Нет данных для выбранной метрики.")
        return

    agg = (
        sub.groupby(["Регион", "Подразделение"], observed=True)["Значение"]
        .sum()
        .reset_index()
    )
    agg["Значение"] = pd.to_numeric(agg["Значение"], errors="coerce")
    agg = agg.dropna(subset=["Значение"])
    if agg.empty:
        st.info("Недостаточно данных после агрегации по подразделениям.")
        return

    percent_metric = is_percent_metric(metric_key)
    if percent_metric and agg["Значение"].abs().median() <= 1.5:
        agg["Значение"] *= 100.0

    rows: List[Dict[str, Any]] = []
    missing_regions: List[str] = []
    for _, row in agg.iterrows():
        coords = resolve_region_coordinates(str(row["Регион"]))
        if not coords:
            missing_regions.append(str(row["Регион"]))
            continue
        rows.append({
            "Регион": row["Регион"],
            "Подразделение": row["Подразделение"],
            "lat": coords[0],
            "lon": coords[1],
            "Значение": float(row["Значение"]),
        })
    if not rows:
        st.info("Не удалось сопоставить регионы с координатами. Добавьте их в словарь REGION_COORDS.")
        return
    df_map = pd.DataFrame(rows)
    if missing_regions:
        st.caption("Не найдены координаты для: " + ", ".join(sorted(set(missing_regions))))

    df_ranked = df_map.sort_values("Значение", ascending=percent_metric and metric_key in METRICS_SMALLER_IS_BETTER)
    if view_mode == "Лента лидеров" or df_map["lat"].isna().all():
        df_ranked = df_ranked.copy()
        df_ranked["Ключ"] = df_ranked["Регион"] + " · " + df_ranked["Подразделение"].fillna("—")
        df_ranked["Значение, отображение"] = df_ranked["Значение"].apply(
            lambda v: fmt_pct(v) if percent_metric else format_rub(v)
        )
        bar_fig = px.bar(
            df_ranked,
            x="Значение",
            y="Ключ",
            color="Значение",
            orientation="h",
            color_continuous_scale="Blues" if not percent_metric else "Oranges",
        )
        bar_fig.update_layout(
            height=420,
            margin=dict(l=40, r=20, t=40, b=40),
            coloraxis_showscale=False,
        )
        bar_fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Значение: %{x:.2f}" + ("%" if percent_metric else " ₽") + "<extra></extra>"
        )
        st.plotly_chart(bar_fig, use_container_width=True, key="region_leaderboard")
        st.dataframe(
            df_ranked[["Регион", "Подразделение", "Значение, отображение"]],
            use_container_width=True,
            hide_index=True,
            column_config={"Значение, отображение": "Значение"},
        )
        st.caption("Столбики отсортированы по значению: сверху топ подразделения, в таблице можно быстро найти нужный регион." )
    else:
        color_values = df_map["Значение"]
        size_raw = color_values.fillna(0.0).abs()
        if percent_metric:
            normalized = size_raw / (size_raw.max() or 1.0)
            size_values = 6 + normalized * 12
        else:
            size_values = 6 + np.log1p(size_raw)
        fig = go.Figure(go.Scattergeo(
            lon=df_map["lon"],
            lat=df_map["lat"],
            text=df_map["Регион"],
            customdata=np.column_stack([df_map["Подразделение"], df_map["Значение"]]),
            marker=dict(
                size=size_values,
                color=color_values,
                colorscale="Blues" if not percent_metric else "Sunset",
                colorbar=dict(
                    title=metric_choice[1],
                    tickformat=".1f" if percent_metric else ".0f",
                ),
                line=dict(width=0.6, color="rgba(15,23,42,0.45)"),
                sizemode="diameter",
                opacity=0.85,
            ),
            hovertemplate="<b>%{text}</b><br>Подразделение: %{customdata[0]}<br>Значение: %{customdata[1]:.2f}" + ("%" if percent_metric else " ₽") + "<extra></extra>",
            name="",
        ))
        fig.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=0, b=0),
            geo=dict(
                projection_type="natural earth",
                showcountries=True,
                showcoastlines=True,
                lonaxis=dict(range=[20, 170]),
                lataxis=dict(range=[40, 75]),
                bgcolor="rgba(255,255,255,0)",
            ),
        )
        st.plotly_chart(fig, use_container_width=True, key="region_map")
        st.caption("Размер круга отражает масштаб показателя, цвет — его уровень. Координаты можно расширить в REGION_COORDS или через авто-геокодер (сохраняется в сессии).")


def render_region_map_block(ctx: PageContext) -> None:  # override with matrix-based view
    st.subheader("📍 Интенсивность по регионам и подразделениям")
    map_metrics = [
        (Metrics.REVENUE.value, "Выручка, ₽"),
        (Metrics.RISK_SHARE.value, "Доля продаж ниже займа, %"),
        (Metrics.MARKUP_PCT.value, "Процент наценки, %"),
    ]
    view_mode = st.radio(
        "Формат отображения",
        options=["Лента лидеров", "Теплокарта"],
        horizontal=True,
        key="region_map_mode"
    )
    metric_choice = st.selectbox(
        "Метрика",
        options=map_metrics,
        format_func=lambda item: item[1],
        key="region_map_metric"
    )
    metric_key = metric_choice[0]
    st.caption(METRIC_HELP.get(metric_key, ""))

    sub = strip_totals_rows(ctx.df_current)
    sub = sub[
        (sub["Регион"].isin(ctx.regions)) &
        (sub["Месяц"].astype(str).isin(ctx.months_range)) &
        (sub["Показатель"] == metric_key)
    ].copy()
    if sub.empty:
        st.info("Нет данных для выбранной метрики.")
        return

    agg = (
        sub.groupby(["Регион", "Подразделение"], observed=True)["Значение"]
        .sum()
        .reset_index()
    )
    agg["Значение"] = pd.to_numeric(agg["Значение"], errors="coerce")
    agg = agg.dropna(subset=["Значение"])
    if agg.empty:
        st.info("Недостаточно данных после агрегации по подразделениям.")
        return

    percent_metric = is_percent_metric(metric_key)
    if percent_metric and agg["Значение"].abs().median() <= 1.5:
        agg["Значение"] *= 100.0

    df_ranked = agg.sort_values("Значение", ascending=percent_metric and metric_key in METRICS_SMALLER_IS_BETTER)
    if view_mode == "Лента лидеров":
        df_ranked = df_ranked.copy()
        df_ranked["Ключ"] = df_ranked["Регион"] + " · " + df_ranked["Подразделение"].fillna("—")
        df_ranked["Значение, отображение"] = df_ranked["Значение"].apply(
            lambda v: fmt_pct(v) if percent_metric else format_rub(v)
        )
        bar_fig = px.bar(
            df_ranked,
            x="Значение",
            y="Ключ",
            color="Значение",
            orientation="h",
            color_continuous_scale="Blues" if not percent_metric else "Oranges",
        )
        bar_fig.update_layout(
            height=420,
            margin=dict(l=40, r=20, t=40, b=40),
            coloraxis_showscale=False,
        )
        bar_fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Значение: %{x:.2f}" + ("%" if percent_metric else " ₽") + "<extra></extra>"
        )
        st.plotly_chart(bar_fig, use_container_width=True, key="region_leaderboard")
        st.dataframe(
            df_ranked[["Регион", "Подразделение", "Значение, отображение"]],
            use_container_width=True,
            hide_index=True,
            column_config={"Значение, отображение": "Значение"},
        )
        st.caption("Столбики отсортированы по значению: сверху топ подразделения, в таблице можно быстро найти нужный регион.")
        return

    pivot = agg.pivot_table(index="Регион", columns="Подразделение", values="Значение", aggfunc="sum", observed=True)
    if pivot.empty:
        st.info("Недостаточно данных для теплокарты.")
        return
    totals = pivot.abs().sum(axis=0).sort_values(ascending=percent_metric and metric_key in METRICS_SMALLER_IS_BETTER)
    top_columns = totals.index[: min(15, len(totals))]
    pivot = pivot[top_columns].fillna(0.0)

    heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="Blues" if not percent_metric else "Sunset",
        colorbar=dict(title=metric_choice[1], tickformat=".1f" if percent_metric else ".0f"),
        hovertemplate="Регион: %{y}<br>Подразделение: %{x}<br>Значение: %{z:.2f}" + ("%" if percent_metric else " ₽") + "<extra></extra>",
    ))
    heat.update_layout(height=420, margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(heat, use_container_width=True, key="region_heatmap")
    st.caption("Теплокарта показывает значения по региону/подразделению. Отображаются подразделения с наибольшим вкладом по выбранной метрике.")


def render_comparison_page(ctx: PageContext) -> None:
    st.markdown("### 🧭 Сравнение регионов и филиалов")
    st.caption("Быстрое сравнение распределения ключевых метрик по регионам в выбранном окне месяцев.")
    render_region_band_chart(ctx)
    st.divider()
    render_markup_candlestick(ctx)
    st.divider()
    render_region_map_block(ctx)
    st.divider()
    revenue_values = period_values_by_region_from_itogo(ctx.df_current, ctx.regions, Metrics.REVENUE.value, ctx.months_range)
    markup_values = period_values_by_region_from_itogo(ctx.df_current, ctx.regions, Metrics.MARKUP_PCT.value, ctx.months_range)
    if revenue_values:
        comparison_df = pd.DataFrame(
            {
                "Регион": list(revenue_values.keys()),
                "Выручка, ₽": [revenue_values.get(reg) for reg in revenue_values.keys()],
                "Наценка, %": [markup_values.get(reg) for reg in revenue_values.keys()],
            }
        )
        comparison_df = comparison_df.sort_values(by="Выручка, ₽", ascending=False)
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Выручка, ₽": st.column_config.NumberColumn("Выручка, ₽", format="%.0f"),
                "Наценка, %": st.column_config.NumberColumn("Наценка, %", format="%.2f"),
            },
        )


def _month_to_quarter(month: str) -> str:
    if month not in ORDER:
        return month
    idx = ORDER.index(month)
    quarter = (idx // 3) + 1
    return f"Q{quarter}"


def render_cohort_page(ctx: PageContext) -> None:
    st.markdown("### 👥 Клиентские когорты")
    st.caption("Анализ потока новых клиентов и сохранения базы по месяцам (приближенно, на основе агрегированных метрик).")
    available_metrics = sorted(
        m for m in ctx.df_current["Показатель"].dropna().astype(str).unique()
        if m not in HIDDEN_METRICS
    )
    default_new_metric = Metrics.NEW_UNIQUE_CLIENTS.value if Metrics.NEW_UNIQUE_CLIENTS.value in available_metrics else (
        next((m for m in available_metrics if "нов" in m.lower() or "new" in m.lower()), None)
    )
    default_total_metric = Metrics.UNIQUE_CLIENTS.value if Metrics.UNIQUE_CLIENTS.value in available_metrics else (
        Metrics.LOAN_ISSUE_UNITS.value if Metrics.LOAN_ISSUE_UNITS.value in available_metrics else (
            next((m for m in available_metrics if "клиент" in m.lower()), None)
        )
    )
    if default_new_metric == default_total_metric:
        default_total_metric = next((m for m in available_metrics if m != default_new_metric), default_total_metric)

    if not available_metrics:
        st.info("Нет загруженных метрик для расчёта когорт.")
        return

    col_new, col_total = st.columns(2)
    new_metric = col_new.selectbox(
        "Метрика новых клиентов",
        options=available_metrics,
        index=available_metrics.index(default_new_metric) if default_new_metric in available_metrics else 0,
        help="Выберите источник для притока новых клиентов. Если специальной метрики нет, можно выбрать выдачи (шт).",
        key="cohort_new_metric",
    )
    total_metric = col_total.selectbox(
        "Метрика текущей базы",
        options=available_metrics,
        index=available_metrics.index(default_total_metric) if default_total_metric in available_metrics else 0,
        help="Выберите показатель, отражающий размер клиентской базы.",
        key="cohort_total_metric",
    )

    if new_metric == total_metric:
        st.warning("Выберите разные метрики для притока и базы — иначе удержание не посчитать.")
        return

    view_mode = st.radio(
        "Режим просмотра",
        options=["Сводно", "Сравнить регионы"],
        horizontal=True,
        key="cohort_view_mode"
    )

    new_series = _monthly_series_for_metric(ctx.df_current, tuple(ctx.regions), new_metric, ctx.months_range)
    total_series = _monthly_series_for_metric(ctx.df_current, tuple(ctx.regions), total_metric, ctx.months_range)
    if view_mode == "Сводно":
        if (new_series is None or new_series.empty) and (total_series is None or total_series.empty):
            st.info("Нет данных по выбранным метрикам. Попробуйте выбрать другой показатель или загрузить значения по клиентам.")
            return
        months = [m for m in ctx.months_range if (new_series is not None and m in new_series.index) or (total_series is not None and m in total_series.index)]
        if not months:
            st.info("Выберите период с метриками по клиентам.")
            return
        data = []
        has_retention = False
        for month in months:
            new_val = float(new_series.get(month, np.nan)) if new_series is not None else np.nan
            total_val = float(total_series.get(month, np.nan)) if total_series is not None else np.nan
            retention = np.nan
            share_new = np.nan
            if pd.notna(total_val) and total_val > 0:
                share_new = (new_val / total_val) * 100 if pd.notna(new_val) else np.nan
                if pd.notna(share_new):
                    retention = max(0.0, 100 - share_new)
                    has_retention = True
            data.append({
                "Месяц": month,
                "Новые клиенты": new_val,
                "Активная база": total_val,
                "Доля новых, %": share_new,
                "Удержание, %": retention,
            })
        df_clients = pd.DataFrame(data)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=df_clients["Месяц"],
                y=df_clients["Новые клиенты"],
                name="Новые клиенты",
                marker_color="rgba(168,85,247,0.55)",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=df_clients["Месяц"],
                y=df_clients["Активная база"],
                mode="lines+markers",
                name="Активная база",
                line=dict(color="#2563eb", width=3),
            ),
            secondary_y=False,
        )
        if has_retention and df_clients["Удержание, %"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df_clients["Месяц"],
                    y=df_clients["Удержание, %"],
                    mode="lines+markers",
                    name="Удержание, %",
                    line=dict(color="#10b981", width=3, dash="dot"),
                ),
                secondary_y=True,
            )
        fig.update_layout(
            height=380,
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="Клиентов", secondary_y=False)
        if has_retention and df_clients["Удержание, %"].notna().any():
            fig.update_yaxes(title_text="Удержание, %", secondary_y=True, range=[0, 110])
        else:
            fig.update_yaxes(secondary_y=True, showgrid=False, visible=False)
        st.plotly_chart(fig, use_container_width=True, key="cohort_trend")
        st.caption(f"Используются метрики: новые — «{new_metric}», база — «{total_metric}».")

        retention_matrix: Dict[str, Dict[str, float]] = {}
        for idx, month in enumerate(months):
            base = float(total_series.get(month, np.nan)) if total_series is not None else np.nan
            if pd.isna(base) or base <= 0:
                continue
            row: Dict[str, float] = {}
            for lag in range(1, 5):
                if idx + lag >= len(months):
                    break
                next_month = months[idx + lag]
                next_total = float(total_series.get(next_month, np.nan)) if total_series is not None else np.nan
                next_new = float(new_series.get(next_month, 0.0)) if new_series is not None else 0.0
                if pd.isna(next_total):
                    continue
                retained = max(0.0, next_total - max(0.0, next_new))
                row[f"+{lag} мес."] = (retained / base) * 100
            if row:
                retention_matrix[month] = row
        if has_retention and retention_matrix:
            cohort_df = pd.DataFrame(retention_matrix).T.fillna(0.0)
            heat = go.Figure(
                go.Heatmap(
                    z=cohort_df.values,
                    x=cohort_df.columns,
                    y=cohort_df.index,
                    colorscale="Blues",
                    zmin=0,
                    zmax=100,
                    hovertemplate="Старт %{y}<br>Горизонт %{x}<br>Удержание: %{z:.1f}%<extra></extra>",
                )
            )
            heat.update_layout(
                height=360,
                margin=dict(l=40, r=40, t=40, b=60),
            )
            st.plotly_chart(heat, use_container_width=True, key="cohort_heatmap")
            st.caption("Матрица удержания показывает, какая доля клиентской базы остаётся через n месяцев (по выбранным метрикам).")
        else:
            st.caption("Недостаточно данных для расчёта удержания — матрица скрыта, чтобы не вводить в заблуждение.")

        st.dataframe(
            df_clients,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Новые клиенты": st.column_config.NumberColumn("Новые клиенты", format="%.0f"),
                "Активная база": st.column_config.NumberColumn("Активная база", format="%.0f"),
                "Доля новых, %": st.column_config.NumberColumn("Доля новых, %", format="%.1f"),
                "Удержание, %": st.column_config.NumberColumn("Удержание, %", format="%.1f"),
            },
        )
    else:
        df_new_reg = _extract_region_month_metric(ctx.df_current, ctx.regions, new_metric, ctx.months_range)
        df_total_reg = _extract_region_month_metric(ctx.df_current, ctx.regions, total_metric, ctx.months_range)
        if df_new_reg.empty or df_total_reg.empty:
            st.info("Недостаточно данных по регионам для выбранных метрик. Попробуйте выбрать другой показатель или загрузить больше данных.")
            return
        merged = (
            df_total_reg.rename(columns={"Значение": "Активная база"})
            .merge(df_new_reg.rename(columns={"Значение": "Новые клиенты"}), on=["Регион", "Месяц"], how="outer")
        )
        merged["Месяц"] = pd.Categorical(merged["Месяц"], categories=ctx.months_range, ordered=True)
        merged = merged.sort_values(["Месяц", "Регион"])
        merged["Доля новых, %"] = np.where(
            merged["Активная база"] > 0,
            (merged["Новые клиенты"] / merged["Активная база"]) * 100,
            np.nan,
        )
        merged["Удержание, %"] = np.where(
            merged["Доля новых, %"].notna(),
            np.clip(100 - merged["Доля новых, %"], 0, 100),
            np.nan,
        )
        region_order = merged.groupby("Регион")["Новые клиенты"].sum().sort_values(ascending=False).index.tolist()
        default_regions = region_order[: min(5, len(region_order))]
        selected_regions = st.multiselect(
            "Регионы для сравнения",
            options=region_order,
            default=default_regions,
            key="cohort_region_select",
        )
        if not selected_regions:
            st.info("Выберите минимум один регион.")
            return
        panel = merged[merged["Регион"].isin(selected_regions)].copy()
        if panel.empty:
            st.info("Нет данных по выбранным регионам.")
            return
        line_share = px.line(
            panel,
            x="Месяц",
            y="Доля новых, %",
            color="Регион",
            markers=True,
            labels={"Доля новых, %": "Доля новых клиентов, %"},
        )
        line_share.update_layout(
            height=360,
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(line_share, use_container_width=True, key="cohort_share_compare")

        if panel["Удержание, %"].notna().any():
            line_ret = px.line(
                panel,
                x="Месяц",
                y="Удержание, %",
                color="Регион",
                markers=True,
                labels={"Удержание, %": "Удержание, %"},
            )
            line_ret.update_layout(
                height=360,
                margin=dict(l=40, r=20, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(line_ret, use_container_width=True, key="cohort_ret_compare")

        summary = (
            panel.groupby("Регион", as_index=False)[["Новые клиенты", "Активная база", "Доля новых, %", "Удержание, %"]]
            .agg({"Новые клиенты": "sum", "Активная база": "mean", "Доля новых, %": "mean", "Удержание, %": "mean"})
            .sort_values("Новые клиенты", ascending=False)
        )
        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Новые клиенты": st.column_config.NumberColumn("Новые клиенты (Σ)", format="%.0f"),
                "Активная база": st.column_config.NumberColumn("Активная база (ср.)", format="%.0f"),
                "Доля новых, %": st.column_config.NumberColumn("Доля новых, % (ср.)", format="%.1f"),
                "Удержание, %": st.column_config.NumberColumn("Удержание, % (ср.)", format="%.1f"),
            },
        )


def render_market_lab_page(ctx: PageContext) -> None:
    st.markdown("### 🧪 Сценарии «что если» по рынку")
    st.caption("Моделируем влияние изменения рынка, доли и скидок на ключевые показатели и динамику регионов.")
    base_revenue = period_value_from_itogo(ctx.df_current, ctx.regions, Metrics.REVENUE.value, ctx.months_range)
    base_issue = period_value_from_itogo(ctx.df_current, ctx.regions, Metrics.LOAN_ISSUE.value, ctx.months_range)
    base_markup = period_value_from_itogo(ctx.df_current, ctx.regions, Metrics.MARKUP_PCT.value, ctx.months_range)
    base_risk = period_value_from_itogo(ctx.df_current, ctx.regions, Metrics.RISK_SHARE.value, ctx.months_range)
    base_loss = period_value_from_itogo(ctx.df_current, ctx.regions, Metrics.BELOW_LOAN.value, ctx.months_range)

    col_market, col_share, col_discount = st.columns(3)
    market_growth = col_market.slider("Рынок (объём)", -30, 30, 0, step=2, format="%d%%")
    share_change = col_share.slider("Доля компании", -20, 20, 0, step=1, format="%d%%")
    discount_change = col_discount.slider(
        "Скидки / промо",
        -20,
        20,
        0,
        step=1,
        format="%d%%",
        help="Положительное значение — агрессивнее скидки, отрицательное — ужесточаем ценовую политику.",
    )

    def _safe(val):
        return float(val) if val is not None and not pd.isna(val) else None

    base_revenue = _safe(base_revenue)
    base_issue = _safe(base_issue)
    base_markup = _safe(_maybe_scale_percent(Metrics.MARKUP_PCT.value, base_markup))
    base_risk = _safe(_maybe_scale_percent(Metrics.RISK_SHARE.value, base_risk))
    base_loss = _safe(base_loss)

    factor_market = 1 + market_growth / 100
    factor_share = 1 + share_change / 100
    factor_discount = 1 - discount_change / 110
    new_revenue = base_revenue * factor_market * factor_share if base_revenue is not None else None
    new_issue = base_issue * factor_market if base_issue is not None else None
    new_markup = max(0.0, base_markup * factor_discount) if base_markup is not None else None
    risk_adjust = 1 + (discount_change / 35) - (share_change / 80)
    new_risk = max(0.0, base_risk * risk_adjust) if base_risk is not None else None
    new_loss = base_loss * (1 + discount_change / 30) if base_loss is not None else None

    col_metric_a, col_metric_b, col_metric_c = st.columns(3)
    col_metric_a.metric(
        "Выручка (рынок)",
        format_rub(new_revenue) if new_revenue is not None else "—",
        delta=format_rub(new_revenue - base_revenue) if new_revenue is not None and base_revenue is not None else "—",
    )
    col_metric_b.metric(
        "Наценка (новая)",
        fmt_pct(new_markup) if new_markup is not None else "—",
        delta=f"{(new_markup - base_markup):+.1f} п.п." if new_markup is not None and base_markup is not None else "—",
    )
    if new_risk is not None and base_risk is not None:
        risk_delta = f"{(new_risk - base_risk):+.1f} п.п."
    else:
        risk_delta = "—"
    col_metric_c.metric(
        "Риск ниже займа",
        fmt_pct(new_risk) if new_risk is not None else "—",
        delta=risk_delta,
    )
    st.caption("Расчёт ориентировочный: используем линейное приближение между скидками, долей и риском.")

    risk_df = _extract_region_month_metric(ctx.df_current, ctx.regions, Metrics.RISK_SHARE.value, ctx.months_range)
    markup_df = _extract_region_month_metric(ctx.df_current, ctx.regions, Metrics.MARKUP_PCT.value, ctx.months_range)
    revenue_df = _extract_region_month_metric(ctx.df_current, ctx.regions, Metrics.REVENUE.value, ctx.months_range)
    if not risk_df.empty and not markup_df.empty:
        merged = risk_df.rename(columns={"Значение": "Risk"}).merge(
            markup_df.rename(columns={"Значение": "Markup"}),
            on=["Регион", "Месяц"],
            how="inner"
        )
        if not revenue_df.empty:
            merged = merged.merge(
                revenue_df.rename(columns={"Значение": "Revenue"}),
                on=["Регион", "Месяц"],
                how="left"
            )
        merged["Risk"] = pd.to_numeric(merged["Risk"], errors="coerce")
        merged["Markup"] = pd.to_numeric(merged["Markup"], errors="coerce")
        merged["Revenue"] = pd.to_numeric(merged.get("Revenue"), errors="coerce").fillna(0.0)
        merged = merged.dropna(subset=["Risk", "Markup"])
        if not merged.empty:
            merged["Месяц"] = pd.Categorical(merged["Месяц"], categories=ctx.months_range, ordered=True)
            merged = merged.sort_values("Месяц")
            scatter = px.scatter(
                merged,
                x="Markup",
                y="Risk",
                color="Регион",
                size=merged["Revenue"].clip(lower=0.0) + 1,
                animation_frame="Месяц",
                size_max=28,
                labels={"Markup": "Наценка, %", "Risk": "Риск ниже займа, %"},
            )
            scatter.update_layout(
                height=420,
                margin=dict(l=40, r=40, t=60, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                xaxis=dict(
                    title="Наценка, %",
                    range=[0, max(10.0, float(merged["Markup"].max()) * 1.15)],
                ),
                yaxis=dict(
                    title="Риск ниже займа, %",
                    range=[0, max(5.0, float(merged["Risk"].max()) * 1.25)],
                ),
            )
            st.plotly_chart(scatter, use_container_width=True, key="market_animation")
            st.caption("Анимированный тренд показывает, как связка «наценка ↔ риск» эволюционирует по регионам.")
    else:
        st.info("Недостаточно данных для построения анимированного тренда по рынку.")

    insights = []
    if new_revenue is not None and base_revenue is not None:
        insights.append(f"Изменение рынка и доли двигает выручку на {format_rub(new_revenue - base_revenue)}.")
    if new_markup is not None and base_markup is not None:
        direction = "снижается" if new_markup < base_markup else "растёт"
        insights.append(f"Наценка {direction} до {fmt_pct(new_markup)} при выбранной скидочной политике.")
    if new_risk is not None and base_risk is not None:
        if new_risk > base_risk:
            insights.append("Риск продаж ниже займа растёт — закладывайте дополнительный лимит на убытки.")
        else:
            insights.append("Риск снижается — можно аккуратно усиливать промо.")
    if new_loss is not None and base_loss is not None:
        insights.append(f"Убыток от распродаж: {format_rub(new_loss)} (изменение {format_rub(new_loss - base_loss)}).")
    if insights:
        bullets = "\n".join(f"- {line}" for line in insights)
        st.markdown(f"**Что важно:**\n{bullets}")


def render_management_tools(ctx: PageContext, stats_current: Dict[str, Dict[str, Any]], stats_previous: Dict[str, Dict[str, Any]] | None) -> None:
    st.markdown("### 🧑‍💼 Управленческий отчёт")
    st.caption("Сформируйте краткий executive-дайджест и поделитесь им в Markdown, PDF или e-mail.")
    summary_lines, action_lines = build_metric_recommendations(
        stats_current,
        ctx.scenario_name,
        ctx.months_range,
        baseline_map=stats_previous or None,
    )
    period_label = f"{ctx.months_range[0]} – {ctx.months_range[-1]}" if ctx.months_range else "Период не выбран"
    report_lines = [
        f"# Executive Brief — НЮЗ ({period_label})",
        f"**Сценарий:** {ctx.scenario_name}",
        f"**Режим анализа:** {'Сравнение годов' if ctx.mode == 'compare' else 'Один год'}",
        f"**Регионов в выборке:** {len(ctx.regions)}",
        "",
        "## KPI Snapshot",
    ]
    if summary_lines:
        report_lines.extend(f"- {line}" for line in summary_lines)
    else:
        report_lines.append("- Недостаточно данных для формирования ключевых выводов.")
    report_lines.append("")
    report_lines.append("## Priority Actions")
    if action_lines:
        report_lines.extend(f"{idx}. {line}" for idx, line in enumerate(action_lines, start=1))
    else:
        report_lines.append("1. Загрузите дополнительные показатели, чтобы подготовить рекомендации.")
    report_lines.append("")
    report_lines.append("## Контекст и покрытие")
    report_lines.append(f"- Период: {period_label}")
    sample_regions = ", ".join(ctx.regions[:10]) + (" …" if len(ctx.regions) > 10 else "")
    report_lines.append(f"- Регионы: {sample_regions if sample_regions else 'не выбраны'}")

    report_md = "\n".join(report_lines)
    report_plain = report_md.replace("**", "").replace("#", "")

    with st.expander("Предпросмотр отчёта", expanded=False):
        st.markdown(report_md)

    st.download_button(
        "⬇️ Скачать отчёт (Markdown)",
        data=report_md.encode("utf-8"),
        file_name="NUZ_management_report.md",
        mime="text/markdown",
    )

    pdf_bytes = None
    try:
        from reportlab.lib.pagesizes import A4  # type: ignore
        from reportlab.pdfgen import canvas  # type: ignore

        buffer = BytesIO()
        canv = canvas.Canvas(buffer, pagesize=A4)
        text = canv.beginText(40, 800)
        text.setFont("Helvetica", 11)
        for line in report_plain.splitlines():
            text.textLine(line)
            if text.getY() <= 40:
                canv.drawText(text)
                canv.showPage()
                text = canv.beginText(40, 800)
                text.setFont("Helvetica", 11)
        canv.drawText(text)
        canv.save()
        pdf_bytes = buffer.getvalue()
    except Exception:
        pdf_bytes = None

    if pdf_bytes:
        st.download_button(
            "⬇️ Отчёт в PDF",
            data=pdf_bytes,
            file_name="NUZ_management_report.pdf",
            mime="application/pdf",
        )
    else:
        st.caption("Для выгрузки в PDF установите пакет `reportlab` (`pip install reportlab`).")

    with st.expander("E-mail дайджест", expanded=False):
        st.caption("Данные формы сохраняются только в сессии Streamlit.")
        default_subject = f"NUZ дайджест — {period_label}"
        smtp_defaults = st.session_state.setdefault(
            "email_digest_defaults",
            {"server": "", "port": 465, "use_ssl": True, "user": "", "recipients": ""},
        )
        with st.form("email_digest_form"):
            smtp_server = st.text_input("SMTP сервер", value=smtp_defaults["server"])
            smtp_port = st.number_input("Порт", value=smtp_defaults["port"], min_value=1, max_value=65535, step=1)
            use_ssl = st.toggle("Использовать SSL (рекомендуется)", value=smtp_defaults["use_ssl"])
            smtp_user = st.text_input("Логин", value=smtp_defaults["user"])
            smtp_password = st.text_input("Пароль", type="password")
            recipients = st.text_input("Получатели", value=smtp_defaults["recipients"], help="Через запятую")
            subject = st.text_input("Тема письма", value=default_subject)
            send_btn = st.form_submit_button("Отправить дайджест")

        smtp_defaults.update({"server": smtp_server, "port": smtp_port, "use_ssl": use_ssl, "user": smtp_user, "recipients": recipients})

        if send_btn:
            if not smtp_server or not recipients:
                st.error("Укажите SMTP сервер и получателей.")
            else:
                try:
                    from email.mime.text import MIMEText  # type: ignore
                    import smtplib  # type: ignore

                    recipient_list = [addr.strip() for addr in recipients.split(",") if addr.strip()]
                    if not recipient_list:
                        st.error("Нет валидных адресов получателей.")
                    else:
                        msg = MIMEText(report_plain, "plain", "utf-8")
                        sender = smtp_user or "noreply@example.com"
                        msg["Subject"] = subject
                        msg["From"] = sender
                        msg["To"] = ", ".join(recipient_list)

                        if use_ssl:
                            server = smtplib.SMTP_SSL(smtp_server, int(smtp_port))
                        else:
                            server = smtplib.SMTP(smtp_server, int(smtp_port))
                            server.starttls()
                        if smtp_user and smtp_password:
                            server.login(smtp_user, smtp_password)
                        server.sendmail(sender, recipient_list, msg.as_string())
                        server.quit()
                        st.success("Дайджест отправлен.")
                except Exception as exc:  # pragma: no cover
                    st.error(f"Не удалось отправить письмо: {exc}")


def render_forecast_page(ctx: PageContext) -> None:
    st.markdown("### 🔮 Прогноз ключевых метрик")
    st.caption("Прогноз основан на линейном тренде с доверительным интервалом 95%.")
    available_metrics = [m for m in FORECAST_METRICS if m in ctx.df_current["Показатель"].unique()]
    if not available_metrics:
        st.info("Недостаточно данных для построения прогнозов.")
        return
    horizon = st.slider("Горизонт прогноза, мес.", min_value=1, max_value=6, value=3, step=1, key="forecast_horizon")
    for metric in available_metrics:
        forecast_bundle = _prepare_forecast(ctx.df_current, ctx.regions, ctx.months_range, metric, horizon=horizon)
        if not forecast_bundle:
            continue
        history = forecast_bundle["history"]
        future_labels = forecast_bundle["future_labels"]
        forecast_vals = forecast_bundle["forecast"]
        lower_vals = forecast_bundle["lower"]
        upper_vals = forecast_bundle["upper"]
        next_value = forecast_vals[0]
        delta_value = next_value - float(history.iloc[-1])

        st.markdown(f"#### {metric}")
        col_info, col_chart = st.columns([1, 2])
        with col_info:
            st.metric(
                f"Прогноз на {future_labels[0]}",
                _format_value_for_metric(metric, next_value),
                delta=_format_value_for_metric(metric, delta_value)
            )
            model_tag = forecast_bundle.get("selected_model") or forecast_bundle.get("method")
            if model_tag == "seasonal":
                model_text = "Модель: линейный тренд + сезонность по месяцам"
            else:
                model_text = "Модель: линейный тренд"
            st.caption(model_text)
            st.caption(f"Среднеквадратичное отклонение: {forecast_bundle['sigma']:.2f}")
            if forecast_bundle.get("selected_model") == "seasonal" and forecast_bundle.get("seasonal"):
                top_seasonal = sorted(forecast_bundle["seasonal"].items(), key=lambda kv: abs(kv[1]), reverse=True)[:2]
                if top_seasonal:
                    seasonal_desc = ", ".join(
                        f"{ORDER[idx % len(ORDER)]}: {value:+.1f}" for idx, value in top_seasonal
                    )
                    st.caption(f"Сезонные поправки: {seasonal_desc}")
            if forecast_bundle.get("selected_model") == "seasonal" and forecast_bundle.get("baseline_sse") and forecast_bundle.get("sse"):
                base_sse = float(forecast_bundle["baseline_sse"]) or 1e-9
                gain = 1 - (float(forecast_bundle["sse"]) / base_sse)
                st.caption(f"Точность улучшена на {gain:.1%} относительно чистого тренда.")
        with col_chart:
            fig = go.Figure()
            hist_x = [str(x) for x in history.index]
            fig.add_trace(go.Scatter(
                x=hist_x,
                y=history.values,
                mode="lines+markers",
                name="Факт",
                line=dict(color="#2563eb", width=3),
            ))
            forecast_x = [hist_x[-1]] + future_labels
            forecast_y = [history.values[-1]] + forecast_vals
            fig.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast_y,
                mode="lines+markers",
                name="Прогноз",
                line=dict(color="#7c3aed", width=2, dash="dot"),
                marker=dict(symbol="circle"),
            ))
            ci_x = future_labels + future_labels[::-1]
            ci_y = upper_vals + lower_vals[::-1]
            fig.add_trace(go.Scatter(
                x=ci_x,
                y=ci_y,
                fill="toself",
                fillcolor="rgba(124,58,237,0.12)",
                line=dict(width=0),
                hoverinfo="skip",
                name="95% интервал"
            ))
            fig.update_layout(
                height=360,
                margin=dict(l=20, r=20, t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            st.plotly_chart(fig, use_container_width=True, key=f"forecast_{_normalize_metric_label(metric)}")

        summary_df = pd.DataFrame({
            "Период": future_labels,
            "Прогноз": forecast_vals,
            "Нижняя граница": lower_vals,
            "Верхняя граница": upper_vals,
        })
        st.dataframe(
            summary_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Прогноз": st.column_config.NumberColumn(format="%.2f"),
                "Нижняя граница": st.column_config.NumberColumn(format="%.2f"),
                "Верхняя граница": st.column_config.NumberColumn(format="%.2f"),
            }
        )








def scenario_overview_block_single(
    df_scope: pd.DataFrame,
    regions: List[str],
    months_range: List[str],
    scenario_name: str,
    year_selected: int,
    *,
    stats_map: Dict[str, Dict[str, Any]] | None = None,
) -> None:
    if stats_map is None:
        stats_map = {
            metric: compute_metric_stats(df_scope, regions, months_range, metric)
            for metric in KEY_DECISION_METRICS + SUPPORT_DECISION_METRICS
        }
    summary_lines, action_lines = build_metric_recommendations(stats_map, scenario_name, months_range)
    st.markdown(f"### 🎯 {scenario_name}: что происходит ({year_selected})")
    desc = SCENARIO_DESCRIPTIONS.get(scenario_name)
    if desc:
        st.caption(desc)
    _render_insights("Ключевые выводы", summary_lines)
    _render_plan("Первичные действия", action_lines[:6])


def scenario_overview_block_compare(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    regions: List[str],
    months_range: List[str],
    scenario_name: str,
    year_a: int,
    year_b: int,
    *,
    stats_current: Dict[str, Dict[str, Any]] | None = None,
    stats_previous: Dict[str, Dict[str, Any]] | None = None,
) -> None:
    if stats_current is None:
        stats_current = {
            metric: compute_metric_stats(df_b, regions, months_range, metric)
            for metric in KEY_DECISION_METRICS + SUPPORT_DECISION_METRICS
        }
    if stats_previous is None:
        stats_previous = {
            metric: compute_metric_stats(df_a, regions, months_range, metric)
            for metric in KEY_DECISION_METRICS + SUPPORT_DECISION_METRICS
        }
    summary_lines, action_lines = build_metric_recommendations(
        stats_current, scenario_name, months_range, baseline_map=stats_previous
    )
    st.markdown(f"### 🎯 {scenario_name}: что происходит ({year_b} vs {year_a})")
    desc = SCENARIO_DESCRIPTIONS.get(scenario_name)
    if desc:
        st.caption(desc)
    _render_insights("Ключевые выводы", summary_lines)
    _render_plan("План действий", action_lines[:6])


def _build_single_year_prompt(year: int, period_label: str, region_list: List[str], metrics: Dict[str, float | None], *, df_source: pd.DataFrame, months_range: List[str], monthly_context: str, forecast_target: str) -> str:
    metrics_lines = "\n".join(_format_metric_for_prompt(k, metrics.get(k)) for k in AI_METRICS_FOCUS)
    raw_values = {k: (None if metrics.get(k) is None else float(metrics[k])) for k in AI_METRICS_FOCUS}
    regional_lines = _regional_context_block(df_source, region_list, months_range, AI_REGION_METRICS)
    instructions = (
        f"Ты — аналитик сети ломбардов. Подготовь отчёт, который помогает принять решения:\n"
        "1. **Ключевые выводы** — 3 тезиса с главными тенденциями; приводите только критически важные числа.\n"
        "2. **Диагностика** — объясни, почему показатели так изменились (по месяцам и регионам), сосредоточься на причинах.\n"
        "3. **Риски и сигналы** — 3–4 пункта с вероятными последствиями и зонами внимания.\n"
        f"4. **Мероприятия и прогноз** — предложи 3 конкретных действия (приоритет, ожидаемый эффект) и дай прогноз на период {forecast_target}.\n"
        "Игнорируй метрики без данных и избегай перечисления всех цифр подряд — цитируй только те значения, которые нужны для аргумента."
        "\nПравила интерпретации:"
        "\n- Нормальная доля продаж ниже займа < 12%."
        "\n- Доля неликвида > 30% — высокий риск склада."
        "\n- Доходность < 15% при росте выдач — проверить просрочку."
        "\n- Резкий рост выручки при снижении наценки — возможный демпинг."
    )
    text = (
        f"Период: {period_label}, год: {year}. Регионов в выборке: {len(region_list)}."
        f"\nСписок регионов: {', '.join(region_list[:8])}{'...' if len(region_list) > 8 else ''}."
        f"\nДанные метрик:\n{metrics_lines}"
    )
    if regional_lines:
        text += f"\n\nРегиональные срезы:\n{regional_lines}"
    if monthly_context:
        text += f"\n\nПомесячные ряды:\n{monthly_context}"
    text += f"\n\nСырые значения (для вычислений): {raw_values}"
    text += f"\nЦель прогноза: {forecast_target}"
    text += f"\n\n{instructions}"
    return text


def _build_compare_prompt(year_a: int, year_b: int, period_label: str, region_list: List[str], metrics_a: Dict[str, float | None], metrics_b: Dict[str, float | None], *, df_a: pd.DataFrame, df_b: pd.DataFrame, months_range: List[str], monthly_a: str, monthly_b: str, forecast_target: str) -> str:
    def fmt_pair(metric: str) -> str:
        v_a, v_b = metrics_a.get(metric), metrics_b.get(metric)
        if v_a is None and v_b is None:
            return f"{metric}: нет данных"
        delta = None
        if v_a is not None and v_b is not None:
            delta = v_b - v_a
        base = f"{_format_metric_for_prompt(metric, v_a)} → {year_b}: {_format_metric_for_prompt(metric, v_b)}"
        if delta is None:
            return base
        if is_percent_metric(metric):
            return base + f" (Δ={delta:.2f} п.п.)"
        if "руб" in metric:
            return base + f" (Δ={delta:,.0f} руб)".replace(",", " ")
        return base + f" (Δ={delta:,.0f})".replace(",", " ")

    lines = "\n".join(fmt_pair(m) for m in AI_METRICS_FOCUS)
    raw_block = {
        metric: {
            str(year_a): None if metrics_a.get(metric) is None else float(metrics_a[metric]),
            str(year_b): None if metrics_b.get(metric) is None else float(metrics_b[metric]),
        }
        for metric in AI_METRICS_FOCUS
    }
    regional_a = _regional_context_block(df_a, region_list, months_range, AI_REGION_METRICS)
    regional_b = _regional_context_block(df_b, region_list, months_range, AI_REGION_METRICS)
    instructions = (
        f"Ты — аналитик. Подготовь сравнение годов, ориентируясь на управленческие решения:\n"
        "1. **Итоги** — 3–4 ключевых изменения с краткими аргументами (используй только значимые цифры).\n"
        "2. **Помесячное сравнение** — объясни 2–3 наибольших расхождения по месяцам и их причины.\n"
        "3. **Риски и сигналы** — 3 пункта с последствиями для бизнеса.\n"
        f"4. **Действия и прогноз** — предложи 3 мероприятия (приоритет, ожидаемый эффект) и прогноз на период {forecast_target} для {year_b}.\n"
        "Избегай перечисления всех чисел подряд; упоминай только те значения, без которых вывод неубедителен."
    )
    text = (
        f"Сравнение {year_b} против {year_a}, период: {period_label}. Регионов: {len(region_list)}."
        f"\nРегиональная выборка: {', '.join(region_list[:8])}{'...' if len(region_list) > 8 else ''}."
        f"\nМетрики:\n{lines}"
    )
    if regional_a:
        text += f"\n\n{year_a}:\n{regional_a}"
    if regional_b:
        text += f"\n\n{year_b}:\n{regional_b}"
    if monthly_a:
        text += f"\n\n{year_a} помесячно:\n{monthly_a}"
    if monthly_b:
        text += f"\n\n{year_b} помесячно:\n{monthly_b}"
    text += f"\n\nСырые значения: {raw_block}"
    text += f"\nЦель прогноза: {forecast_target}"
    text += f"\n\n{instructions}"
    return text


GEMINI_ENDPOINT_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


def _call_gemini(api_key: str, prompt: str, model: str = GEMINI_DEFAULT_MODEL, *, timeout: int = 40) -> str:
    api_key = api_key.strip()
    if not api_key:
        raise RuntimeError("Gemini API-ключ пустой.")
    url = f"{GEMINI_ENDPOINT_BASE}/{model}:generateContent"
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": api_key}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": str(prompt)}
                ]
            }
        ]
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API вернул {resp.status_code}: {resp.text[:200]}")
    try:
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise KeyError("candidates")
        parts = candidates[0].get("content", {}).get("parts") or []
        if not parts:
            raise KeyError("parts")
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        joined = "\n".join(t for t in texts if t)
        return joined.strip() or "(Gemini вернул пустой ответ)"
    except (KeyError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Не удалось разобрать ответ Gemini: {resp.text[:200]}") from exc


def _resolve_gemini_key() -> str:
    cached = st.session_state.get("ai_gemini_key")
    if cached:
        return cached
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return ""


def _save_gemini_key(value: str) -> None:
    if value:
        st.session_state["ai_gemini_key"] = value.strip()


def ai_analysis_block_single_year(df_scope: pd.DataFrame, regions: List[str], months_range: List[str], year_selected: int) -> None:
    st.subheader("🤖 AI-анализ периода")
    st.caption("Интеграция с Google Gemini. Передавайте обезличенные данные.")

    if "ai_gemini_key_input" not in st.session_state:
        st.session_state["ai_gemini_key_input"] = _resolve_gemini_key()
    gemini_key = st.text_input(
        "Gemini API-ключ",
        type="password",
        key="ai_gemini_key_input",
        help="Создайте ключ в Google AI Studio и храните его в секрете."
    )
    _save_gemini_key(gemini_key)
    default_index = GEMINI_MODELS.index(GEMINI_DEFAULT_MODEL) if GEMINI_DEFAULT_MODEL in GEMINI_MODELS else 0
    model_id = st.selectbox(
        "Модель Gemini",
        options=GEMINI_MODELS,
        index=default_index,
        key="ai_gemini_model_single",
        help="Вызовы идут напрямую в Google Generative Language API."
    )

    period_label = months_range[0] if len(months_range) == 1 else f"{months_range[0]} – {months_range[-1]}"
    metrics = _collect_period_metrics(df_scope, regions, months_range)

    metrics_df = pd.DataFrame(
        {
            "Показатель": list(metrics.keys()),
            "Значение": [
                fmt_pct(v) if is_percent_metric(k) else (
                    fmt_days(v) if "дней" in k else format_rub(v) if "руб" in k else ("—" if v is None else f"{v:,.0f}".replace(",", " "))
                )
                for k, v in metrics.items()
            ],
        }
    )
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    cache = st.session_state.setdefault("ai_analysis_cache", {})
    cache_key = (
        "single",
        model_id,
        year_selected,
        tuple(sorted(regions)),
        tuple(months_range),
        tuple(sorted((k, None if metrics[k] is None else round(float(metrics[k]), 4)) for k in metrics))
    )

    monthly_context = _monthly_context_block(df_scope, regions, months_range, AI_METRICS_FOCUS)
    forecast_target = _forecast_target_label(months_range[-1]) if months_range else "Следующий период"

    prompt = _build_single_year_prompt(
        year_selected,
        period_label,
        regions,
        metrics,
        df_source=df_scope,
        months_range=months_range,
        monthly_context=monthly_context,
        forecast_target=forecast_target,
    )
    with st.expander("Промпт (для проверки)", expanded=False):
        st.code(prompt)

    result_placeholder = st.empty()
    if cache_key in cache:
        result_placeholder.markdown(cache[cache_key])

    if st.button("Сгенерировать анализ", type="primary"):
        if not gemini_key:
            st.error("Введите API-ключ Gemini.")
            return
        call_func = lambda: _call_gemini(gemini_key, prompt, model=model_id)
        with st.spinner("Запрашиваю модель…"):
            try:
                ai_text = call_func()
                cache[cache_key] = ai_text
                result_placeholder.markdown(ai_text)
            except Exception as exc:
                st.error(str(exc))


def ai_analysis_block_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame, regions: List[str], months_range: List[str], year_a: int, year_b: int) -> None:
    st.subheader("🤖 AI-анализ (сравнение годов)")
    st.caption("Используем только Google Gemini — ключ храните в секрете.")

    if "ai_gemini_key_input" not in st.session_state:
        st.session_state["ai_gemini_key_input"] = _resolve_gemini_key()
    gemini_key = st.text_input(
        "Gemini API-ключ",
        type="password",
        key="ai_gemini_key_input",
        help="Создайте ключ в Google AI Studio и вставьте его сюда."
    )
    _save_gemini_key(gemini_key)
    default_index = GEMINI_MODELS.index(GEMINI_DEFAULT_MODEL) if GEMINI_DEFAULT_MODEL in GEMINI_MODELS else 0
    model_id = st.selectbox(
        "Модель Gemini",
        options=GEMINI_MODELS,
        index=default_index,
        key="ai_gemini_model_compare",
        help="Вызовы идут напрямую в Google Generative Language API."
    )

    period_label = months_range[0] if len(months_range) == 1 else f"{months_range[0]} – {months_range[-1]}"
    metrics_a, metrics_b = _collect_comparison_metrics(df_a, df_b, regions, months_range)

    df_display = pd.DataFrame(
        {
            "Показатель": AI_METRICS_FOCUS,
            f"{year_a}": [metrics_a.get(m) for m in AI_METRICS_FOCUS],
            f"{year_b}": [metrics_b.get(m) for m in AI_METRICS_FOCUS],
        }
    )
    for col in [f"{year_a}", f"{year_b}"]:
        df_display[col] = [
            fmt_pct(v) if is_percent_metric(metric) else (
                fmt_days(v) if "дней" in metric else format_rub(v) if "руб" in metric else ("—" if v is None else f"{v:,.0f}".replace(",", " "))
            )
            for metric, v in zip(AI_METRICS_FOCUS, [metrics_a.get(metric) if col == f"{year_a}" else metrics_b.get(metric) for metric in AI_METRICS_FOCUS])
        ]
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    cache = st.session_state.setdefault("ai_analysis_cache", {})
    cache_key = (
        "compare",
        model_id,
        year_a,
        year_b,
        tuple(sorted(regions)),
        tuple(months_range),
        tuple(sorted((metric, None if metrics_a.get(metric) is None else round(float(metrics_a[metric]), 4)) for metric in AI_METRICS_FOCUS)),
        tuple(sorted((metric, None if metrics_b.get(metric) is None else round(float(metrics_b[metric]), 4)) for metric in AI_METRICS_FOCUS))
    )

    monthly_a = _monthly_context_block(df_a, regions, months_range, AI_METRICS_FOCUS)
    monthly_b = _monthly_context_block(df_b, regions, months_range, AI_METRICS_FOCUS)
    forecast_target = _forecast_target_label(months_range[-1]) if months_range else "Следующий период"

    prompt = _build_compare_prompt(
        year_a,
        year_b,
        period_label,
        regions,
        metrics_a,
        metrics_b,
        df_a=df_a,
        df_b=df_b,
        months_range=months_range,
        monthly_a=monthly_a,
        monthly_b=monthly_b,
        forecast_target=forecast_target,
    )
    with st.expander("Промпт (для проверки)", expanded=False):
        st.code(prompt)

    result_placeholder = st.empty()
    if cache_key in cache:
        result_placeholder.markdown(cache[cache_key])

    if st.button("Сгенерировать анализ", type="primary"):
        if not gemini_key:
            st.error("Введите API-ключ Gemini.")
            return
        call_func = lambda: _call_gemini(gemini_key, prompt, model=model_id)
        with st.spinner("Запрашиваю модель…"):
            try:
                ai_text = call_func()
                cache[cache_key] = ai_text
                result_placeholder.markdown(ai_text)
            except Exception as exc:
                st.error(str(exc))

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
    if name is None:
        return ""
    key = _normalize_metric_label(name)
    return METRIC_ALIAS_MAP.get(key, "")

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
        if not metric_name:
            continue

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

        override_cat = METRIC_CATEGORY_OVERRIDES.get(metric_name)
        if override_cat:
            cat = override_cat

        if NUZ_ONLY and str(cat).strip().lower() != "нюз":
            continue

        # Обновляем «липкую» метку, если нашли явную
        if cat in {"НЮЗ", "ЮЗ"}:
            last_cat = cat

        code_match = re.search(r"№\s*(\d+)", str(current_branch))
        code = code_match.group(1) if code_match else ""

        month_values = []
        raw_total_value = np.nan
        for j in month_indices:
            month_label = month_map[j]
            value = coerce_number(df.iat[r, j])
            if month_label == "Итого":
                raw_total_value = value
                continue
            month_values.append((month_label, value))

        if not month_values:
            continue

        arr = np.array([val for _, val in month_values], dtype=float)
        mask_fact = ~np.isnan(arr)
        if not mask_fact.any():
            continue

        first, last = int(np.where(mask_fact)[0][0]), int(np.where(mask_fact)[0][-1])
        clean_months: list[tuple[str, float]] = []
        for idx, (month_label, value) in enumerate(month_values):
            if np.isnan(value) or ((idx < first or idx > last) and (abs(value) <= 1e-12)):
                continue
            clean_months.append((month_label, float(value)))
            rows.append({
                "Регион": str(canonical_region),
                "ИсточникФайла": "TOTALS_FILE" if is_totals_file else "BRANCHES_FILE",
                "Код": code,
                "Подразделение": str(current_branch),
                "Показатель": metric_name,
                "Месяц": month_label,
                "Значение": float(value),
                "Категория": cat,
            })

        if not clean_months:
            continue

        total_value = np.nan
        series = pd.Series([val for _, val in clean_months], index=[m for m, _ in clean_months], dtype=float)
        rule = aggregation_rule(metric_name)

        if rule == "sum":
            total_value = float(series.sum())
        elif rule == "mean":
            total_value = float(series.mean())
        elif rule == "last":
            total_value = float(series.iloc[-1])

        if np.isnan(total_value) and not np.isnan(raw_total_value):
            total_value = float(raw_total_value)

        if not np.isnan(total_value):
            rows.append({
                "Регион": str(canonical_region),
                "ИсточникФайла": "RECALC_TOTAL",
                "Код": code,
                "Подразделение": str(current_branch),
                "Показатель": metric_name,
                "Месяц": "Итого",
                "Значение": float(total_value),
                "Категория": cat,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Данные не распознаны.")

    if NUZ_ONLY:
        mask_nuz = (
            out.get("Категория")
               .astype(str)
               .str.strip()
               .str.lower()
               .eq("нюз")
        )
        out = out.loc[mask_nuz].copy()
        if out.empty:
            raise ValueError("В файле не найдено строк с данными НЮЗ.")

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
    pivot = sub.pivot_table(index=["Регион","Подразделение","Месяц"], columns="Показатель", values="Значение", aggfunc="sum", observed=False)
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
        insight_lines = []
        for metric in KPI_COLUMNS:
            if metric in kpi_table.columns:
                insight = _describe_metric_series(kpi_table[metric], metric)
                if insight:
                    insight_lines.append(insight)
        _render_insights("Коротко по KPI", insight_lines)
        action_lines: List[str] = []
        for metric in KPI_COLUMNS:
            if metric in kpi_table.columns:
                action_lines.extend(_generate_actions_for_series(kpi_table[metric], metric))
        _render_plan("Что делаем по KPI", action_lines[:4])
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

    insight_lines: List[str] = []
    action_lines: List[str] = []

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
            insight = _describe_metric_series(pd.Series(per_reg), Metrics.AVG_LOAN_TERM.value)
            if insight:
                insight_lines.append(insight)
            action_lines.extend(_generate_actions_for_series(pd.Series(per_reg), Metrics.AVG_LOAN_TERM.value))

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
            insight = _describe_metric_series(pd.Series(per_reg_avg_loan), Metrics.AVG_LOAN.value)
            if insight:
                insight_lines.append(insight)
            action_lines.extend(_generate_actions_for_series(pd.Series(per_reg_avg_loan), Metrics.AVG_LOAN.value))

    _render_metric_if_value(cH, f"Выкупленные залоги ({lbl_cur})", v_redeemed, kind="num")

    # Общие выводы по периодам
    monthly_focus_metrics = [Metrics.REVENUE.value, Metrics.MARKUP_PCT.value, Metrics.RISK_SHARE.value]
    for metric in monthly_focus_metrics:
        series = _monthly_series_for_metric(df_all, regions, metric, months_range)
        if series is None or series.empty or len(series.dropna()) < 2:
            continue
        first, last = series.dropna().iloc[0], series.dropna().iloc[-1]
        delta = last - first
        direction = "вырос" if delta > 0 else "снизился"
        if abs(delta) < 1e-6:
            continue
        insight_lines.append(
            f"{metric}: {direction} с {_format_value_for_metric(metric, first)} до {_format_value_for_metric(metric, last)} ({_format_delta(metric, delta)})."
        )
        action_lines.extend(_generate_actions_for_deltas([(f"выборка", delta)], metric))

    _render_insights("Выводы по периоду", insight_lines)
    _render_plan("Ближайшие действия", action_lines[:4])

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
    raw_metrics = [m for m in sorted(sub["Показатель"].dropna().unique()) if m not in HIDDEN_METRICS]

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
        use_container_width=True,
        column_config=default_column_config(region_summary)
    )

    key_metrics = [
        m for m in [Metrics.REVENUE.value, Metrics.MARKUP_PCT.value, Metrics.RISK_SHARE.value, Metrics.ILLIQUID_BY_VALUE_PCT.value]
        if m in region_summary.columns
    ]
    insight_lines = []
    for metric in key_metrics:
        insight = _describe_metric_series(region_summary[metric], metric)
        if insight:
            insight_lines.append(insight)
    _render_insights("Главное по регионам", insight_lines)

    action_lines: List[str] = []
    for metric in key_metrics:
        action_lines.extend(_generate_actions_for_series(region_summary[metric], metric))
    _render_plan("Шаги по регионам", action_lines[:4])

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

def leaderboard_block(
    df_all: pd.DataFrame,
    regions: list[str],
    available_months: list[str],
    *,
    default_metric: str | None = None,
    selection_key: str = "leaderboard_metric",
    period_slider_key: str = "leaderboard_period"
) -> None:
    st.subheader("🏆 Лидеры и аутсайдеры")
    st.caption("Сравниваем филиалы по выбранной метрике: слева — лидеры, справа — те, кто требует внимания.")

    if df_all.empty or not available_months:
        st.info("Нет данных для отображения.")
        return

    last_quarter = available_months[max(0, len(available_months)-3):]
    start_m, end_m = st.select_slider(
        "Выберите период для рейтинга:", options=available_months,
        value=(last_quarter[0], last_quarter[-1]),
        key=period_slider_key
    )
    leaderboard_months = ORDER[ORDER.index(start_m): ORDER.index(end_m) + 1]

    agg_data = get_aggregated_data(df_all, tuple(regions), tuple(leaderboard_months))
    if agg_data.empty:
        st.warning("Нет данных за выбранный период.")
        return

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
    metric_options = sorted([c for c in numeric_cols if c in raw_metric_names and c not in HIDDEN_METRICS])

    if not metric_options:
        st.warning("В исходных файлах не найдено числовых метрик для рейтинга.")
        return

    default_idx = 0
    if default_metric and default_metric in metric_options:
        default_idx = metric_options.index(default_metric)
    elif Metrics.REVENUE.value in metric_options:
        default_idx = metric_options.index(Metrics.REVENUE.value)
    chosen_metric = st.selectbox(
        "Показатель",
        options=metric_options,
        index=default_idx,
        key=selection_key
    )
    st.caption(METRIC_HELP.get(chosen_metric, ""))

    # определим правило «чем больше — тем лучше»
    percent_metric = is_percent_metric(chosen_metric)
    is_money = "руб" in chosen_metric.lower()
    is_days = "дней" in chosen_metric.lower()
    if chosen_metric in METRICS_BIGGER_IS_BETTER:
        ascending = False
    elif chosen_metric in METRICS_SMALLER_IS_BETTER:
        ascending = True
    else:
        ascending = False

    sorted_data = agg_data.dropna(subset=[chosen_metric]).sort_values(by=chosen_metric, ascending=ascending)
    metric_series = sorted_data[chosen_metric]
    if not metric_series.empty:
        if not percent_metric:
            total_val = float(metric_series.abs().sum())
            if total_val > 1e-9:
                sorted_data["Доля, %"] = (metric_series / total_val) * 100
        mean_val = float(metric_series.mean()) if not metric_series.empty else 0.0
        sorted_data["Отклонение от среднего"] = metric_series - mean_val
    top_limit = min(20, len(sorted_data)) if not sorted_data.empty else 0
    if top_limit == 0:
        st.warning("Нет записей для рейтинга после сортировки")
        return
    min_slider = 5 if top_limit >= 5 else 1
    default_val = min(10, top_limit)
    default_val = default_val if default_val >= min_slider else top_limit
    top_n = st.slider(
        "Размер выборки",
        min_value=min_slider,
        max_value=top_limit,
        value=default_val,
        step=1,
        key=f"{selection_key}_topn"
    )

    if chosen_metric in METRICS_SMALLER_IS_BETTER:
        title_best = f"✅ Топ-{top_n} лучших (меньше = лучше)"
        title_worst = f"❌ Топ-{top_n} худших (больше = хуже)"
    else:
        title_best = f"✅ Топ-{top_n} лучших"
        title_worst = f"❌ Топ-{top_n} худших"

    c1, c2 = st.columns(2)
    display_cols = ["Подразделение","Регион",chosen_metric]
    if "Доля, %" in sorted_data.columns:
        display_cols.append("Доля, %")
    if "Отклонение от среднего" in sorted_data.columns:
        display_cols.append("Отклонение от среднего")

    col_cfg = default_column_config(sorted_data)
    if "Доля, %" in display_cols:
        col_cfg["Доля, %"] = st.column_config.NumberColumn("Доля от итога, %", format="%.1f%%")
    if "Отклонение от среднего" in display_cols:
        format_label = "Отклонение, п.п." if percent_metric else ("Отклонение, дн." if is_days else "Отклонение от среднего")
        fmt = "%.2f"
        if is_money and not percent_metric and not is_days:
            fmt = "%.0f"
        col_cfg["Отклонение от среднего"] = st.column_config.NumberColumn(format_label, format=fmt)

    with c1:
        st.markdown(f"**{title_best} по _{chosen_metric}_**")
        st.dataframe(sorted_data.head(top_n)[display_cols], use_container_width=True, column_config=col_cfg)
    with c2:
        st.markdown(f"**{title_worst} по _{chosen_metric}_**")
        worst5 = sorted_data.tail(top_n)
        worst5 = worst5.iloc[::-1].copy()
        st.dataframe(worst5[display_cols], use_container_width=True, column_config=col_cfg)
    st.caption("Используйте долю и отклонение, чтобы понять вклад филиала и его дистанцию от среднего уровня.")

    insight_lines = []
    if not sorted_data.empty:
        top_row = sorted_data.iloc[0]
        best_name = f"{top_row['Подразделение']} ({top_row['Регион']})"
        best_val = _format_value_for_metric(chosen_metric, top_row[chosen_metric])
        extra = ""
        if "Доля, %" in sorted_data.columns:
            extra = f" — доля {top_row['Доля, %']:.1f}%"
        insight_lines.append(f"Лидирует {best_name}: {best_val}{extra}.")
        bottom_row = sorted_data.iloc[-1]
        if bottom_row.name != top_row.name:
            worst_name = f"{bottom_row['Подразделение']} ({bottom_row['Регион']})"
            worst_val = _format_value_for_metric(chosen_metric, bottom_row[chosen_metric])
            extra_w = ""
            if "Доля, %" in sorted_data.columns:
                extra_w = f" — доля {bottom_row['Доля, %']:.1f}%"
            insight_lines.append(f"Наименьшее значение у {worst_name}: {worst_val}{extra_w}.")
    _render_insights("Что важно в рейтинге", insight_lines)
    series_for_actions = sorted_data.set_index(["Подразделение","Регион"])[chosen_metric]
    action_lines = _generate_actions_for_series(series_for_actions, chosen_metric)
    _render_plan("План по филиалам", action_lines[:4])


def comparison_block(
    df_all: pd.DataFrame,
    regions: list[str],
    available_months: list[str],
    *,
    default_metric: str | None = None,
    selection_key: str = "comparison_metric",
    period_a_key: str = "comparison_period_a",
    period_b_key: str = "comparison_period_b"
) -> None:
    st.subheader("⚖️ Сравнение периодов")
    st.caption("Выберите два диапазона месяцев: период A — база, период B — сравнение. Таблица покажет изменения по каждому филиалу.")
    if df_all.empty or not available_months: st.info("Нет данных для сравнения."); return
    c1, c2 = st.columns(2)
    with c1:
        start_a, end_a = st.select_slider(
            "Период A (базовый):",
            options=available_months,
            value=(available_months[0], available_months[0]),
            key=period_a_key
        )
    with c2:
        start_b, end_b = st.select_slider(
            "Период B (сравниваемый):",
            options=available_months,
            value=(available_months[-1], available_months[-1]),
            key=period_b_key
        )
    months_a = ORDER[ORDER.index(start_a): ORDER.index(end_a)+1]
    months_b = ORDER[ORDER.index(start_b): ORDER.index(end_b)+1]
    data_a = get_aggregated_data(df_all, tuple(regions), tuple(months_a))
    data_b = get_aggregated_data(df_all, tuple(regions), tuple(months_b))
    if data_a.empty or data_b.empty: st.warning("Нет данных для одного или обоих периодов."); return
    comparison_df = pd.merge(data_a, data_b, on=["Регион","Подразделение"], how="outer", suffixes=("_A","_B"))

    raw_metric_names = set(df_all["Показатель"].dropna().unique())
    all_metrics = sorted([c for c in data_a.columns if pd.api.types.is_numeric_dtype(data_a[c]) and c != "Код"])
    metric_options = [m for m in all_metrics if m in raw_metric_names and m not in HIDDEN_METRICS]
    if not metric_options:
        st.warning("Нет метрик из файла для сравнения.")
        return

    default_idx = 0
    if default_metric and default_metric in metric_options:
        default_idx = metric_options.index(default_metric)
    chosen_metric = st.selectbox(
        "Показатель для анализа:",
        options=metric_options,
        index=default_idx,
        key=selection_key,
        help=METRIC_HELP.get(metric_options[default_idx], "")
    )
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
    st.dataframe(
        comparison_df[["Подразделение","Регион",col_a,col_b,"Абсолютное изменение","Относительное изменение, %"]]
        .sort_values("Абсолютное изменение", ascending=False)
        .dropna(subset=["Абсолютное изменение"]),
        use_container_width=True,
        column_config=cfg
    )
    st.caption("Положительное изменение говорит о росте относительно базы, отрицательное — о просадке. Ориентируйтесь на столбец с относительным изменением, чтобы оценить масштаб." )

    insight_lines = []
    delta_series = comparison_df.dropna(subset=["Абсолютное изменение"]).set_index(["Подразделение","Регион"])["Абсолютное изменение"]
    if not delta_series.empty:
        inc = delta_series.idxmax()
        dec = delta_series.idxmin()
        inc_val = delta_series.loc[inc]
        dec_val = delta_series.loc[dec]
        if inc_val != 0:
            inc_name = f"{inc[0]} ({inc[1]})"
            insight_lines.append(f"Наибольший рост по {chosen_metric}: {inc_name} ({_format_delta(chosen_metric, inc_val)}).")
        if dec_val != 0 and dec != inc:
            dec_name = f"{dec[0]} ({dec[1]})"
            insight_lines.append(f"Сильнее всего просел {dec_name}: {_format_delta(chosen_metric, dec_val)}." )
    _render_insights("Итоги сравнения", insight_lines)
    delta_pairs = [(_label_from_index(idx), val) for idx, val in delta_series.items() if not pd.isna(val) and val != 0]
    action_lines = _generate_actions_for_deltas(delta_pairs, chosen_metric)
    _render_plan("Корректирующие действия", action_lines[:4])

def dynamics_block(
    df_all: pd.DataFrame,
    regions: list[str],
    months_range: list[str],
    color_map: Dict[str, str],
    *,
    default_metrics: List[str] | None = None,
    widget_prefix: str = "dyn"
) -> None:
    st.subheader("📈 Динамика по регионам")

    raw_metric_names = sorted(set(df_all["Показатель"].dropna().unique()))
    if not raw_metric_names:
        st.warning("В файлах не найдено метрик для построения динамики.")
        return

    base_defaults = [Metrics.REVENUE.value, Metrics.LOAN_ISSUE.value, Metrics.MARKUP_PCT.value]
    if default_metrics:
        default_selection = [m for m in default_metrics if m in raw_metric_names]
    else:
        default_selection = [m for m in base_defaults if m in raw_metric_names]
    if not default_selection:
        default_selection = raw_metric_names[:3]

    metrics = st.multiselect(
        "Показатели",
        options=raw_metric_names,
        default=default_selection[:3],
        key=f"{widget_prefix}_metrics"
    )

    if not metrics:
        st.info("Выберите метрики.");
        return

    c1, c2, c3, c4 = st.columns(4)
    only_actual = c1.checkbox("Только фактические месяцы", True, key=f"{widget_prefix}_actual")
    show_trend = c2.checkbox("Линия тренда", True, key=f"{widget_prefix}_trend")
    fast_plot = c3.checkbox("Облегчить отрисовку", False, key=f"{widget_prefix}_fast")
    use_log = c4.checkbox("Лог. ось Y", False, key=f"{widget_prefix}_log")

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
        deltas: List[tuple[str, float]] = []
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
            clean_series = series.dropna()
            if len(clean_series) >= 2:
                delta_val = float(clean_series.iloc[-1] - clean_series.iloc[0])
                deltas.append((str(reg), delta_val))
        if not any_drawn:
            st.info(f"Для «{met}» данные есть, но после выравнивания по календарю все серии пустые (разные месяцы у источников). Выберите «Только фактические месяцы» или сузьте период.")
            continue

        rule = aggregation_rule(met)
        rule_text = 'Сумма' if rule=='sum' else 'Среднее' if rule=='mean' else 'Последний месяц'
        subtitle = f"Источник: строки «Итого по месяцу». Агрегация за период: {rule_text}."
        fig.update_layout(title={'text': f"{met}<br><sup>{subtitle}</sup>", 'x':0}, hovermode="x unified", margin=dict(t=70,l=0,r=0,b=0))
        fig.update_yaxes(tickformat=tickfmt, ticksuffix=suf.strip(), title_text=suf.strip() or None)
        fig.update_yaxes(type="log" if use_log else "linear")

        st.plotly_chart(fig, use_container_width=True)
        insight = _describe_deltas(deltas, met)
        if insight:
            _render_insights(f"Выводы по {met}", [insight])
        with st.expander(f"Данные для графика «{met}»"):
            st.dataframe(
                gp.pivot_table(index="Месяц", columns="Регион", values="Значение", aggfunc="sum", observed=False)
                  .reindex(x_domain),
                use_container_width=True
            )
        action_lines = _generate_actions_for_deltas(deltas, met)
        _render_plan(f"Действия по {met}", action_lines[:3])


def dynamics_compare_block(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    regions: list[str],
    months_range: list[str],
    color_map: Dict[str, str],
    year_a: int,
    year_b: int,
    *,
    default_metrics: List[str] | None = None,
    widget_prefix: str = "dyncmp"
) -> None:
    st.subheader(f"📈 Динамика по регионам: {year_b} vs {year_a}")

    raw_metric_names = sorted(set(pd.concat([df_a, df_b])["Показатель"].dropna().unique()))

    base_defaults = [Metrics.REVENUE.value, Metrics.LOAN_ISSUE.value, Metrics.MARKUP_PCT.value]
    if default_metrics:
        default_selection = [m for m in default_metrics if m in raw_metric_names]
    else:
        default_selection = [m for m in base_defaults if m in raw_metric_names]
    if not default_selection:
        default_selection = raw_metric_names[:3]

    metrics = st.multiselect(
        "Показатели",
        options=raw_metric_names,
        default=default_selection,
        key=f"{widget_prefix}_metrics"
    )
    c1, c2, c3 = st.columns(3)
    only_actual = c1.checkbox("Только фактические месяцы", True, key=f"{widget_prefix}_actual")
    show_trend = c2.checkbox("Линия тренда", False, key=f"{widget_prefix}_trend")
    use_log = c3.checkbox("Лог. ось Y", False, key=f"{widget_prefix}_log")
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

        delta_records: List[tuple[str, float]] = []
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
                clean_s = s.dropna()
                if len(clean_s) >= 2:
                    delta_records.append((f"{reg} · {y}", float(clean_s.iloc[-1] - clean_s.iloc[0])))
        subtitle = f"Источник: строки «Итого по месяцу». Агрегация за период: {rule_text}."
        fig.update_layout(title={'text': f"{met}<br><sup>{subtitle}</sup>", 'x': 0},
                          hovermode="x unified", margin=dict(t=70, l=0, r=0, b=0))
        fig.update_yaxes(tickformat=tickfmt, ticksuffix=suf.strip(), title_text=suf.strip() or None)
        fig.update_yaxes(type="log" if use_log else "linear")
        st.plotly_chart(fig, use_container_width=True)
        insight = _describe_deltas(delta_records, met)
        if insight:
            _render_insights(f"Выводы по {met}", [insight])

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


def treemap_heatmap_block(
    df_all: pd.DataFrame,
    regions: list[str],
    months_range: list[str],
    color_map: Dict[str, str],
    *,
    default_metric: str | None = None,
    metric_key: str = "treemap_metric",
    month_key: str = "treemap_month",
    mode_key: str = "treemap_mode",
    heat_metric_key: str = "heatmap_metric"
) -> None:
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
    default_idx = 0
    default_candidate = default_metric or (Metrics.REVENUE.value if Metrics.REVENUE.value in raw_metric_names else None)
    if default_candidate and default_candidate in raw_metric_names:
        default_idx = raw_metric_names.index(default_candidate)
    metric = st.selectbox(
        "Метрика для площади",
        options=raw_metric_names,
        index=default_idx,
        key=metric_key,
        help=METRIC_HELP.get(default_candidate, "")
    )

    months_present = sorted_months_safe(sub["Месяц"])
    if not months_present:
        st.info("Нет данных за период."); return
    month_for_tree = st.selectbox("Месяц для структуры", options=months_present, index=len(months_present)-1, key=month_key)

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
        insight_lines = []
        region_series = tree_data.groupby("Регион")[["Size"]].sum()["Size"]
        insight = _describe_metric_series(region_series, metric)
        if insight:
            insight_lines.append(insight)
        top_branch = tree_data.sort_values("Size", ascending=False).iloc[0]
        top_name = f"{top_branch['Подразделение']} ({top_branch['Регион']})"
        insight_lines.append(f"Крупнейший вклад даёт {top_name}: {_format_value_for_metric(metric, top_branch['Size'])}.")
        _render_insights("Что видно по структуре", insight_lines)
        action_lines = _generate_actions_for_series(region_series, metric)
        _render_plan("Что сделать со структурой", action_lines[:3])
    else:
        st.info("Недостаточно данных для структуры.")

    st.divider()

    # ==== Теплокарта ====
    st.markdown("**Распределение по месяцам (тепловая карта)**")
    st.caption("Используем значения из строк «Итого по месяцу». Для метрик-снимков (задолженность, кол-во ломбардов) ничего не суммируем — показываем месячный снимок.")

    heat_default_idx = raw_metric_names.index(metric) if metric in raw_metric_names else 0
    heat_metric = st.selectbox(
        "Метрика для теплокарты",
        options=raw_metric_names,
        index=heat_default_idx,
        key=heat_metric_key,
        help=METRIC_HELP.get(metric, "")
    )
    by_subdiv = st.checkbox(
        "Показывать по подразделениям (в строках)",
        value=False,
        key=mode_key,
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

        hm = df_loc.pivot_table(index="RowLabel", columns="Месяц", values="Значение", aggfunc=aggfunc, observed=False)
    else:
        # региональный уровень: уже берём ровно то, что в «Итого по месяцу»
        mat = month_totals_matrix(df_all, tuple(regions), heat_metric)
        hm = mat.pivot_table(index="Регион", columns="Месяц", values="Значение", aggfunc="first", observed=False)

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
        insight_lines = []
        stack_vals = hm.stack().dropna()
        if not stack_vals.empty:
            if heat_metric in METRICS_SMALLER_IS_BETTER:
                target = stack_vals.idxmin()
            else:
                target = stack_vals.idxmax()
            row_label, month_label = target
            insight_lines.append(
                f"Пиковое значение: {row_label} / {month_label} ({_format_value_for_metric(heat_metric, stack_vals.loc[target])})."
            )
        row_means = hm.mean(axis=1, numeric_only=True)
        if not row_means.dropna().empty:
            insight = _describe_metric_series(row_means, heat_metric)
            if insight:
                insight_lines.append(insight)
        _render_insights("Выводы по теплокарте", insight_lines)
        action_lines = _generate_actions_for_series(row_means, heat_metric)
        _render_plan("Действия по теплокарте", action_lines[:3])
    else:
        st.info("Недостаточно данных для теплокарты.")




def monthly_totals_table(df_raw: pd.DataFrame, regions: list[str], months_range: list[str], all_available_months: list[str]) -> pd.DataFrame:
    if df_raw.empty or not months_range:
        return pd.DataFrame()

    filtered = df_raw[
        df_raw["Регион"].isin(regions) &
        df_raw["Подразделение"].str.contains(r"^\s*итого\b", case=False, na=False)
    ].copy()
    if filtered.empty:
        filtered = df_raw[df_raw["Регион"].isin(regions)].copy()

    all_metrics = sorted(filtered["Показатель"].dropna().unique().tolist())
    rows = []

    for metric in all_metrics:
        dfm = get_monthly_totals_from_file(df_raw, tuple(regions), metric)
        if dfm.empty:
            fallback = df_raw[
                df_raw["Регион"].isin(regions) &
                (df_raw["Показатель"] == metric) &
                (df_raw["Месяц"].astype(str) != "Итого")
            ]
            dfm = fallback.groupby("Месяц", observed=True)["Значение"].sum().reset_index()
        else:
            dfm = dfm.groupby("Месяц", observed=True)["Значение"].sum().reset_index()
        row = {"Показатель": metric}
        rule = aggregation_rule(metric)

        for m in months_range:
            mask = dfm["Месяц"].astype(str) == m
            vals = pd.to_numeric(dfm.loc[mask, "Значение"], errors="coerce")
            if vals.empty:
                row[m] = np.nan
            else:
                if rule == "mean":
                    row[m] = float(vals.mean())
                else:
                    row[m] = float(vals.sum())

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

    priority_map = {
        Metrics.DEBT_NO_SALE.value: 0,
        Metrics.DEBT.value: 1,
        Metrics.DEBT_UNITS.value: 2,
    }

    def row_order(s):
        return s.map(lambda name: priority_map.get(name, 3 if "руб" in name or "шт" in name else 4))

    return dfw.sort_values(by="Показатель", key=row_order).reset_index(drop=True)

def provided_totals_from_files(df_all: pd.DataFrame, regions: list[str], months_range: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_tot = df_all[
        (df_all["Регион"].isin(regions)) &
        df_all["Подразделение"].str.contains(r"^\s*итого\b", case=False, na=False) &
        (df_all["Месяц"].astype(str).isin(months_range + ["Итого"]))
    ].copy()
    if df_tot.empty:
        return pd.DataFrame(), pd.DataFrame()

    priority_map = {"RECALC_TOTAL": 0, "TOTALS_FILE": 1}
    src = df_tot.get("ИсточникФайла", pd.Series(index=df_tot.index, dtype=object))
    df_tot["__prio__"] = src.map(priority_map).fillna(2).astype(int)
    df_tot.sort_values(["Показатель","Месяц","__prio__"], inplace=True)
    best = df_tot.groupby(["Показатель","Месяц"], observed=True).first().reset_index()

    totals_row = best.pivot_table(index="Показатель", columns="Месяц", values="Значение", aggfunc="first", observed=False).reset_index()
    cols_ordered = ["Показатель"] + [m for m in months_range if m in totals_row.columns] + (["Итого"] if "Итого" in totals_row.columns else [])
    totals_row = totals_row.reindex(columns=cols_ordered)

    totals_col = pd.DataFrame()
    if "Итого" in best["Месяц"].astype(str).unique():
        it_col = best[best["Месяц"].astype(str) == "Итого"][["Показатель", "Значение"]].rename(columns={"Значение": "Итого"})
        totals_col = it_col.groupby("Показатель")["Итого"].first().reset_index()

    return totals_row, totals_col


def month_check_block(df_all: pd.DataFrame, regions: list[str], months_range: list[str], all_available_months: list[str]):
    st.subheader("📅 Сводная таблица по месяцам")
    st.caption("Итоговые значения по всем выбранным регионам, рассчитанные по согласованной методике.")
    if df_all.empty or not months_range: st.info("Нет данных."); return
    table = monthly_totals_table(df_all, regions, months_range, all_available_months)
    if table.empty: st.info("Нет данных для таблицы."); return
    colcfg = {c: st.column_config.NumberColumn(c, format="%.0f") for c in table.columns if c != "Показатель"}
    colcfg["Показатель"] = st.column_config.TextColumn("Показатель")
    st.dataframe(table, use_container_width=True, column_config=colcfg)

def export_block(df_long: pd.DataFrame):
    st.subheader("📥 Экспорт данных"); st.caption("Длинный формат: Регион · Год · Подразделение · Показатель · Месяц · Значение.")
    csv_bytes = df_long.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Скачать объединённый датасет (CSV)", data=csv_bytes, file_name="NUZ_combined_Long.csv", mime="text/csv")

def info_block():
    st.header("ℹ️ Справка по методике")
    st.markdown("""
### Ключ расчёта
- **В упрощенном режиме все показатели берутся строго из файла.**
- **В анализ включаем только строки с категорией НЮЗ (ЮЗ и «Общее» отбрасываем при загрузке).**
- **Показатели распознаём по фиксированному словарю НЮЗ-метрик; строки вне словаря пропускаем.**
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
    sidebar = st.sidebar.container()
    with sidebar:
        st.markdown("<p class='sidebar-title'>Файлы</p>", unsafe_allow_html=True)
        region_prefix = st.text_input(
            "Префикс региона",
            value="",
            placeholder="Префикс региона",
            label_visibility="collapsed"
        )
        uploads = st.file_uploader(
            "Файлы Excel",
            type=["xlsx", "xls"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        st.caption("Перетащите один или несколько .xlsx/.xls." )
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
    df_all = append_risk_share_metric(df_all)

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

    scenario_options = list(SCENARIO_CONFIGS.keys())
    with sidebar:
        st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
        st.markdown("<p class='sidebar-title'>Настройки анализа</p>", unsafe_allow_html=True)
        scenario_name = st.selectbox(
            "Сценарий",
            options=scenario_options,
            index=0,
            label_visibility="collapsed"
        )
        st.caption("Определяет, какие метрики и выводы будут приоритетными.")
        mode_year = st.radio(
            "Режим",
            options=["Один год", "Сравнение годов"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key="analysis_mode"
        )
        st.caption("Сравнение годов позволит увидеть динамику год-к-году.")

    if mode_year == "Один год":
        with sidebar:
            st.markdown("<p class='sidebar-title'>Год</p>", unsafe_allow_html=True)
            year_current = st.selectbox(
                "Год",
                options=years_all,
                index=len(years_all) - 1,
                label_visibility="collapsed",
                key="single_year_select"
            )
        df_current = df_all[df_all["Год"] == year_current].copy()
        months_in_data = sorted_months_safe(df_current["Месяц"])
        if not months_in_data:
            st.error("В выбранном году нет данных с распознанными месяцами."); st.stop()
        df_previous = None
        year_previous = None
    else:
        with sidebar:
            st.markdown("<p class='sidebar-title'>Годы сравнения</p>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            year_previous = col_a.selectbox("Год A", options=years_all, index=max(0, len(years_all) - 2), key="year_a", label_visibility="collapsed")
            year_current = col_b.selectbox("Год B", options=years_all, index=len(years_all) - 1, key="year_b", label_visibility="collapsed")
        df_previous = df_all[df_all["Год"] == year_previous].copy()
        df_current = df_all[df_all["Год"] == year_current].copy()
        months_a = sorted_months_safe(df_previous["Месяц"])
        months_b = sorted_months_safe(df_current["Месяц"])
        months_in_data = [m for m in ORDER if m in months_a and m in months_b]
        if not months_in_data:
            st.error("В пересечении выбранных годов нет данных по месяцу."); st.stop()

    with sidebar:
        st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
        st.markdown("<p class='sidebar-title'>Быстрый период</p>", unsafe_allow_html=True)
    if "global_period" not in st.session_state:
        st.session_state["global_period"] = (months_in_data[0], months_in_data[-1])

    preset = st.radio(
        "Быстрый выбор",
        options=["Весь период", "Квартал", "2 мес.", "Текущий мес."],
        index=0,
        horizontal=True,
        key="period_preset",
        label_visibility="collapsed"
    )
    if preset == "Весь период":
        st.session_state["global_period"] = (months_in_data[0], months_in_data[-1])
    elif preset == "Квартал":
        rng = months_in_data[-3:] if len(months_in_data) >= 3 else months_in_data
        st.session_state["global_period"] = (rng[0], rng[-1])
    elif preset == "2 мес.":
        rng = months_in_data[-2:] if len(months_in_data) >= 2 else months_in_data
        st.session_state["global_period"] = (rng[0], rng[-1])
    else:
        last_month = months_in_data[-1]
        st.session_state["global_period"] = (last_month, last_month)

    start_default, end_default = st.session_state.get("global_period", (months_in_data[0], months_in_data[-1]))
    if start_default not in months_in_data or end_default not in months_in_data:
        start_default, end_default = months_in_data[0], months_in_data[-1]

    with sidebar:
        start_m, end_m = st.select_slider(
            "Период",
            options=months_in_data,
            value=(start_default, end_default),
            key="period_slider",
            label_visibility="collapsed"
        )
        st.caption(f"Период: {start_m} – {end_m}")
    months_range = ORDER[ORDER.index(start_m): ORDER.index(end_m) + 1]

    with sidebar:
        st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
        st.markdown("<p class='sidebar-title'>Пороговые значения</p>", unsafe_allow_html=True)
        thresholds_state = st.session_state.get("thresholds_config", {"min_markup": 45.0, "max_risk": 25.0, "loss_cap": 5.0})
        min_markup_threshold = st.number_input(
            "Мин. наценка, %",
            min_value=0.0,
            max_value=200.0,
            value=float(thresholds_state.get("min_markup", 45.0)),
            step=1.0,
            key="threshold_min_markup"
        )
        max_risk_threshold = st.number_input(
            "Макс. риск, %",
            min_value=0.0,
            max_value=100.0,
            value=float(thresholds_state.get("max_risk", 25.0)),
            step=1.0,
            key="threshold_max_risk"
        )
        loss_cap_threshold = st.number_input(
            "Лимит убытка, млн ₽",
            min_value=0.0,
            max_value=500.0,
            value=float(thresholds_state.get("loss_cap", 5.0)),
            step=0.5,
            key="threshold_loss_cap"
        )

    thresholds_config = {
        "min_markup": float(min_markup_threshold),
        "max_risk": float(max_risk_threshold),
        "loss_cap": float(loss_cap_threshold),
    }
    st.session_state["thresholds_config"] = thresholds_config

    with sidebar:
        st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
        st.markdown("<p class='sidebar-title'>Действия</p>", unsafe_allow_html=True)
        col_reset, col_restart = st.columns(2)
        reset_trigger = col_reset.button("Сброс", use_container_width=True, key="btn_reset")
        restart_trigger = col_restart.button("Новый старт", use_container_width=True, key="btn_restart")

    if reset_trigger:
        st.session_state.clear()
        st.rerun()

    if restart_trigger:
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

    tab_specs = [
        ("📊", "Главная", render_home_page),
        ("🚀", "Выдачи", render_issuance_page),
        ("💰", "Проценты", render_interest_page),
        ("🛍️", "Распределение", render_sales_page),
        ("🧭", "Сравнения", render_comparison_page),
        ("👥", "Когорты", render_cohort_page),
        ("🧪", "Сценарии+", render_market_lab_page),
        ("🔮", "Прогноз", render_forecast_page),
        ("⚠️", "Риски", render_risk_page),
        ("📁", "Данные", render_data_page),
        ("🤖", "AI", render_ai_page),
    ]

    if mode_year == "Один год":
        regions_all = sorted(map(str, df_current["Регион"].unique()))
        with sidebar:
            st.markdown("<p class='sidebar-title'>Регионы</p>", unsafe_allow_html=True)
            pending_key = "single_regions_pending"
            if pending_key in st.session_state:
                st.session_state["single_regions"] = st.session_state.pop(pending_key)
            if "single_regions" not in st.session_state:
                st.session_state["single_regions"] = regions_all
            regions = st.multiselect(
                "Регионы",
                options=regions_all,
                default=regions_all,
                label_visibility="collapsed",
                placeholder="Выберите регионы",
                key="single_regions"
            )
            st.caption(f"Используем {len(regions)} из {len(regions_all)} регионов.")
            preset_cols = st.columns(3)
            top_revenue_regions = _top_regions_by_metric(df_current, regions_all, months_range, Metrics.REVENUE.value, top_n=5)
            high_risk_regions = _top_regions_by_metric(df_current, regions_all, months_range, Metrics.RISK_SHARE.value, top_n=5)
            branch_map = period_values_by_region_from_itogo(df_current, regions_all, Metrics.BRANCH_NEW_COUNT.value, months_range)
            new_branch_regions = [reg for reg, val in sorted(branch_map.items(), key=lambda kv: kv[1] if kv[1] is not None else 0, reverse=True) if val and not pd.isna(val) and val > 0][:5] if branch_map else []
            def _queue_single_regions(values: list[str]) -> None:
                st.session_state[pending_key] = values

            preset_cols[0].button(
                "ТОП-5 выручка",
                use_container_width=True,
                key="single_preset_revenue",
                disabled=not top_revenue_regions,
                on_click=_queue_single_regions,
                args=(top_revenue_regions,),
            )
            preset_cols[1].button(
                "Высокий риск",
                use_container_width=True,
                key="single_preset_risk",
                disabled=not high_risk_regions,
                on_click=_queue_single_regions,
                args=(high_risk_regions,),
            )
            preset_cols[2].button(
                "Новые филиалы",
                use_container_width=True,
                key="single_preset_new",
                disabled=not new_branch_regions,
                on_click=_queue_single_regions,
                args=(new_branch_regions,),
            )
        if not regions:
            st.warning("Выберите хотя бы один регион для анализа.")
            st.stop()

        color_map = consistent_color_map(tuple(regions_all))
        agg_current = get_aggregated_data(df_current, tuple(regions), tuple(months_range))

        meta = f"{len(regions)} из {len(regions_all)} регионов • Период: {months_range[0]} – {months_range[-1]}"
        st.markdown(
            f"""
            <div class="hero">
                <span class="hero-pill">{scenario_name}</span>
                <div class="hero__title">Анализ {year_current}</div>
                <div class="hero__meta">{meta}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        stats_cols = st.columns(4)
        stats_cols[0].metric("Файлов", len(uploads))
        stats_cols[1].metric("Регионов", len(regions_all))
        stats_cols[2].metric("Подразделений", strip_totals_rows(df_current)["Подразделение"].nunique())
        stats_cols[3].metric("Период данных", f"{months_in_data[0]} – {months_in_data[-1]}")
        st.divider()

        ctx = PageContext(
            mode="single",
            df_current=df_current,
            df_previous=None,
            agg_current=agg_current,
            regions=regions,
            months_range=months_range,
            months_available=months_in_data,
            scenario_name=scenario_name,
            year_current=year_current,
            year_previous=None,
            color_map=color_map,
            strict_mode=strict_mode,
            thresholds=thresholds_config,
        )

        tabs = st.tabs([f"{icon} {title}" for icon, title, _ in tab_specs])
        for tab, (_, _, renderer) in zip(tabs, tab_specs):
            with tab:
                renderer(ctx)
    else:
        combined = pd.concat([df_previous, df_current], ignore_index=True)
        regions_all = sorted(map(str, combined["Регион"].unique()))
        with sidebar:
            st.markdown("<p class='sidebar-title'>Регионы</p>", unsafe_allow_html=True)
            pending_key = "compare_regions_pending"
            if pending_key in st.session_state:
                st.session_state["compare_regions"] = st.session_state.pop(pending_key)
            if "compare_regions" not in st.session_state:
                st.session_state["compare_regions"] = regions_all
            regions = st.multiselect(
                "Регионы",
                options=regions_all,
                default=regions_all,
                label_visibility="collapsed",
                placeholder="Выберите регионы",
                key="compare_regions"
            )
            st.caption(f"Используем {len(regions)} из {len(regions_all)} регионов.")
            preset_cols = st.columns(3)
            top_revenue_regions = _top_regions_by_metric(df_current, regions_all, months_range, Metrics.REVENUE.value, top_n=5)
            high_risk_regions = _top_regions_by_metric(df_current, regions_all, months_range, Metrics.RISK_SHARE.value, top_n=5)
            branch_map_cmp = period_values_by_region_from_itogo(df_current, regions_all, Metrics.BRANCH_NEW_COUNT.value, months_range)
            new_branch_regions = [reg for reg, val in sorted(branch_map_cmp.items(), key=lambda kv: kv[1] if kv[1] is not None else 0, reverse=True) if val and not pd.isna(val) and val > 0][:5] if branch_map_cmp else []
            def _queue_compare_regions(values: list[str]) -> None:
                st.session_state[pending_key] = values

            preset_cols[0].button(
                "ТОП-5 выручка",
                use_container_width=True,
                key="compare_preset_revenue",
                disabled=not top_revenue_regions,
                on_click=_queue_compare_regions,
                args=(top_revenue_regions,),
            )
            preset_cols[1].button(
                "Высокий риск",
                use_container_width=True,
                key="compare_preset_risk",
                disabled=not high_risk_regions,
                on_click=_queue_compare_regions,
                args=(high_risk_regions,),
            )
            preset_cols[2].button(
                "Новые филиалы",
                use_container_width=True,
                key="compare_preset_new",
                disabled=not new_branch_regions,
                on_click=_queue_compare_regions,
                args=(new_branch_regions,),
            )
        if not regions:
            st.warning("Выберите хотя бы один регион для анализа.")
            st.stop()

        color_map = consistent_color_map(tuple(regions_all))
        agg_current = get_aggregated_data(df_current, tuple(regions), tuple(months_range))

        meta = f"{len(regions)} из {len(regions_all)} регионов • Период: {months_range[0]} – {months_range[-1]}"
        st.markdown(
            f"""
            <div class="hero">
                <span class="hero-pill">{scenario_name}</span>
                <div class="hero__title">Сравнение {year_current} vs {year_previous}</div>
                <div class="hero__meta">{meta}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        stats_cols = st.columns(4)
        stats_cols[0].metric("Файлов", len(uploads))
        stats_cols[1].metric("Регионов", len(regions_all))
        stats_cols[2].metric("Подразделений", strip_totals_rows(combined)["Подразделение"].nunique())
        stats_cols[3].metric("Период данных", f"{months_in_data[0]} – {months_in_data[-1]}")
        st.divider()

        ctx = PageContext(
            mode="compare",
            df_current=df_current,
            df_previous=df_previous,
            agg_current=agg_current,
            regions=regions,
            months_range=months_range,
            months_available=months_in_data,
            scenario_name=scenario_name,
            year_current=year_current,
            year_previous=year_previous,
            color_map=color_map,
            strict_mode=strict_mode,
            thresholds=thresholds_config,
        )

        tabs = st.tabs([f"{icon} {title}" for icon, title, _ in tab_specs])
        for tab, (_, _, renderer) in zip(tabs, tab_specs):
            with tab:
                renderer(ctx)


def render_home_page(ctx: PageContext) -> None:
    available_metrics = set(ctx.df_current["Показатель"].dropna().unique())
    missing_core = [m for m in KEY_DECISION_METRICS if m not in available_metrics]
    tab_hints = {
        name: [metric for metric in metrics if metric not in available_metrics]
        for name, metrics in TAB_METRIC_DEPENDENCIES.items()
    }
    if missing_core or any(tab_hints.values()):
        st.warning("В загруженных данных нет части ключевых показателей. Добавьте перечисленные метрики, чтобы раскрыть полный функционал.")
        if missing_core:
            st.markdown("**Минимальный набор:**")
            st.markdown("\n".join(f"- {metric}" for metric in missing_core))
        actionable = [(tab, metrics) for tab, metrics in tab_hints.items() if metrics]
        if actionable:
            st.markdown("**Для вкладок:**")
            st.markdown("\n".join(f"- {tab}: {', '.join(metrics)}" for tab, metrics in actionable))

    metrics_sequence = KEY_DECISION_METRICS + SUPPORT_DECISION_METRICS
    stats_current = {
        metric: compute_metric_stats(ctx.df_current, ctx.regions, ctx.months_range, metric)
        for metric in metrics_sequence
    }
    stats_previous = None
    if ctx.mode == "compare" and ctx.df_previous is not None:
        stats_previous = {
            metric: compute_metric_stats(ctx.df_previous, ctx.regions, ctx.months_range, metric)
            for metric in metrics_sequence
        }

    if ctx.mode == "single":
        scenario_overview_block_single(
            ctx.df_current,
            ctx.regions,
            ctx.months_range,
            ctx.scenario_name,
            ctx.year_current,
            stats_map=stats_current,
        )
    else:
        scenario_overview_block_compare(
            ctx.df_previous,
            ctx.df_current,
            ctx.regions,
            ctx.months_range,
            ctx.scenario_name,
            ctx.year_previous,
            ctx.year_current,
            stats_current=stats_current,
            stats_previous=stats_previous,
        )

    render_executive_summary(ctx, stats_current, stats_previous)
    st.divider()
    title_suffix = f" ({ctx.year_current})" if ctx.mode == "compare" else ""
    st.markdown(f"### 🔍 Основные KPI{title_suffix}")
    kpi_block(ctx.df_current, ctx.regions, ctx.months_range, ctx.months_available, ctx.strict_mode)
    st.divider()
    st.markdown("### 🌍 Картина по регионам")
    summary_block(ctx.agg_current, ctx.df_current, ctx.regions, ctx.months_range, ctx.months_available, ctx.strict_mode)
    st.divider()
    render_scenario_simulator(ctx)
    st.divider()
    render_margin_capacity_planner(ctx, widget_prefix="home_margin")
    st.divider()
    render_management_tools(ctx, stats_current, stats_previous)


def render_issuance_page(ctx: PageContext) -> None:
    suffix = "_cmp" if ctx.mode == "compare" else ""
    title = "### 🚀 Выдачи и портфель"
    if ctx.mode == "compare":
        title += f" ({ctx.year_current})"
    st.markdown(title)
    render_tab_summary(ctx, TAB_METRIC_SETS["issuance"], title="#### 🧭 Executive summary — выдачи")
    st.divider()
    _render_metric_trend_section(
        "Динамика выдач" if ctx.mode == "single" else "Динамика выдач (год B)",
        [
            Metrics.LOAN_ISSUE.value,
            Metrics.LOAN_ISSUE_UNITS.value,
            Metrics.AVG_LOAN.value
        ],
        ctx.df_current,
        ctx.regions,
        ctx.months_range,
        widget_key=f"issuance_trend{suffix}"
    )
    st.divider()
    leaderboard_block(
        ctx.df_current,
        ctx.regions,
        ctx.months_available,
        default_metric=Metrics.LOAN_ISSUE.value,
        selection_key=f"issuance_leader_metric{suffix}",
        period_slider_key=f"issuance_period{suffix}"
    )
    st.divider()
    comparison_block(
        ctx.df_current,
        ctx.regions,
        ctx.months_available,
        default_metric=Metrics.LOAN_ISSUE.value,
        selection_key=f"issuance_comparison_metric{suffix}",
        period_a_key=f"issuance_period_a{suffix}",
        period_b_key=f"issuance_period_b{suffix}"
    )
    st.divider()
    if ctx.mode == "compare":
        dynamics_compare_block(
            ctx.df_previous,
            ctx.df_current,
            ctx.regions,
            ctx.months_range,
            ctx.color_map,
            year_a=ctx.year_previous,
            year_b=ctx.year_current,
            default_metrics=[Metrics.LOAN_ISSUE.value, Metrics.LOAN_ISSUE_UNITS.value],
            widget_prefix="issuance_dyn_cmp"
        )
    else:
        dynamics_block(
            ctx.df_current,
            ctx.regions,
            ctx.months_range,
            ctx.color_map,
            default_metrics=[Metrics.LOAN_ISSUE.value, Metrics.LOAN_ISSUE_UNITS.value, Metrics.AVG_LOAN.value],
            widget_prefix="issuance_dyn"
        )


def render_interest_page(ctx: PageContext) -> None:
    suffix = "_cmp" if ctx.mode == "compare" else ""
    title = "### 💰 Проценты и погашения"
    if ctx.mode == "compare":
        title += f" ({ctx.year_current})"
    st.markdown(title)
    render_tab_summary(ctx, TAB_METRIC_SETS["interest"], title="#### 🧭 Executive summary — проценты")
    st.divider()
    _render_metric_trend_section(
        "Динамика процентных показателей" if ctx.mode == "single" else "Проценты и погашения (год B)",
        [
            Metrics.PENALTIES_RECEIVED.value,
            Metrics.YIELD.value,
            Metrics.LOAN_REPAYMENT_SUM.value
        ],
        ctx.df_current,
        ctx.regions,
        ctx.months_range,
        widget_key=f"interest_trend{suffix}"
    )
    st.divider()
    leaderboard_block(
        ctx.df_current,
        ctx.regions,
        ctx.months_available,
        default_metric=Metrics.PENALTIES_RECEIVED.value,
        selection_key=f"interest_leader_metric{suffix}",
        period_slider_key=f"interest_period{suffix}"
    )
    st.divider()
    comparison_block(
        ctx.df_current,
        ctx.regions,
        ctx.months_available,
        default_metric=Metrics.PENALTIES_RECEIVED.value,
        selection_key=f"interest_comparison_metric{suffix}",
        period_a_key=f"interest_period_a{suffix}",
        period_b_key=f"interest_period_b{suffix}"
    )
    st.divider()
    if ctx.mode == "compare":
        dynamics_compare_block(
            ctx.df_previous,
            ctx.df_current,
            ctx.regions,
            ctx.months_range,
            ctx.color_map,
            year_a=ctx.year_previous,
            year_b=ctx.year_current,
            default_metrics=[Metrics.PENALTIES_RECEIVED.value, Metrics.YIELD.value, Metrics.LOAN_REPAYMENT_SUM.value],
            widget_prefix="interest_dyn_cmp"
        )
    else:
        dynamics_block(
            ctx.df_current,
            ctx.regions,
            ctx.months_range,
            ctx.color_map,
            default_metrics=[Metrics.PENALTIES_RECEIVED.value, Metrics.YIELD.value, Metrics.LOAN_REPAYMENT_SUM.value],
            widget_prefix="interest_dyn"
        )


def render_sales_page(ctx: PageContext) -> None:
    suffix = "_cmp" if ctx.mode == "compare" else ""
    title = "### 🛍️ Распродажа и маржа"
    if ctx.mode == "compare":
        title = "### 🛍️ Распродажа" + f" ({ctx.year_current})"
    st.markdown(title)
    render_tab_summary(ctx, TAB_METRIC_SETS["sales"], title="#### 🧭 Executive summary — распродажа")
    st.divider()
    _render_metric_trend_section(
        "Динамика продаж" if ctx.mode == "single" else "Продажи и маржа (год B)",
        [
            Metrics.REVENUE.value,
            Metrics.MARKUP_PCT.value,
            Metrics.PENALTIES_PLUS_MARKUP.value
        ],
        ctx.df_current,
        ctx.regions,
        ctx.months_range,
        widget_key=f"sales_trend{suffix}"
    )
    st.divider()
    treemap_heatmap_block(
        ctx.df_current,
        ctx.regions,
        ctx.months_range,
        ctx.color_map,
        default_metric=Metrics.REVENUE.value,
        metric_key=f"sales_treemap_metric{suffix}",
        month_key=f"sales_treemap_month{suffix}",
        mode_key=f"sales_treemap_mode{suffix}",
        heat_metric_key=f"sales_heat_metric{suffix}"
    )
    st.divider()
    render_revenue_waterfall(ctx)
    st.divider()
    sales_intelligence_block(ctx, ctx.thresholds)
    st.divider()
    render_margin_capacity_planner(ctx, widget_prefix="sales_margin")
    st.divider()
    if ctx.mode == "compare":
        dynamics_compare_block(
            ctx.df_previous,
            ctx.df_current,
            ctx.regions,
            ctx.months_range,
            ctx.color_map,
            year_a=ctx.year_previous,
            year_b=ctx.year_current,
            default_metrics=[Metrics.REVENUE.value, Metrics.MARKUP_PCT.value, Metrics.PENALTIES_PLUS_MARKUP.value],
            widget_prefix="sales_dyn_cmp"
        )
    else:
        dynamics_block(
            ctx.df_current,
            ctx.regions,
            ctx.months_range,
            ctx.color_map,
            default_metrics=[Metrics.REVENUE.value, Metrics.MARKUP_PCT.value, Metrics.PENALTIES_PLUS_MARKUP.value],
            widget_prefix="sales_dyn"
        )


def render_risk_page(ctx: PageContext) -> None:
    suffix = "_cmp" if ctx.mode == "compare" else ""
    title = "### ⚠️ Риски и устойчивость" if ctx.mode == "single" else "### ⚠️ Риски (сравнение годов)"
    st.markdown(title)
    st.caption("Ключевой индикатор — «Доля ниже займа, %»: показывает, какая часть выручки от распродажи получена по товарам, проданным дешевле суммы займа. Рост означает усиление убыточных продаж.")
    render_tab_summary(ctx, TAB_METRIC_SETS["risk"], title="#### 🧭 Executive summary — риски")
    st.divider()
    alert_config = risk_alerts_block(ctx)
    st.divider()
    _render_metric_trend_section(
        "Динамика рисковых показателей" if ctx.mode == "single" else "Динамика рисков (год B)",
        [
            Metrics.RISK_SHARE.value,
            Metrics.ILLIQUID_BY_VALUE_PCT.value,
            Metrics.PLAN_ISSUE_PCT.value
        ],
        ctx.df_current,
        ctx.regions,
        ctx.months_range,
        widget_key=f"risk_trend{suffix}"
    )
    st.divider()
    leaderboard_block(
        ctx.df_current,
        ctx.regions,
        ctx.months_available,
        default_metric=Metrics.RISK_SHARE.value,
        selection_key=f"risk_leader_metric{suffix}",
        period_slider_key=f"risk_period{suffix}"
    )
    st.divider()
    risk_markup_heatmap_block(ctx)
    st.divider()
    render_risk_dependency(ctx)
    st.divider()
    risk_failure_forecast_block(ctx, alert_config.get("risk_threshold") if alert_config else None)
    st.divider()
    if ctx.mode == "compare":
        dynamics_compare_block(
            ctx.df_previous,
            ctx.df_current,
            ctx.regions,
            ctx.months_range,
            ctx.color_map,
            year_a=ctx.year_previous,
            year_b=ctx.year_current,
            default_metrics=[Metrics.RISK_SHARE.value, Metrics.ILLIQUID_BY_VALUE_PCT.value, Metrics.PLAN_REVENUE_PCT.value],
            widget_prefix="risk_dyn_cmp"
        )
    else:
        dynamics_block(
            ctx.df_current,
            ctx.regions,
            ctx.months_range,
            ctx.color_map,
            default_metrics=[Metrics.RISK_SHARE.value, Metrics.ILLIQUID_BY_VALUE_PCT.value, Metrics.PLAN_ISSUE_PCT.value],
            widget_prefix="risk_dyn"
        )


def render_data_page(ctx: PageContext) -> None:
    title = "### 📅 Валидация и данные" if ctx.mode == "single" else "### 📅 Валидация (год B)"
    st.markdown(title)
    render_health_check(ctx)
    st.divider()
    month_check_block(ctx.df_current, ctx.regions, ctx.months_range, ctx.months_available)
    st.divider()
    render_correlation_block(ctx.df_current, ctx.regions, ctx.months_range, default_metrics=FORECAST_METRICS)
    st.divider()
    if ctx.mode == "compare" and ctx.df_previous is not None:
        export_source = pd.concat([ctx.df_previous, ctx.df_current], ignore_index=True)
    else:
        export_source = ctx.df_current
    export_filtered = export_source[
        (export_source["Регион"].isin(ctx.regions))
        & (export_source["Месяц"].astype(str).isin(ctx.months_range))
    ]
    export_block(export_filtered)
    info_block()
    render_faq_block()


def render_ai_page(ctx: PageContext) -> None:
    if ctx.mode == "single":
        ai_analysis_block_single_year(ctx.df_current, ctx.regions, ctx.months_range, ctx.year_current)
    else:
        ai_analysis_block_comparison(ctx.df_previous, ctx.df_current, ctx.regions, ctx.months_range, ctx.year_previous, ctx.year_current)


if __name__ == "__main__":
    main()
