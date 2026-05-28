"""
AI Loan Decision Assistant — Dissertation MVP
Topic: AI-Based Credit Decision Assistant for Responsible Borrowing
in Digital Financial Services (Uzbekistan context)

Deploy:
  - app.py, data_fw.xlsx, requirements.txt in GitHub root
  - Streamlit Cloud → main file: app.py
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
try:
    import anthropic as _anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    import joblib as _joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

# ─── Page config — must be FIRST ─────────────────────────────
st.set_page_config(
    page_title="AI Loan Decision Assistant",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Constants ───────────────────────────────────────────────
APP_DIR    = Path(__file__).resolve().parent
DATA_FILE  = APP_DIR / "data_fw.xlsx"
MAIN_SHEET = "Final_Monthly_ML_Dataset"
MC_MODEL_FILE = APP_DIR / "loan_risk_model.pkl"

LANG_OPTIONS = {"Русский": "ru", "English": "en", "O'zbekcha": "uz"}

FEATURE_COLUMNS = [
    "policy_rate_pct","inflation_yoy_pct","cpi_mom_pct",
    "nominal_wage_monthly_approx","real_wage_growth_pct",
    "real_policy_rate_pct","debt_burden_indicator_pct",
]
REQUIRED_COLUMNS = [
    "date","policy_rate_pct","inflation_yoy_pct","cpi_mom_pct",
    "nominal_wage_monthly_approx","real_wage_growth_pct",
    "real_policy_rate_pct","debt_burden_indicator_pct","repayment_pressure_level",
]

# ══════════════════════════════════════════════════════════════
# TRANSLATIONS
# ══════════════════════════════════════════════════════════════
TR: dict[str, dict] = {

# ─── Russian (default) ───────────────────────────────────────
"ru": {
    # steps
    "s1":"Кредит", "s2":"Бюджет", "s3":"Ситуация", "s4":"Результат",
    "btn_back":"← Назад", "btn_next":"Далее →", "btn_calc":"Оценить кредит →",
    "btn_restart":"Начать заново",
    # welcome
    "hero_title":   "💳 AI Loan Decision Assistant",
    "hero_sub":     "Поможем оценить кредит перед тем, как его брать",
    "w_what":       "Что делает этот инструмент",
    "w_body":       "Вы вводите данные о кредите и своём бюджете. Инструмент считает, насколько кредит будет нагрузкой на ваш бюджет каждый месяц — и даёт понятный результат.",
    "w_not":        "Важно понимать:",
    "w_not1":       "Это не одобрение и не отказ в кредите",
    "w_not2":       "Это не финансовый совет",
    "w_not3":       "Это исследовательский инструмент для оценки нагрузки",
    "w_prepare":    "Что нужно знать перед заполнением:",
    "w_p1":         "Сумму кредита и процентную ставку",
    "w_p2":         "Свой ежемесячный доход",
    "w_p3":         "Ежемесячные расходы: обязательные и необязательные",
    "w_p4":         "День получения зарплаты",
    "btn_start":    "Начать оценку →",
    # step 1
    "s1_h":         "Шаг 1 из 3 — Параметры кредита",
    "f_amount":     "Сумма кредита",
    "f_amount_ph":  "Например: 10 000 000",
    "f_rate":       "Годовая процентная ставка (%)",
    "f_rate_ph":    "Например: 24",
    "f_term":       "Срок кредита",
    "f_start":      "Когда планируете взять кредит?",
    "yr":           "лет / г.",
    "term_months":  "месяцев",
    "est_pmt_lbl":  "Примерный ежемесячный платёж",
    "est_total_lbl":"Общая сумма к выплате",
    "v_amount":     "Введите сумму кредита",
    "v_rate":       "Введите процентную ставку",
    # step 2
    "s2_h":         "Шаг 2 из 3 — Ваш бюджет",
    "f_income":     "Ежемесячный доход (после налогов)",
    "f_income_ph":  "Например: 5 000 000",
    "f_ess":        "Обязательные расходы в месяц",
    "f_ess_help":   "Аренда, коммунальные, еда, транспорт — то, без чего нельзя",
    "f_ess_ph":     "Например: 2 000 000",
    "f_flex":       "Необязательные расходы в месяц",
    "f_flex_help":  "Кафе, подписки, шопинг — то, что можно сократить при необходимости",
    "f_flex_ph":    "Например: 500 000",
    "f_exist":      "Текущие выплаты по другим кредитам",
    "f_exist_help": "Если есть — укажите общую сумму в месяц",
    "f_exist_ph":   "Например: 300 000 (или 0)",
    "f_exist_mo":   "Сколько месяцев ещё платить по старым кредитам",
    "f_exist_mo_h": "Через сколько месяцев эти выплаты закончатся? Если нет — поставьте 0",
    "f_sav":        "Финансовый запас / сбережения",
    "f_sav_help":   "Сколько накоплено — на случай непредвиденного",
    "f_sav_ph":     "Например: 1 000 000 (или 0)",
    "f_sal_day":    "В какой день приходит зарплата?",
    "f_pay_day":    "В какой день нужно платить по кредиту?",
    "budget_title": "Как выглядит ваш бюджет",
    "b_income":     "Доход",
    "b_ess":        "Обязательные расходы",
    "b_flex":       "Необязательные расходы",
    "b_exist":      "Выплаты по старым кредитам",
    "b_new":        "Новый платёж по кредиту",
    "b_left":       "Остаток",
    "w_negative":   "⚠️ Судя по данным, этот платёж не вписывается в бюджет. Рассмотрите меньшую сумму или более длинный срок.",
    "w_tight":      "⚠️ После всех платежей остаётся очень мало. Рекомендуем иметь финансовый запас минимум на 1–2 месяца.",
    "v_income":     "Введите ваш доход",
    # step 3
    "s3_h":         "Шаг 3 из 3 — Ваша ситуация",
    "q_stable":     "Ваша зарплата стабильна?",
    "q_stable_y":   "Да, приходит вовремя",
    "q_stable_n":   "Иногда бывают задержки",
    "q_purpose":    "Для чего нужен кредит?",
    "q_purpose_opts":["Срочная необходимость (здоровье, жильё, учёба)",
                      "Бизнес или инвестиции",
                      "Техника, товары, мебель",
                      "Отдых или развлечения",
                      "Погашение другого кредита",
                      "Другое"],
    "q_terms":      "Вы понимаете все условия кредита: комиссии, штрафы, переплату?",
    "q_terms_y":    "Да, разобрался(-ась)",
    "q_terms_n":    "Не до конца",
    "q_delay":      "Если зарплата задержится на 3 дня — сможете ли сделать платёж?",
    "q_delay_y":    "Да, есть запас",
    "q_delay_n":    "Скорее нет",
    "q_refi":       "Этот кредит для погашения другого кредита?",
    "q_refi_y":     "Да",
    "q_refi_n":     "Нет",
    "q_sav_mo":     "На сколько месяцев платежей хватит сбережений?",
    "proj_note":    "Инструмент рассчитывает нагрузку на весь срок кредита, учитывая ожидаемый рост доходов и расходов.",
    "proj_how":     "Как прогнозировать ваш доход?",
    "proj_avg":     "По средней динамике зарплат",
    "proj_own":     "Укажу сам(а)",
    "proj_none":    "Без роста",
    "proj_pct":     "Ожидаемый рост дохода в год (%)",
    "proj_infl":    "Учитывать рост расходов вместе с инфляцией",
    "calc_note":    "Расчёт учитывает ваши данные и базовые экономические параметры (инфляцию, динамику зарплат). Результат — ориентировочный.",
    # result
    "s4_h":         "Результат оценки",
    "r_sub":        "На основе ваших данных",
    "r_lbl_pmt":    "Ежемесячный платёж",
    "r_lbl_pct":    "Доля от дохода",
    "r_lbl_left":   "Остаток после всех платежей",
    "r_lbl_first":  "Первый рискованный месяц",
    "r_na":         "нет",
    # result levels
    "r_low_h":      "✅ Кредит выглядит посильным",
    "r_low_b":      "По вашим данным нагрузка умеренная. Платёж занимает небольшую долю дохода, и после всех обязательных расходов должен оставаться запас.",
    "r_med_h":      "⚠️ Умеренная нагрузка — стоит подготовиться",
    "r_med_b":      "Кредит может быть управляемым, но в некоторые месяцы может быть туго. Рекомендуем проверить детали и иметь финансовый запас.",
    "r_high_h":     "🔶 Высокая нагрузка — рекомендуем пересмотреть условия",
    "r_high_b":     "Кредит будет занимать значительную часть бюджета. Стоит рассмотреть меньшую сумму, более длинный срок или подождать до улучшения финансовой ситуации.",
    "r_crit_h":     "🔴 Кредит может быть финансово рискованным",
    "r_crit_b":     "По вашим данным нагрузка очень высокая. Рекомендуем рассмотреть альтернативы: меньшую сумму, более длинный срок или отложить оформление. Это не финансовый совет — обратитесь к специалисту.",
    # distribution
    "dist_h":       "Как будет меняться нагрузка",
    "dist_low":     "Нормальных месяцев",
    "dist_med":     "Напряжённых месяцев",
    "dist_high":    "Тяжёлых месяцев",
    "dist_crit":    "Критических месяцев",
    # recommendations
    "rec_before_h": "До оформления кредита",
    "rec_reduce_h": "Как снизить нагрузку",
    "rec_timing_h": "Риск по дате платежа",
    "rec_careful_h":"Когда стоит быть особенно осторожным",
    # check items
    "chk_date":     "День платежа раньше дня зарплаты. Попросите банк перенести дату платежа на день после зарплаты — это уберёт ежемесячный риск.",
    "chk_terms":    "Вы не до конца знаете условия. Уточните у банка: общую сумму переплаты, все комиссии, штрафы за досрочное погашение.",
    "chk_total":    "Узнайте полную стоимость кредита — не только ежемесячный платёж, но и итоговую переплату.",
    "chk_buffer":   "Рекомендуем иметь финансовый запас минимум на 1–2 месяца платежей перед подписанием.",
    "chk_delay":    "Вы отметили, что при задержке зарплаты сделать платёж будет сложно. Создайте небольшой резерв — хотя бы один платёж.",
    # reduce
    "red_amount":   "Уменьшить сумму кредита — это снизит ежемесячный платёж.",
    "red_term":     "Увеличить срок — меньше платить каждый месяц (но больше в итоге).",
    "red_flex":     "Временно сократить необязательные расходы — кафе, подписки, шопинг.",
    "red_no_extra": "Не брать новые кредиты, пока этот не выплачен.",
    # timing
    "tim_date":     "День платежа ({p}) раньше дня зарплаты ({s}). Это создаёт ежемесячный риск — деньги могут ещё не поступить.",
    "tim_ok":       "День платежа ({p}) после зарплаты ({s}) — это хорошо. Риск по дате платежа минимальный.",
    # careful
    "car_salary":   "Зарплата может задерживаться. Держите запас минимум на 2 платежа на отдельном счёте.",
    "car_non_urg":  "Кредит берётся не на срочные нужды. При высокой нагрузке лучше подождать или взять меньше.",
    "car_refi":     "Новый кредит для погашения старого — высокий риск долговой ловушки. Сначала обсудите реструктуризацию с банком.",
    "car_low_inc":  "Ваш доход ниже среднего по стране. Платёж будет занимать значительную долю бюджета.",
    "car_macro":    "Экономические условия создают дополнительное давление на бюджет. Рекомендуем взять меньше или подождать.",
    "car_flex":     "Большая часть необязательных расходов. Если придётся их сократить, это освободит часть бюджета.",
    # expanders
    "exp_charts":   "Подробный прогноз по месяцам",
    "exp_table":    "Таблица симуляции",
    "exp_macro":    "Какие экономические параметры используются",
    "exp_method":   "Как считается результат",
    "ch1_title":    "Сколько дохода уходит на кредит — по месяцам",
    "ch2_title":    "Остаток после платежей — по месяцам",
    "ml_proba_title": "Уверенность модели (Random Forest) — вероятность уровня нагрузки по месяцам",
    "macro_note":   "Эти данные используются только внутри расчёта. Вам не нужно их анализировать — они влияют на прогноз нагрузки.",
    "download":     "Скачать таблицу (CSV)",
    "method_body":  """**Ежемесячный платёж** рассчитан по формуле аннуитета (равные платежи каждый месяц).

**Макроэкономический уровень нагрузки** определяется обученной моделью **Random Forest** (200 деревьев, class_weight=balanced). Модель обучена на исторических месячных данных Узбекистана (2010–2025, 192 наблюдения). Признаки: инфляция, ставка рефинансирования, CPI MoM, номинальная зарплата, реальный рост зарплат, реальная ставка, индикатор долговой нагрузки. Целевая переменная: Low / Medium / High.

По результатам анализа (диссертация): точность Random Forest на тестовой выборке — 92.3%, F1-macro — 0.651. Decision Tree: точность 79.5%, F1-macro 0.760. Логистическая регрессия показала наибольшую стабильность при временно́й кросс-валидации (mean F1 0.58).

**Нагрузка по месяцам** дополнительно оценивается исходя из:
- доли платежа от дохода (каждый месяц)
- остатка после всех расходов и платежей
- наличия финансового запаса
- даты платежа и даты зарплаты

**Уровни нагрузки (финальные):**
- ✅ Нормальная — платёж занимает до 25% дохода, запас есть
- ⚠️ Умеренная — 25–40%, иногда может быть напряжённо
- 🔶 Высокая — 40–50%, рекомендуем пересмотреть условия
- 🔴 Критическая — больше 50%, высокий финансовый риск

Инструмент не принимает решений за банк и не является финансовым советником.""",
    # pressure labels
    "lbl_low":"Нормальная","lbl_med":"Умеренная","lbl_high":"Высокая","lbl_crit":"Критическая",
    # table columns
    "tbl_date":"Дата","tbl_month":"Месяц","tbl_pmt":"Платёж","tbl_inc":"Доход",
    "tbl_ess":"Обяз.","tbl_flex":"Необяз.","tbl_pti":"PTI %","tbl_tdb":"TDB %",
    "tbl_cash":"Остаток","tbl_bal":"Баланс","tbl_lvl":"Уровень",
    # custom term
    "term_custom":"Свой срок (месяцев)","term_mode_pre":"Выбрать из списка","term_mode_own":"Ввести вручную",
    # validation
    "v_rate_high":"Ставка выше 100% — проверьте, нет ли ошибки.",
    "v_exp_high":"Сумма расходов превышает доход — проверьте введённые данные.",
    "v_savings_low":"Сбережений меньше одного платежа. Рекомендуем создать запас перед оформлением.",
    # AI features
    "ai_insight_btn":   "💡 Получить AI-инсайт",
    "ai_insight_title": "💡 Персональный AI-инсайт",
    "ai_insight_spin":  "Анализирую вашу ситуацию...",
    "ai_opt_btn":       "🔄 AI: предложить лучшие параметры",
    "ai_opt_title":     "🔄 Альтернативные сценарии от AI",
    "ai_opt_spin":      "Подбираю варианты...",
    "ai_chat_title":    "💬 Спросите AI-ассистента",
    "ai_chat_ph":       "Например: что будет если я потеряю работу на 2 месяца?",
    "ai_chat_btn":      "Отправить →",
    "ai_chat_spin":     "Думаю...",
    "ai_no_key":        "⚙️ Для AI-функций добавьте ANTHROPIC_API_KEY в Streamlit Secrets.",
    "ai_error":         "AI временно недоступен. Попробуйте позже.",
    "ai_chat_clear":    "Очистить чат",
    # Monte Carlo ML model
    "mc_default_lbl":   "Вероятность дефолта (ML)",
    "mc_default_sub":   "Monte Carlo · Random Forest",
    "mc_no_model":      "Файл loan_risk_model.pkl не найден. Запустите model_training.py локально.",
    # errors
    "err_dataset":  "Файл data_fw.xlsx не найден. Поместите его в ту же папку GitHub, что и app.py.",
    "err_forecast": "Не удалось построить прогноз на весь срок. Проверьте data_fw.xlsx.",
    "disclaimer":   "Исследовательский инструмент. Не является финансовым советом, кредитным скорингом или решением банка. Не одобряет и не отказывает в кредите.",
},

# ─── English ─────────────────────────────────────────────────
"en": {
    "s1":"Loan","s2":"Budget","s3":"Situation","s4":"Result",
    "btn_back":"← Back","btn_next":"Next →","btn_calc":"Estimate loan →",
    "btn_restart":"Start over",
    "hero_title":"💳 AI Loan Decision Assistant",
    "hero_sub":  "Estimate whether a loan will be affordable — before you apply",
    "w_what":    "What this tool does",
    "w_body":    "You enter your loan details and budget. The tool calculates how much pressure the loan puts on your monthly budget — and gives you a clear result.",
    "w_not":     "Important to understand:",
    "w_not1":    "This is not a loan approval or rejection",
    "w_not2":    "This is not financial advice",
    "w_not3":    "This is a research tool for estimating loan affordability",
    "w_prepare": "What you will need:",
    "w_p1":      "Loan amount and interest rate",
    "w_p2":      "Your monthly net income",
    "w_p3":      "Your essential and optional monthly expenses",
    "w_p4":      "Your salary day and expected payment due day",
    "btn_start": "Start estimation →",
    "s1_h":      "Step 1 of 3 — Loan details",
    "f_amount":  "Loan amount","f_amount_ph":"e.g. 10 000 000",
    "f_rate":    "Annual interest rate (%)","f_rate_ph":"e.g. 24",
    "f_term":    "Loan term","f_start":"When do you plan to take the loan?",
    "yr":        "year(s)","term_months":"months",
    "est_pmt_lbl":"Estimated monthly payment","est_total_lbl":"Total repayable amount",
    "v_amount":"Please enter the loan amount","v_rate":"Please enter the interest rate",
    "s2_h":      "Step 2 of 3 — Your budget",
    "f_income":  "Monthly net income (after taxes)","f_income_ph":"e.g. 5 000 000",
    "f_ess":     "Essential monthly expenses","f_ess_help":"Rent, utilities, food, transport — unavoidable costs",
    "f_ess_ph":  "e.g. 2 000 000",
    "f_flex":    "Optional monthly expenses","f_flex_help":"Dining, subscriptions, shopping — can be reduced if needed",
    "f_flex_ph": "e.g. 500 000",
    "f_exist":   "Existing loan payments / month","f_exist_help":"Current total monthly loan instalments",
    "f_exist_ph":"e.g. 300 000 (or 0)",
    "f_exist_mo":"Remaining months on existing loans",
    "f_exist_mo_h":"After this many months, existing loan payments stop. Enter 0 if none.",
    "f_sav":     "Savings / emergency buffer","f_sav_help":"Your current savings amount",
    "f_sav_ph":  "e.g. 1 000 000 (or 0)",
    "f_sal_day": "Salary day of month","f_pay_day":"Loan payment due day",
    "budget_title":"Your monthly budget preview",
    "b_income":"Income","b_ess":"Essential expenses","b_flex":"Optional expenses",
    "b_exist":"Existing loan payments","b_new":"New loan payment","b_left":"Money left",
    "w_negative":"⚠️ The new payment does not fit the current budget. Consider a smaller amount or longer term.",
    "w_tight":   "⚠️ Very little money left after all payments. Try to have at least 1–2 months of payments in savings.",
    "v_income":  "Please enter your income",
    "s3_h":      "Step 3 of 3 — Your situation",
    "q_stable":  "Is your salary stable?","q_stable_y":"Yes, arrives on time","q_stable_n":"Sometimes delayed",
    "q_purpose": "What is this loan for?",
    "q_purpose_opts":["Urgent necessity (health, housing, education)","Business or investment",
                      "Electronics, goods, furniture","Leisure or travel",
                      "Paying off another loan","Other"],
    "q_terms":   "Do you understand all loan conditions — fees, penalties, total cost?",
    "q_terms_y": "Yes, fully","q_terms_n":"Not fully",
    "q_delay":   "If salary is delayed 3 days, can you still make the payment?",
    "q_delay_y": "Yes, I have a buffer","q_delay_n":"Probably not",
    "q_refi":    "Is this loan to pay off another existing loan?","q_refi_y":"Yes","q_refi_n":"No",
    "q_sav_mo":  "How many months of loan payments can your savings cover?",
    "proj_note": "The tool simulates loan pressure over the full loan term, factoring in expected income and expense changes.",
    "proj_how":  "How to project your income?","proj_avg":"Follow average salary trends",
    "proj_own":  "Enter my own estimate","proj_none":"Assume no growth",
    "proj_pct":  "Expected annual income growth (%)","proj_infl":"Let expenses grow with expected price growth",
    "calc_note": "The calculation uses your budget data and basic economic assumptions (inflation, salary trends). The result is approximate.",
    "s4_h":      "Your loan assessment","r_sub":"Based on your inputs",
    "r_lbl_pmt": "Monthly payment","r_lbl_pct":"Share of income",
    "r_lbl_left":"Money left after all payments","r_lbl_first":"First risky month","r_na":"none",
    "r_low_h":   "✅ Loan looks affordable",
    "r_low_b":   "Based on your data, the loan pressure is manageable. The payment is a reasonable share of income and there should be money left after all expenses.",
    "r_med_h":   "⚠️ Moderate pressure — prepare carefully",
    "r_med_b":   "The loan may be manageable overall, but some months could be tight. We recommend checking the details and keeping a financial buffer.",
    "r_high_h":  "🔶 High pressure — consider revising the terms",
    "r_high_b":  "The loan will take up a significant portion of your budget. Consider a smaller amount, longer term, or waiting until your financial situation improves.",
    "r_crit_h":  "🔴 Loan may be financially risky",
    "r_crit_b":  "Based on your data, the loan pressure is very high. We recommend exploring alternatives: smaller amount, longer term, or postponing. This is not financial advice.",
    "dist_h":"Pressure distribution over loan term","dist_low":"Low-pressure months",
    "dist_med":"Moderate months","dist_high":"High-pressure months","dist_crit":"Critical months",
    "rec_before_h":"Before applying","rec_reduce_h":"How to reduce pressure",
    "rec_timing_h":"Payment timing","rec_careful_h":"When to be especially careful",
    "chk_date":  "Payment day is before salary day. Ask the bank to move the payment date to after your salary arrives.",
    "chk_terms": "You don't fully know the conditions. Ask the bank: total cost, all fees, penalties and early repayment rules.",
    "chk_total": "Find out the full loan cost — not just monthly payment, but total repayable amount.",
    "chk_buffer":"Have at least 1–2 months of payments saved up before signing.",
    "chk_delay": "You indicated you may not be able to pay if salary is delayed. Build a small reserve — at least one payment.",
    "red_amount":"Reduce the loan amount to lower the monthly payment.",
    "red_term":  "Choose a longer term — lower monthly payment (but higher total cost).",
    "red_flex":  "Temporarily cut optional spending — dining, subscriptions, shopping.",
    "red_no_extra":"Avoid taking new loans while this one is active.",
    "tim_date":  "Payment day ({p}) is before salary day ({s}). This creates a monthly timing risk.",
    "tim_ok":    "Payment day ({p}) is after salary day ({s}) — good. Timing risk is minimal.",
    "car_salary":"Salary may be delayed. Keep at least 2 payments in a separate accessible account.",
    "car_non_urg":"Loan is for non-essential purpose. If pressure is high, consider waiting or reducing the amount.",
    "car_refi":  "New loan to pay off old loan — high risk of a debt cycle. First discuss restructuring with the bank.",
    "car_low_inc":"Your income is below the national average. The payment will take a high share of your budget.",
    "car_macro": "Economic conditions are adding pressure. Consider borrowing less or waiting.",
    "car_flex":  "Large optional expenses. Reducing them could free up budget if needed.",
    "exp_charts":"Detailed monthly forecast","exp_table":"Monthly simulation table",
    "exp_macro": "Economic assumptions used in the calculation","exp_method":"How the result is calculated",
    "ch1_title": "Loan payment as % of income — month by month",
    "ch2_title": "Money left after payments — month by month",
    "ml_proba_title": "Model confidence (Random Forest) — predicted pressure probability per month",
    "macro_note":"These parameters are used internally. You do not need to analyse them — they affect the pressure forecast.",
    "download":  "Download table (CSV)",
    "method_body":"""**Monthly payment** is calculated using the annuity formula (equal monthly payments).

**Macro-level pressure** is classified by a trained **Random Forest** model (200 trees, class_weight=balanced), fit on monthly Uzbekistan macro-financial data (2010–2025, 192 observations). Features: inflation, policy rate, CPI MoM, nominal wage, real wage growth, real policy rate, debt burden indicator. Target: Low / Medium / High.

Model performance (dissertation analysis): Random Forest test accuracy 92.3%, F1-macro 0.651. Decision Tree: accuracy 79.5%, F1-macro 0.760. Logistic Regression showed the most stable time-series cross-validation performance (mean F1 0.58).

**Monthly pressure** is further scored based on:
- Payment as % of income each month
- Money left after all expenses and payments
- Savings buffer
- Payment timing vs salary date

**Final pressure levels:**
- ✅ Low — payment is up to 25% of income, buffer exists
- ⚠️ Moderate — 25–40%, some months may be tight
- 🔶 High — 40–50%, consider revising terms
- 🔴 Critical — above 50%, significant financial risk

The tool does not make bank decisions or provide financial advice.""",
    "lbl_low":"Low","lbl_med":"Moderate","lbl_high":"High","lbl_crit":"Critical",
    # table columns
    "tbl_date":"Date","tbl_month":"Month","tbl_pmt":"Payment","tbl_inc":"Income",
    "tbl_ess":"Essential","tbl_flex":"Optional","tbl_pti":"PTI %","tbl_tdb":"TDB %",
    "tbl_cash":"Cash left","tbl_bal":"Balance","tbl_lvl":"Level",
    # custom term
    "term_custom":"Custom term (months)","term_mode_pre":"Choose from list","term_mode_own":"Enter manually",
    # validation
    "v_rate_high":"Interest rate above 100% — please double-check.",
    "v_exp_high":"Total expenses exceed income — please verify your inputs.",
    "v_savings_low":"Savings are less than one monthly payment. We recommend building a buffer before applying.",
    # AI features
    "ai_insight_btn":   "💡 Get AI insight",
    "ai_insight_title": "💡 Personal AI insight",
    "ai_insight_spin":  "Analysing your situation...",
    "ai_opt_btn":       "🔄 AI: suggest better parameters",
    "ai_opt_title":     "🔄 Alternative scenarios from AI",
    "ai_opt_spin":      "Finding alternatives...",
    "ai_chat_title":    "💬 Ask the AI assistant",
    "ai_chat_ph":       "E.g. what happens if I lose my job for 2 months?",
    "ai_chat_btn":      "Send →",
    "ai_chat_spin":     "Thinking...",
    "ai_no_key":        "⚙️ To enable AI features, add ANTHROPIC_API_KEY to Streamlit Secrets.",
    "ai_error":         "AI is temporarily unavailable. Please try again later.",
    "ai_chat_clear":    "Clear chat",
    # Monte Carlo ML model
    "mc_default_lbl":   "Default probability (ML)",
    "mc_default_sub":   "Monte Carlo · Random Forest",
    "mc_no_model":      "loan_risk_model.pkl not found. Run model_training.py locally first.",
    # errors
    "err_dataset":"data_fw.xlsx not found. Place it in the same GitHub folder as app.py.",
    "err_forecast":"Could not build forecast for the full loan term. Check data_fw.xlsx.",
    "disclaimer":"Research tool. Not financial advice, not a credit score, not a bank decision. Does not approve or reject loans.",
},

# ─── Uzbek Latin ─────────────────────────────────────────────
"uz": {
    "s1":"Kredit","s2":"Byudjet","s3":"Vaziyat","s4":"Natija",
    "btn_back":"← Orqaga","btn_next":"Keyingi →","btn_calc":"Baholash →",
    "btn_restart":"Qaytadan boshlash",
    "hero_title":"💳 AI Loan Decision Assistant",
    "hero_sub":  "Kreditni olishdan oldin uning byudjetga ta'sirini baholang",
    "w_what":    "Bu vosita nima qiladi",
    "w_body":    "Kredit va byudjet ma'lumotlarini kiritasiz. Vosita har oyda kreditning byudjetingizga qancha yuklama qilishini hisoblaydi.",
    "w_not":     "Muhim:",
    "w_not1":    "Bu kredit tasdiqlash yoki rad etish emas",
    "w_not2":    "Bu moliyaviy maslahat emas",
    "w_not3":    "Bu kredit yuklamasini baholash uchun tadqiqot vositasi",
    "w_prepare": "Boshlashdan oldin kerak bo'ladiganlar:",
    "w_p1":      "Kredit miqdori va foiz stavkasi",
    "w_p2":      "Oylik sof daromad",
    "w_p3":      "Majburiy va ixtiyoriy oylik xarajatlar",
    "w_p4":      "Ish haqi kuni va to'lov muddati kuni",
    "btn_start": "Baholashni boshlash →",
    "s1_h":      "1-qadam / 3 — Kredit parametrlari",
    "f_amount":  "Kredit miqdori","f_amount_ph":"Masalan: 10 000 000",
    "f_rate":    "Yillik foiz stavkasi (%)","f_rate_ph":"Masalan: 24",
    "f_term":    "Kredit muddati","f_start":"Kreditni qachon olishni rejalashtirmoqdasiz?",
    "yr":        "yil","term_months":"oy",
    "est_pmt_lbl":"Taxminiy oylik to'lov","est_total_lbl":"Jami to'lanadigan summa",
    "v_amount":"Kredit miqdorini kiriting","v_rate":"Foiz stavkasini kiriting",
    "s2_h":      "2-qadam / 3 — Byudjetingiz",
    "f_income":  "Oylik sof daromad","f_income_ph":"Masalan: 5 000 000",
    "f_ess":     "Majburiy oylik xarajatlar","f_ess_help":"Ijara, kommunal, oziq-ovqat, transport",
    "f_ess_ph":  "Masalan: 2 000 000",
    "f_flex":    "Ixtiyoriy oylik xarajatlar","f_flex_help":"Kafe, obunalar, xarid — kerak bo'lsa kamaytiriladi",
    "f_flex_ph": "Masalan: 500 000",
    "f_exist":   "Mavjud kredit to'lovlari / oy","f_exist_help":"Hozir to'layotgan kredit to'lovlari jami",
    "f_exist_ph":"Masalan: 300 000 (yoki 0)",
    "f_exist_mo":"Mavjud kreditlarning qolgan muddati (oy)",
    "f_exist_mo_h":"Shu oylardan so'ng mavjud to'lovlar to'xtaydi. Yo'q bo'lsa 0 kiriting.",
    "f_sav":     "Jamg'arma / moliyaviy bufer","f_sav_help":"Hozirgi jamg'arma miqdori",
    "f_sav_ph":  "Masalan: 1 000 000 (yoki 0)",
    "f_sal_day": "Ish haqi keladigan kun","f_pay_day":"Kredit to'lovi muddati kuni",
    "budget_title":"Oylik byudjetingiz",
    "b_income":"Daromad","b_ess":"Majburiy xarajatlar","b_flex":"Ixtiyoriy xarajatlar",
    "b_exist":"Mavjud kredit to'lovlari","b_new":"Yangi kredit to'lovi","b_left":"Qoldiq",
    "w_negative":"⚠️ Yangi to'lov joriy byudjetga sig'maydi. Miqdorni kamaytiring yoki muddatni uzaytiring.",
    "w_tight":   "⚠️ Barcha to'lovlardan so'ng juda kam qoladi. Kamida 1–2 oylik to'lov uchun jamg'arma bo'lsin.",
    "v_income":  "Daromadingizni kiriting",
    "s3_h":      "3-qadam / 3 — Vaziyatingiz",
    "q_stable":  "Ish haqingiz barqarormi?","q_stable_y":"Ha, o'z vaqtida keladi","q_stable_n":"Ba'zan kechikadi",
    "q_purpose": "Kredit nima uchun kerak?",
    "q_purpose_opts":["Shoshilinch ehtiyoj (sog'liq, uy-joy, ta'lim)","Biznes yoki investitsiya",
                      "Texnika, tovar, mebel","Dam olish yoki sayohat",
                      "Boshqa kreditni to'lash","Boshqa"],
    "q_terms":   "Kredit shartlarini to'liq bilasizmi: komissiyalar, jarimalar, jami narxi?",
    "q_terms_y": "Ha, to'liq","q_terms_n":"To'liq emas",
    "q_delay":   "Ish haqi 3 kun kechiksa, to'lovni amalga oshira olasizmi?",
    "q_delay_y": "Ha, zahiram bor","q_delay_n":"Qiyin bo'ladi",
    "q_refi":    "Bu kredit boshqa kreditni to'lash uchunmi?","q_refi_y":"Ha","q_refi_n":"Yo'q",
    "q_sav_mo":  "Jamg'arma necha oylik kredit to'lovini qoplaydi?",
    "proj_note": "Vosita butun kredit muddati bo'yicha yuklamani hisoblaydi.",
    "proj_how":  "Daromadni qanday prognoz qilish?","proj_avg":"O'rtacha ish haqi dinamikasiga ko'ra",
    "proj_own":  "O'zim kiritaman","proj_none":"O'smaydi deb hisoblash",
    "proj_pct":  "Kutilayotgan yillik daromad o'sishi (%)","proj_infl":"Xarajatlar inflyatsiya bilan o'ssin",
    "calc_note": "Hisob-kitob byudjet ma'lumotlaringiz va asosiy iqtisodiy parametrlarni hisobga oladi. Natija taxminiy.",
    "s4_h":      "Baholash natijasi","r_sub":"Ma'lumotlaringiz asosida",
    "r_lbl_pmt": "Oylik to'lov","r_lbl_pct":"Daromadning ulushi",
    "r_lbl_left":"Barcha to'lovlardan keyingi qoldiq","r_lbl_first":"Birinchi xavfli oy","r_na":"yo'q",
    "r_low_h":   "✅ Kredit ko'tarilishi mumkin ko'rinadi",
    "r_low_b":   "Ma'lumotlaringizga ko'ra yuklama o'rtacha. To'lov daromadning kichik qismini egallaydi.",
    "r_med_h":   "⚠️ O'rtacha yuklama — tayyorgarlik ko'ring",
    "r_med_b":   "Kredit boshqarilishi mumkin, ammo ayrim oylar qiyin bo'lishi mumkin.",
    "r_high_h":  "🔶 Yuqori yuklama — shartlarni qayta ko'rib chiqing",
    "r_high_b":  "Kredit byudjetingizning katta qismini egallaydi. Miqdorni kamaytiring yoki muddatni uzaytiring.",
    "r_crit_h":  "🔴 Kredit moliyaviy xavfli bo'lishi mumkin",
    "r_crit_b":  "Ma'lumotlaringizga ko'ra yuklama juda yuqori. Alternativalarni ko'rib chiqing: kam miqdor, uzoq muddat yoki kechiktirish. Bu moliyaviy maslahat emas.",
    "dist_h":"Kredit muddati bo'yicha yuklama","dist_low":"Oddiy oylar",
    "dist_med":"O'rtacha oylar","dist_high":"Qiyin oylar","dist_crit":"Kritik oylar",
    "rec_before_h":"Kreditni olishdan oldin","rec_reduce_h":"Yuklamani qanday kamaytirish",
    "rec_timing_h":"To'lov sanasi xavfi","rec_careful_h":"Ehtiyot bo'lish kerak bo'lgan hollar",
    "chk_date":  "To'lov kuni ish haqi kunidan oldin. Bankdan to'lov kunini ish haqidan keyinga ko'chirishni so'rang.",
    "chk_terms": "Shartlarni to'liq bilmaysiz. Bankdan so'rang: jami qiymat, komissiyalar, jarimalar.",
    "chk_total": "Kredit narxini to'liq bilib oling — nafaqat oylik to'lov, balki jami to'lanadigan summa.",
    "chk_buffer":"Imzolashdan oldin kamida 1–2 oylik to'lov miqdori jamg'armada bo'lsin.",
    "chk_delay": "Ish haqi kechiksa to'lovga qiynalasiz. Kamida bir oylik to'lov uchun zahira yarating.",
    "red_amount":"Kredit miqdorini kamaytiring — oylik to'lov kamayadi.",
    "red_term":  "Muddatni uzaytiring — oylik to'lov kamayadi (lekin jami ko'p bo'ladi).",
    "red_flex":  "Ixtiyoriy xarajatlarni vaqtincha kamaytiring.",
    "red_no_extra":"Bu kredit to'languncha yangi kredit olmang.",
    "tim_date":  "To'lov kuni ({p}) ish haqi kunidan ({s}) oldin. Bu har oy xavf yaratadi.",
    "tim_ok":    "To'lov kuni ({p}) ish haqidan ({s}) keyin — yaxshi. Sana xavfi minimal.",
    "car_salary":"Ish haqi kechikishi mumkin. Alohida hisobda kamida 2 oylik to'lov zahirasi saqlang.",
    "car_non_urg":"Kredit shoshilinch ehtiyoj uchun emas. Yuklama yuqori bo'lsa — kutish yoki kam olish yaxshiroq.",
    "car_refi":  "Yangi kredit eski kreditni to'lash uchun — qarz tuzoqchasi xavfi. Avval bank bilan qayta tuzishni muhokama qiling.",
    "car_low_inc":"Daromadingiz mamlakatdagi o'rtachadan past. To'lov byudjetning katta qismini egallaydi.",
    "car_macro": "Iqtisodiy sharoitlar qo'shimcha bosim yaratmoqda. Kamroq qarz olish yoki kutish tavsiya etiladi.",
    "car_flex":  "Ixtiyoriy xarajatlar ko'p. Ularni kamaytirish byudjetni bo'shatishi mumkin.",
    "exp_charts":"Oyma-oy batafsil prognoz","exp_table":"Simulyatsiya jadvali",
    "exp_macro": "Hisob-kitobda ishlatiladigan iqtisodiy parametrlar","exp_method":"Natija qanday hisoblanadi",
    "ch1_title": "Daromadning kredit to'loviga ketadigan ulushi — oylar bo'yicha",
    "ch2_title": "To'lovlardan keyingi qoldiq — oylar bo'yicha",
    "ml_proba_title": "Model ishonchi (Random Forest) — oylar bo'yicha bosim ehtimolligi",
    "macro_note":"Bu parametrlar faqat ichki hisob-kitob uchun ishlatiladi.",
    "download":  "Jadvalni yuklab olish (CSV)",
    "method_body":"""**Oylik to'lov** annuitet formulasi yordamida hisoblanadi.

**Makro bosim darajasi** o'qitilgan **Random Forest** modeli (200 daraxt, class_weight=balanced) tomonidan aniqlanadi. Model O'zbekistonning oylik makroiqtisodiy ma'lumotlari asosida o'qitilgan (2010–2025, 192 kuzatuv). Belgilar: inflyatsiya, qayta moliyalashtirish stavkasi, CPI MoM, nominal ish haqi, real ish haqi o'sishi, real stavka, qarz ko'rsatkichi. Maqsad: Low / Medium / High.

Model ko'rsatkichlari: Random Forest test aniqlik 92.3%, F1-macro 0.651. Decision Tree: aniqlik 79.5%, F1-macro 0.760.

**Oylik yuklama** quyidagilarga asoslanadi:
- To'lovning daromaddagi ulushi
- Barcha xarajatlar va to'lovlardan keyingi qoldiq
- Jamg'arma zahirasi
- To'lov va ish haqi sanasining nisbati

**Yuklama darajalari:**
- ✅ Oddiy — to'lov daromadning 25% gacha, zahira bor
- ⚠️ O'rtacha — 25–40%, ba'zi oylar qiyin bo'lishi mumkin
- 🔶 Yuqori — 40–50%, shartlarni qayta ko'rib chiqing
- 🔴 Kritik — 50% dan yuqori, yuqori moliyaviy xavf

Vosita bank qarorlarini qabul qilmaydi.""",
    "lbl_low":"Oddiy","lbl_med":"O'rtacha","lbl_high":"Yuqori","lbl_crit":"Kritik",
    # table columns
    "tbl_date":"Sana","tbl_month":"Oy","tbl_pmt":"To'lov","tbl_inc":"Daromad",
    "tbl_ess":"Majburiy","tbl_flex":"Ixtiyoriy","tbl_pti":"PTI %","tbl_tdb":"TDB %",
    "tbl_cash":"Qoldiq","tbl_bal":"Balans","tbl_lvl":"Daraja",
    # custom term
    "term_custom":"O'z muddati (oy)","term_mode_pre":"Ro'yxatdan tanlash","term_mode_own":"Qo'lda kiritish",
    # validation
    "v_rate_high":"Foiz stavkasi 100% dan yuqori — iltimos, tekshiring.",
    "v_exp_high":"Xarajatlar daromaddan oshib ketdi — ma'lumotlarni tekshiring.",
    "v_savings_low":"Jamg'arma bir oylik to'lovdan kam. Ariza berishdan oldin zahira yaratishni tavsiya etamiz.",
    # AI features
    "ai_insight_btn":   "💡 AI tahlil olish",
    "ai_insight_title": "💡 Shaxsiy AI tahlili",
    "ai_insight_spin":  "Vaziyatingiz tahlil qilinmoqda...",
    "ai_opt_btn":       "🔄 AI: yaxshiroq parametrlar taklif qilish",
    "ai_opt_title":     "🔄 AI dan muqobil ssenariylar",
    "ai_opt_spin":      "Variantlar qidirilmoqda...",
    "ai_chat_title":    "💬 AI yordamchisidan so'rang",
    "ai_chat_ph":       "Masalan: 2 oy ishsiz qolsam nima bo'ladi?",
    "ai_chat_btn":      "Yuborish →",
    "ai_chat_spin":     "O'ylamoqda...",
    "ai_no_key":        "⚙️ AI funksiyalari uchun Streamlit Secrets ga ANTHROPIC_API_KEY qo'shing.",
    "ai_error":         "AI vaqtincha mavjud emas. Keyinroq urinib ko'ring.",
    "ai_chat_clear":    "Chatni tozalash",
    # Monte Carlo ML model
    "mc_default_lbl":   "Default ehtimolligi (ML)",
    "mc_default_sub":   "Monte Carlo · Random Forest",
    "mc_no_model":      "loan_risk_model.pkl topilmadi. Avval model_training.py ni lokal ishga tushiring.",
    # errors
    "err_dataset":"data_fw.xlsx topilmadi. Uni app.py bilan bir xil GitHub papkasiga joylashtiring.",
    "err_forecast":"Prognoz tuzilmadi. data_fw.xlsx ni tekshiring.",
    "disclaimer":"Tadqiqot vositasi. Moliyaviy maslahat, kredit skoringi yoki bank qarori emas.",
},
}  # end TR

# ══════════════════════════════════════════════════════════════
# CSS — blue primary, clean, minimal
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Layout ── */
.main>.block-container{max-width:760px;padding-top:.5rem;padding-bottom:3rem;}

/* ── Hero ── */
.hero{background:linear-gradient(135deg,#1d4ed8 0%,#2563eb 60%,#60a5fa 100%);
  border-radius:20px;padding:22px 28px 18px;color:#fff;margin-bottom:16px;}
.hero h1{font-size:1.55rem;font-weight:850;margin:0 0 4px;letter-spacing:-.03em;}
.hero p{margin:0;opacity:.9;font-size:.92rem;}

/* ── Welcome card ── */
.wcard{background:#fff;border:1px solid #dbeafe;border-radius:18px;
  padding:20px 24px;margin-bottom:14px;box-shadow:0 3px 14px rgba(15,23,42,.06);}

/* ── Wizard step bar ── */
.wbar{display:flex;margin-bottom:20px;border-radius:14px;overflow:hidden;border:1px solid #e2e8f0;}
.wi{flex:1;padding:9px 4px;text-align:center;font-size:.71rem;font-weight:750;
  background:#f8fafc;color:#94a3b8;border-right:1px solid #e2e8f0;}
.wi:last-child{border-right:none;}
.wi-done{background:#dbeafe;color:#1d4ed8;}
.wi-active{background:#2563eb;color:#fff;}

/* ── Input hint ── */
.hint{font-size:.73rem;color:#2563eb;font-weight:600;margin-top:-6px;
  margin-bottom:8px;font-family:'SF Mono',monospace;min-height:15px;letter-spacing:.01em;}

/* ── Budget preview ── */
.bcard{background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;
  padding:14px 18px;margin:10px 0;}
.brow{display:flex;justify-content:space-between;padding:5px 0;
  border-bottom:1px solid #f1f5f9;font-size:.85rem;}
.brow:last-child{border-bottom:none;padding-top:8px;margin-top:2px;font-weight:800;font-size:.9rem;}
.blbl{color:#64748b;}
.bval{font-weight:650;font-family:monospace;color:#0f172a;}
.bneg{color:#dc2626 !important;}
.bblue{color:#2563eb;}

/* ── Result boxes ── */
.rbox-low{background:#f0fdf4;border:1.5px solid #86efac;border-radius:16px;
  padding:16px 20px;color:#166534;margin-bottom:10px;}
.rbox-med{background:#fefce8;border:1.5px solid #fde047;border-radius:16px;
  padding:16px 20px;color:#854d0e;margin-bottom:10px;}
.rbox-high{background:#fff7ed;border:1.5px solid #fb923c;border-radius:16px;
  padding:16px 20px;color:#9a3412;margin-bottom:10px;}
.rbox-crit{background:#fef2f2;border:1.5px solid #f87171;border-radius:16px;
  padding:16px 20px;color:#991b1b;margin-bottom:10px;}

/* ── Metric cards ── */
.mcard{background:#fff;border:1px solid #e2e8f0;border-radius:14px;
  padding:12px 16px;text-align:center;box-shadow:0 2px 8px rgba(15,23,42,.04);}
.mlbl{color:#64748b;font-size:.76rem;font-weight:650;margin-bottom:3px;}
.mval{color:#0f172a;font-size:1.1rem;font-weight:850;letter-spacing:-.02em;}

/* ── Pressure pills ── */
.p-low{display:inline-block;padding:3px 11px;border-radius:999px;font-weight:750;
  font-size:.8rem;background:#f0fdf4;color:#166534;border:1px solid #86efac;}
.p-med{display:inline-block;padding:3px 11px;border-radius:999px;font-weight:750;
  font-size:.8rem;background:#fefce8;color:#854d0e;border:1px solid #fde047;}
.p-high{display:inline-block;padding:3px 11px;border-radius:999px;font-weight:750;
  font-size:.8rem;background:#fff7ed;color:#9a3412;border:1px solid #fb923c;}
.p-crit{display:inline-block;padding:3px 11px;border-radius:999px;font-weight:750;
  font-size:.8rem;background:#fef2f2;color:#991b1b;border:1px solid #f87171;}

/* ── Rec section label ── */
.rech{font-size:.76rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;
  color:#94a3b8;margin:16px 0 5px;}

/* ── Warning / info ── */
.wbox{background:#fefce8;border:1px solid #fde047;border-radius:12px;
  padding:10px 14px;color:#713f12;font-size:.87rem;margin-bottom:10px;}
.ibox{background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;
  padding:10px 14px;color:#1e3a8a;font-size:.87rem;margin-bottom:10px;}

/* ── Disclaimer ── */
.disc{background:#f8fafc;border:1px dashed #cbd5e1;border-radius:12px;
  padding:10px 14px;font-size:.76rem;color:#64748b;margin-top:1.2rem;}

/* ── Override Streamlit button colors: primary → blue ── */
.stButton>button[kind="primary"] {
    background-color: #2563eb !important;
    border-color: #2563eb !important;
    color: white !important;
}
.stButton>button[kind="primary"]:hover {
    background-color: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════
def ss(k, v):
    if k not in st.session_state: st.session_state[k] = v

ss("step",     0)
ss("sim_done", False)
ss("result",   None)
ss("lang",     "ru")

# ── Language selector ─────────────────────────────────────────
lc, _ = st.columns([2, 8])
with lc:
    sel_lang = st.selectbox(
        "🌐", list(LANG_OPTIONS.keys()),
        index=list(LANG_OPTIONS.values()).index(st.session_state["lang"]),
        label_visibility="collapsed", key="lang_sel",
    )
LANG = LANG_OPTIONS[sel_lang]
st.session_state["lang"] = LANG

def t(k: str) -> str:
    return TR.get(LANG, TR["ru"]).get(k, TR["ru"].get(k, k))

def pill(level: str) -> str:
    css = {"Low":"p-low","Medium":"p-med","High":"p-high","Critical":"p-crit"}.get(level,"p-med")
    lbl = {"Low":t("lbl_low"),"Medium":t("lbl_med"),"High":t("lbl_high"),"Critical":t("lbl_crit")}.get(level,level)
    return f'<span class="{css}">{lbl}</span>'

# ── Hero ──────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>{t('hero_title')}</h1>
  <p>{t('hero_sub')}</p>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CORE HELPERS
# ══════════════════════════════════════════════════════════════
def fmt(v: float) -> str:
    """Format as spaced thousands + UZS."""
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    return f"{abs(v):,.0f}".replace(",", " ") + " UZS"

def hint(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ") + " UZS" if v and v > 0 else ""

def pct(v: float, d: int = 1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    return f"{v:.{d}f}%"

def ann(p: float, r_pct: float, m: int) -> float:
    if p <= 0 or m <= 0: return 0.0
    r = r_pct / 100 / 12
    if r == 0: return p / m
    return p * r * (1+r)**m / ((1+r)**m - 1)

def mcard(label: str, value: str, sub: str = "") -> None:
    sub_html = f"<div style='font-size:.72rem;color:#94a3b8;margin-top:2px'>{sub}</div>" if sub else ""
    st.markdown(f"""
<div class="mcard">
  <div class="mlbl">{label}</div>
  <div class="mval">{value}</div>
  {sub_html}
</div>""", unsafe_allow_html=True)

def wbar(active: int) -> str:
    # steps: 0=welcome (hidden), 1,2,3,4
    steps = [t("s1"), t("s2"), t("s3"), t("s4")]
    html  = '<div class="wbar">'
    for i, lbl in enumerate(steps, 1):
        if i < active:    c = "wi wi-done"; pfx = "✓ "
        elif i == active: c = "wi wi-active"; pfx = ""
        else:             c = "wi"; pfx = ""
        html += f'<div class="{c}">{pfx}{lbl}</div>'
    html += '</div>'
    return html

# ══════════════════════════════════════════════════════════════
# DATA LOADING — silent, cached
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError("data_fw.xlsx not found.")
    xls   = pd.ExcelFile(DATA_FILE)
    sheet = MAIN_SHEET if MAIN_SHEET in xls.sheet_names else xls.sheet_names[0]
    df    = pd.read_excel(DATA_FILE, sheet_name=sheet)
    df.columns = (df.columns.astype(str).str.strip()
                  .str.replace(" ","_",regex=False).str.replace("-","_",regex=False))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing: raise ValueError(f"Dataset missing columns: {missing}")
    df = df.dropna(how="all").copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["repayment_pressure_level"] = df["repayment_pressure_level"].astype(str).str.strip()
    df = df[df["repayment_pressure_level"].isin(["Low","Medium","High"])].copy()
    if "year"  not in df.columns: df["year"]  = df["date"].dt.year
    if "month" not in df.columns: df["month"] = df["date"].dt.month
    return df.reset_index(drop=True)

try:
    macro_df = load_data()
except Exception as e:
    st.error(t("err_dataset")); st.stop()

try:
    clf, le = train_model(macro_df)
except Exception as e:
    st.error(f"Model training failed: {e}"); st.stop()

# ══════════════════════════════════════════════════════════════
# ML MODEL — train RandomForest on historical macro data
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def train_model(_df: pd.DataFrame):
    """Train RandomForestClassifier on the historical macro dataset.
    Uses the same 7 features as the dissertation analysis.
    Returns (clf, le) where le is the fitted LabelEncoder.
    """
    X = _df[FEATURE_COLUMNS].copy()
    for c in FEATURE_COLUMNS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median())
    y = _df["repayment_pressure_level"].astype(str).str.strip()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = RandomForestClassifier(
        n_estimators=200, random_state=42,
        class_weight="balanced", max_depth=6,
    )
    clf.fit(X.values, y_enc)
    return clf, le

def ml_pressure(row, clf, le) -> str:
    """Predict macro pressure level using the trained RF model."""
    feat = np.array([[float(row.get(c, 0) or 0) for c in FEATURE_COLUMNS]])
    pred = clf.predict(feat)[0]
    return le.inverse_transform([pred])[0]

def ml_pressure_proba(row, clf, le) -> dict[str, float]:
    """Return {class: probability} dict from the RF model."""
    feat = np.array([[float(row.get(c, 0) or 0) for c in FEATURE_COLUMNS]])
    proba = clf.predict_proba(feat)[0]
    return {le.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(proba)}

# ══════════════════════════════════════════════════════════════
# MONTE CARLO DEFAULT RISK MODEL
# ══════════════════════════════════════════════════════════════
_MC_FEATURES = [
    "income", "loan_amount", "term_months", "interest_rate",
    "essential_exp", "flex_exp", "existing_loans", "savings",
    "unstable", "timing_risk",
]

@st.cache_resource(show_spinner=False)
def load_mc_model():
    """Load the Monte Carlo trained loan_risk_model.pkl if it exists."""
    if not _JOBLIB_AVAILABLE:
        return None
    try:
        payload = _joblib.load(MC_MODEL_FILE)
        return payload  # dict: {model, features, version, ...}
    except Exception:
        return None

def mc_default_probability(u, unstable: bool = False, timing_risk: bool = False) -> float | None:
    """Return P(default) from the Monte Carlo RF model, or None if unavailable."""
    mc = load_mc_model()
    if mc is None:
        return None
    try:
        X = pd.DataFrame([{
            "income":        u.income,
            "loan_amount":   u.amount,
            "term_months":   u.months,
            "interest_rate": u.rate,
            "essential_exp": u.ess,
            "flex_exp":      u.flex,
            "existing_loans":u.exist,
            "savings":       u.savings,
            "unstable":      float(not u.stable),
            "timing_risk":   float(u.pay_day < u.sal_day),
        }])
        cols = mc.get("features", _MC_FEATURES)
        proba = mc["model"].predict_proba(X[cols])[0]
        default_idx = list(mc["model"].classes_).index(1)
        return round(float(proba[default_idx]), 4)
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════
# MACRO FORECAST — cached by params
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def build_forecast(start_str: str, months: int, _clf, _le) -> pd.DataFrame:
    df = macro_df

    def damped(s: pd.Series, h: int, lo=None, hi=None) -> np.ndarray:
        c = pd.Series(s).astype(float).dropna().reset_index(drop=True)
        if c.empty: return np.zeros(h)
        w  = c.tail(min(36,len(c))).reset_index(drop=True)
        sl = 0.0
        if len(w)>=6:
            try: sl,_ = np.polyfit(np.arange(len(w)), w.values, 1)
            except Exception: pass
        sl *= 0.35
        fc = np.array([float(w.iloc[-1]) + sl*(i+1) for i in range(h)])
        if lo is not None: fc = np.maximum(fc, lo)
        if hi is not None: fc = np.minimum(fc, hi)
        return fc

    def sal_fc(s: pd.Series, h: int) -> np.ndarray:
        c = pd.Series(s).astype(float).dropna().reset_index(drop=True)
        if c.empty: return np.ones(h)*5_000_000
        g = c.pct_change().tail(24).replace([np.inf,-np.inf],np.nan).dropna()
        mg = float(np.clip(g.median() if not g.empty else 0.005, 0.001, 0.018))
        cur = float(c.iloc[-1])
        out: list[float] = []
        for _ in range(h): cur *= (1+mg); out.append(cur)
        return np.array(out)

    start = pd.to_datetime(start_str).to_period("M").to_timestamp()
    end   = start + pd.DateOffset(months=months-1)
    last  = df["date"].max().to_period("M").to_timestamp()

    fut_frames: list[pd.DataFrame] = []
    if end > last:
        fdt = pd.date_range(start=last+pd.DateOffset(months=1), end=end, freq="MS")
        fh  = len(fdt)
        fut = pd.DataFrame({"date": fdt})
        fut["policy_rate_pct"]             = damped(df["policy_rate_pct"],             fh, lo=0, hi=40)
        fut["inflation_yoy_pct"]           = damped(df["inflation_yoy_pct"],           fh, lo=0, hi=50)
        fut["cpi_mom_pct"]                 = damped(df["cpi_mom_pct"],                 fh, lo=-5, hi=10)
        fut["nominal_wage_monthly_approx"] = sal_fc(df["nominal_wage_monthly_approx"], fh)
        fut["debt_burden_indicator_pct"]   = damped(df["debt_burden_indicator_pct"],   fh, lo=0, hi=100)
        fut_frames.append(fut)

    hist = df[["date","policy_rate_pct","inflation_yoy_pct","cpi_mom_pct",
               "nominal_wage_monthly_approx","debt_burden_indicator_pct"]].copy()
    cb = pd.concat([hist]+fut_frames, ignore_index=True).sort_values("date")
    cb["yoy_nom"] = cb["nominal_wage_monthly_approx"].pct_change(12)*100
    cb["real_wage_growth_pct"]  = cb["yoy_nom"] - cb["inflation_yoy_pct"]
    cb["real_policy_rate_pct"]  = cb["policy_rate_pct"] - cb["inflation_yoy_pct"]
    last_rwg = float(df["real_wage_growth_pct"].dropna().iloc[-1])
    cb["real_wage_growth_pct"]  = cb["real_wage_growth_pct"].fillna(last_rwg)
    cb["real_policy_rate_pct"]  = cb["real_policy_rate_pct"].fillna(
        cb["policy_rate_pct"]-cb["inflation_yoy_pct"])
    cb["predicted_macro_pressure"] = cb.apply(
        lambda r: ml_pressure(r, _clf, _le), axis=1
    )
    # ML probability columns for each class
    proba_series = cb.apply(lambda r: ml_pressure_proba(r, _clf, _le), axis=1)
    cb["prob_Low"]    = proba_series.apply(lambda x: x.get("Low",    0.0))
    cb["prob_Medium"] = proba_series.apply(lambda x: x.get("Medium", 0.0))
    cb["prob_High"]   = proba_series.apply(lambda x: x.get("High",   0.0))

    fc = cb[(cb["date"]>=start)&(cb["date"]<=end)].copy()
    fc = fc.head(months).reset_index(drop=True)
    fc["loan_month"] = np.arange(1, len(fc)+1)
    return fc

# ══════════════════════════════════════════════════════════════
# USER INPUT DATACLASS
# ══════════════════════════════════════════════════════════════
@dataclass
class Inp:
    amount: float
    rate:   float
    months: int
    income: float
    ess:    float
    flex:   float
    exist:  float
    exist_mo: int
    savings:  float
    sal_day:  int
    pay_day:  int
    start_str:str
    proj:     str      # "avg"|"own"|"none"
    proj_pct: float
    infl_exp: bool
    stable:   bool
    purpose:  str
    knows:    bool
    can_delay:bool
    is_refi:  bool
    sav_mo:   int

# ══════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════
def simulate(u: Inp, fc: pd.DataFrame) -> dict:
    mp       = ann(u.amount, u.rate, u.months)
    inc      = float(u.income)
    ess      = float(u.ess)
    flex     = float(u.flex)
    bal      = float(u.savings)
    prev_sal = float(fc["nominal_wage_monthly_approx"].iloc[0])
    fg       = (1 + u.proj_pct/100)**(1/12) - 1
    rows: list[dict] = []

    for _, r in fc.iterrows():
        lm = int(r["loan_month"])
        if lm > 1:
            if u.proj == "avg":
                cs   = float(r["nominal_wage_monthly_approx"])
                g    = float(np.clip((cs/prev_sal-1) if prev_sal>0 else 0, -0.02, 0.03))
                inc *= (1+g); prev_sal = cs
            elif u.proj == "own":
                inc *= (1+fg)
            if u.infl_exp:
                cpi  = float(np.clip(float(r["cpi_mom_pct"])/100, -0.03, 0.05))
                ess  *= (1+cpi); flex *= (1+cpi)

        ex_pmt = u.exist if lm <= u.exist_mo else 0.0
        ttl    = ess + flex
        pti    = mp/inc if inc>0 else np.nan
        tdb    = (mp+ex_pmt)/inc if inc>0 else np.nan
        cash   = inc - ttl - ex_pmt - mp
        bal   += cash
        avgs   = float(r["nominal_wage_monthly_approx"])
        ivsa   = inc/avgs if avgs>0 else 1.0
        pb4    = u.pay_day < u.sal_day
        mp_lvl = r["predicted_macro_pressure"]
        p_low  = float(r.get("prob_Low",    0.0))
        p_med  = float(r.get("prob_Medium", 0.0))
        p_high = float(r.get("prob_High",   0.0))

        sc = 0
        if not np.isnan(pti):
            if pti>=0.40: sc+=3
            elif pti>=0.25: sc+=1
        if not np.isnan(tdb):
            if tdb>=0.50: sc+=3
            elif tdb>=0.35: sc+=1
        if cash<0: sc+=3
        elif inc>0 and cash<inc*0.10: sc+=1
        if bal<0: sc+=1
        if ivsa<0.80: sc+=1
        if pb4: sc+=1
        if mp_lvl=="High": sc+=2
        elif mp_lvl=="Medium": sc+=1
        if float(r["inflation_yoy_pct"])>=12: sc+=1
        if inc>0 and flex/inc>0.20: sc+=1

        if sc<=1: lv="Low"
        elif sc<=3: lv="Medium"
        elif sc<=6: lv="High"
        else: lv="Critical"

        rows.append({
            "date":r["date"],"month":lm,"payment":mp,
            "income":inc,"ess":ess,"flex":flex,"exist_pmt":ex_pmt,
            "pti_pct":pti*100 if not np.isnan(pti) else np.nan,
            "tdb_pct":tdb*100 if not np.isnan(tdb) else np.nan,
            "cash":cash,"balance":bal,
            "infl":r["inflation_yoy_pct"],"rate":r["policy_rate_pct"],
            "avg_sal":avgs,"macro":mp_lvl,
            "prob_low":p_low,"prob_med":p_med,"prob_high":p_high,
            "score":sc,"level":lv,
        })

    sim  = pd.DataFrame(rows)
    n    = len(sim)
    low  = int((sim["level"]=="Low").sum())
    med  = int((sim["level"]=="Medium").sum())
    high = int((sim["level"]=="High").sum())
    crit = int((sim["level"]=="Critical").sum())
    mc   = float(sim["cash"].min())
    mb   = float(sim["balance"].min())
    hs   = (high+crit)/n if n else 0
    cs   = crit/n if n else 0

    fh_s = sim[sim["level"].isin(["High","Critical"])]["month"]
    frm: int|None = int(fh_s.min()) if len(fh_s)>0 else None

    if cs>=0.15 or (crit>0 and mc<0) or mb<-u.income:  res="Critical"
    elif hs>=0.30 or mc<0 or mb<0:                       res="High"
    elif high>0 or (med/n>=0.50 if n else False):        res="Medium"
    else:                                                 res="Low"

    return {
        "sim":sim, "mp":round(mp,2),
        "total":round(mp*u.months,2),
        "low":low,"med":med,"high":high,"crit":crit,
        "avg_pti":round(float(sim["pti_pct"].mean()),2),
        "max_tdb":round(float(sim["tdb_pct"].max()),2),
        "min_cash":round(mc,2),
        "first_risky":frm, "result":res, "fc":fc,
        "default_prob": mc_default_probability(u),
    }


# ══════════════════════════════════════════════════════════════
# AI HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════
def _get_ai_client():
    """Return Anthropic client if available and key is set."""
    if not _ANTHROPIC_AVAILABLE:
        return None
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if not key:
            return None
        return _anthropic.Anthropic(api_key=key)
    except Exception:
        return None

def _build_context(R: dict, inp, lang: str) -> str:
    """Build a compact loan-context string for Claude."""
    sim = R["sim"]
    lang_name = {"ru": "Russian", "en": "English", "uz": "Uzbek (Latin)"}[lang]
    return f"""You are a financial literacy assistant helping a user in Uzbekistan evaluate a loan.
Respond ONLY in {lang_name}. Be concise, warm, and practical. Never claim to be giving official financial advice.

LOAN PARAMETERS:
- Amount: {inp.amount:,.0f} UZS
- Annual rate: {inp.rate}%
- Term: {inp.months} months
- Monthly payment: {R['mp']:,.0f} UZS
- Total repayable: {R['total']:,.0f} UZS

USER BUDGET (monthly):
- Income: {inp.income:,.0f} UZS
- Essential expenses: {inp.ess:,.0f} UZS
- Optional expenses: {inp.flex:,.0f} UZS
- Existing loan payments: {inp.exist:,.0f} UZS (for {inp.exist_mo} more months)
- Savings: {inp.savings:,.0f} UZS

SIMULATION RESULTS:
- Overall result: {R['result']}
- Avg payment-to-income: {R['avg_pti']:.1f}%
- Min free cash (worst month): {R['min_cash']:,.0f} UZS
- Burden distribution: Low={R['low']} months, Medium={R['med']} months, High={R['high']} months, Critical={R['crit']} months
- First risky month: {R['first_risky'] or 'none'}
- Monte Carlo default probability: {f"{R['default_prob']*100:.1f}%" if R.get('default_prob') is not None else 'model not loaded'}

USER SITUATION:
- Salary stable: {inp.stable}
- Loan purpose: {inp.purpose}
- Understands loan terms: {inp.knows}
- Can pay if salary delayed 3 days: {inp.can_delay}
- Is refinancing: {inp.is_refi}
- Savings cover months of payment: {inp.sav_mo}
"""

def ai_stream_response(prompt: str, system: str) -> str:
    """Call Claude and stream response into a Streamlit container. Returns full text."""
    client = _get_ai_client()
    if not client:
        return None
    full = ""
    with st.empty() as placeholder:
        try:
            with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=600,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for chunk in stream.text_stream:
                    full += chunk
                    placeholder.markdown(full + "▌")
            placeholder.markdown(full)
        except Exception:
            placeholder.markdown("")
            return None
    return full


# ══════════════════════════════════════════════════════════════
# STEP 0 — WELCOME
# ══════════════════════════════════════════════════════════════
if st.session_state["step"] == 0:
    st.markdown(f"""
<div class="wcard">
  <h3 style="margin-top:0;color:#1d4ed8;font-size:1.1rem">{t('w_what')}</h3>
  <p style="color:#334155;margin-bottom:14px">{t('w_body')}</p>
  <div style="background:#f0f9ff;border-radius:10px;padding:12px 14px;margin-bottom:14px">
    <b style="color:#0369a1;font-size:.85rem">{t('w_not')}</b>
    <ul style="color:#334155;font-size:.87rem;margin:6px 0 0;padding-left:20px">
      <li>{t('w_not1')}</li><li>{t('w_not2')}</li><li>{t('w_not3')}</li>
    </ul>
  </div>
  <b style="color:#1e3a8a;font-size:.88rem">{t('w_prepare')}</b>
  <ul style="color:#334155;font-size:.87rem;margin:6px 0 0;padding-left:20px">
    <li>{t('w_p1')}</li><li>{t('w_p2')}</li><li>{t('w_p3')}</li><li>{t('w_p4')}</li>
  </ul>
</div>""", unsafe_allow_html=True)

    if st.button(t("btn_start"), type="primary", use_container_width=True):
        st.session_state["step"] = 1; st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════
# STEP 1 — LOAN DETAILS
# ══════════════════════════════════════════════════════════════
if st.session_state["step"] == 1:
    st.markdown(wbar(1), unsafe_allow_html=True)
    st.markdown(f"#### {t('s1_h')}")

    amount_raw = st.session_state.get("amount", None)
    rate_raw   = st.session_state.get("rate",   None)

    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input(
            t("f_amount"), min_value=0.0,
            value=float(amount_raw) if amount_raw else 0.0,
            step=500_000.0, format="%.0f",
            placeholder=t("f_amount_ph"),
        )
        st.markdown(f'<div class="hint">{hint(amount)}</div>', unsafe_allow_html=True)
    with c2:
        rate = st.number_input(
            t("f_rate"), min_value=0.0, max_value=100.0,
            value=float(rate_raw) if rate_raw else 0.0,
            step=0.5, format="%.1f",
            placeholder=t("f_rate_ph"),
        )

    c3, c4 = st.columns(2)
    with c3:
        term_mode = st.radio(
            t("f_term"),
            [t("term_mode_pre"), t("term_mode_own")],
            horizontal=True,
            index=0 if st.session_state.get("term_mode","pre")=="pre" else 1,
        )
        if term_mode == t("term_mode_pre"):
            yr_lbl = t("yr")
            term_opts = [1,2,3,5,7,10,15,20]
            prev_ty = st.session_state.get("term_years", 3)
            if prev_ty not in term_opts: prev_ty = 3
            term_years = st.selectbox(
                "", term_opts,
                index=term_opts.index(prev_ty),
                format_func=lambda x: f"{x} {yr_lbl}",
                label_visibility="collapsed",
            )
            term_months = int(term_years * 12)
            st.session_state["term_mode"] = "pre"
        else:
            prev_tm = st.session_state.get("term_months", 36)
            term_months = st.number_input(
                t("term_custom"), min_value=1, max_value=360,
                value=int(prev_tm), step=1,
                label_visibility="collapsed",
            )
            term_years = round(term_months / 12, 1)
            st.session_state["term_mode"] = "own"
    with c4:
        loan_start = st.date_input(
            t("f_start"),
            value=st.session_state.get("loan_start", date(2026,1,1)),
        )

    st.markdown(f"<div class='ibox'>📅 {term_months} {t('term_months')}</div>", unsafe_allow_html=True)

    # Preview estimated payment (only if both fields filled)
    if amount > 0 and rate > 0:
        if rate > 100:
            st.warning(t("v_rate_high"))
        est_pmt   = ann(amount, rate, term_months)
        est_total = est_pmt * term_months
        p1, p2 = st.columns(2)
        with p1: mcard(t("est_pmt_lbl"),   fmt(est_pmt))
        with p2: mcard(t("est_total_lbl"), fmt(est_total))

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _, nb = st.columns([1,1])
    with nb:
        if st.button(t("btn_next"), type="primary", use_container_width=True):
            if amount <= 0:
                st.warning(t("v_amount"))
            elif rate <= 0:
                st.warning(t("v_rate"))
            else:
                st.session_state.update({
                    "amount": amount, "rate": rate,
                    "term_years": term_years, "term_months": term_months,
                    "loan_start": loan_start,
                })
                st.session_state["step"] = 2; st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════
# STEP 2 — BUDGET
# ══════════════════════════════════════════════════════════════
if st.session_state["step"] == 2:
    st.markdown(wbar(2), unsafe_allow_html=True)
    st.markdown(f"#### {t('s2_h')}")

    est_pmt = ann(
        st.session_state.get("amount",0),
        st.session_state.get("rate",0),
        st.session_state.get("term_months",12),
    )

    def get(k, d): return st.session_state.get(k, d)

    c1, c2 = st.columns(2)
    with c1:
        income = st.number_input(t("f_income"), min_value=0.0,
                                 value=float(get("income",0.0)),
                                 step=100_000.0, format="%.0f",
                                 placeholder=t("f_income_ph"))
        st.markdown(f'<div class="hint">{hint(income)}</div>', unsafe_allow_html=True)
    with c2:
        ess = st.number_input(t("f_ess"), min_value=0.0,
                              value=float(get("ess",0.0)),
                              step=100_000.0, format="%.0f",
                              help=t("f_ess_help"),
                              placeholder=t("f_ess_ph"))
        st.markdown(f'<div class="hint">{hint(ess)}</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        flex = st.number_input(t("f_flex"), min_value=0.0,
                               value=float(get("flex",0.0)),
                               step=50_000.0, format="%.0f",
                               help=t("f_flex_help"),
                               placeholder=t("f_flex_ph"))
        st.markdown(f'<div class="hint">{hint(flex)}</div>', unsafe_allow_html=True)
    with c4:
        exist = st.number_input(t("f_exist"), min_value=0.0,
                                value=float(get("exist",0.0)),
                                step=50_000.0, format="%.0f",
                                help=t("f_exist_help"),
                                placeholder=t("f_exist_ph"))
        st.markdown(f'<div class="hint">{hint(exist)}</div>', unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        exist_mo = st.number_input(t("f_exist_mo"), min_value=0, max_value=360,
                                   value=int(get("exist_mo",0)), step=1,
                                   help=t("f_exist_mo_h"))
    with c6:
        savings = st.number_input(t("f_sav"), min_value=0.0,
                                  value=float(get("savings",0.0)),
                                  step=100_000.0, format="%.0f",
                                  help=t("f_sav_help"),
                                  placeholder=t("f_sav_ph"))
        st.markdown(f'<div class="hint">{hint(savings)}</div>', unsafe_allow_html=True)

    c7, c8 = st.columns(2)
    with c7:
        sal_day = st.selectbox(t("f_sal_day"), list(range(1,32)),
                               index=int(get("sal_day_idx",4)),
                               format_func=str)
    with c8:
        pay_day = st.selectbox(t("f_pay_day"), list(range(1,32)),
                               index=int(get("pay_day_idx",9)),
                               format_func=str)

    # Live budget preview
    free  = income - ess - flex - exist - est_pmt
    fcs   = "bneg" if free < 0 else ""
    fs    = ("−" if free < 0 else "") + fmt(abs(free))
    st.markdown(f"""
<div class="bcard">
  <div style="font-weight:800;font-size:.85rem;color:#1e3a8a;margin-bottom:8px">
    📊 {t('budget_title')}
  </div>
  <div class="brow"><span class="blbl">{t('b_income')}</span><span class="bval">{fmt(income)}</span></div>
  <div class="brow"><span class="blbl">{t('b_ess')}</span><span class="bval">−{fmt(ess)}</span></div>
  <div class="brow"><span class="blbl">{t('b_flex')}</span><span class="bval">−{fmt(flex)}</span></div>
  <div class="brow"><span class="blbl">{t('b_exist')}</span><span class="bval">−{fmt(exist)}</span></div>
  <div class="brow"><span class="blbl bblue">{t('b_new')}</span><span class="bval bblue">−{fmt(est_pmt)}</span></div>
  <div class="brow"><span class="blbl">{t('b_left')}</span>
    <span class="bval {fcs}">{fs}</span></div>
</div>""", unsafe_allow_html=True)

    if income > 0 and free < 0:
        st.markdown(f'<div class="wbox">{t("w_negative")}</div>', unsafe_allow_html=True)
    elif income > 0 and free < income * 0.10:
        st.markdown(f'<div class="wbox">{t("w_tight")}</div>', unsafe_allow_html=True)

    if income > 0 and (ess + flex + exist) > income:
        st.warning(t("v_exp_high"))

    if income > 0 and savings < est_pmt and est_pmt > 0:
        st.warning(t("v_savings_low"))

    bk, nx = st.columns(2)
    with bk:
        if st.button(t("btn_back"), use_container_width=True):
            st.session_state["step"] = 1; st.rerun()
    with nx:
        if st.button(t("btn_next"), type="primary", use_container_width=True):
            if income <= 0:
                st.warning(t("v_income"))
            else:
                st.session_state.update({
                    "income":income,"ess":ess,"flex":flex,"exist":exist,
                    "exist_mo":exist_mo,"savings":savings,
                    "sal_day":sal_day,"pay_day":pay_day,
                    "sal_day_idx":sal_day-1,"pay_day_idx":pay_day-1,
                })
                st.session_state["step"] = 3; st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════
# STEP 3 — PERSONAL SITUATION
# ══════════════════════════════════════════════════════════════
if st.session_state["step"] == 3:
    st.markdown(wbar(3), unsafe_allow_html=True)
    st.markdown(f"#### {t('s3_h')}")

    def get(k, d): return st.session_state.get(k, d)

    qa, qb = st.columns(2)
    with qa:
        stable_ans = st.radio(t("q_stable"),
                              [t("q_stable_y"),t("q_stable_n")], horizontal=True,
                              index=0 if get("stable",True) else 1)
        stable = stable_ans == t("q_stable_y")

        terms_ans = st.radio(t("q_terms"),
                             [t("q_terms_y"),t("q_terms_n")], horizontal=True,
                             index=0 if get("knows",True) else 1)
        knows = terms_ans == t("q_terms_y")

        refi_ans = st.radio(t("q_refi"),
                            [t("q_refi_n"),t("q_refi_y")], horizontal=True,
                            index=1 if get("is_refi",False) else 0)
        is_refi = refi_ans == t("q_refi_y")

    with qb:
        opts = t("q_purpose_opts")
        prev = get("purpose", opts[0])
        if prev not in opts: prev = opts[0]
        purpose = st.selectbox(t("q_purpose"), opts, index=opts.index(prev))

        delay_ans = st.radio(t("q_delay"),
                             [t("q_delay_y"),t("q_delay_n")], horizontal=True,
                             index=0 if get("can_delay",True) else 1)
        can_delay = delay_ans == t("q_delay_y")

        sav_mo = st.selectbox(t("q_sav_mo"), [0,1,2,3,6,12],
                              index=get("sav_mo_idx",1),
                              format_func=str)

    st.markdown("---")

    proj_opts   = [t("proj_avg"),t("proj_own"),t("proj_none")]
    proj_keys   = ["avg","own","none"]
    pp, pg, pi  = st.columns(3)
    with pp:
        proj_lbl  = st.selectbox(t("proj_how"), proj_opts,
                                 index=proj_keys.index(get("proj","avg")))
        proj_mode = proj_keys[proj_opts.index(proj_lbl)]
    with pg:
        proj_pct = st.number_input(t("proj_pct"), min_value=-20.0, max_value=100.0,
                                   value=float(get("proj_pct",7.0)), step=0.5,
                                   disabled=(proj_mode!="own"))
    with pi:
        infl_exp = st.toggle(t("proj_infl"), value=bool(get("infl_exp",True)))

    st.markdown(f'<div class="ibox">ℹ️ {t("calc_note")}</div>', unsafe_allow_html=True)

    bk, calc = st.columns(2)
    with bk:
        if st.button(t("btn_back"), use_container_width=True):
            st.session_state["step"] = 2; st.rerun()
    with calc:
        if st.button(t("btn_calc"), type="primary", use_container_width=True):
            st.session_state.update({
                "stable":stable,"knows":knows,"is_refi":is_refi,
                "purpose":purpose,"can_delay":can_delay,
                "sav_mo":sav_mo,"sav_mo_idx":[0,1,2,3,6,12].index(sav_mo),
                "proj":proj_mode,"proj_pct":proj_pct,"infl_exp":infl_exp,
            })
            G = st.session_state
            inp = Inp(
                amount=G["amount"], rate=G["rate"], months=G["term_months"],
                income=G["income"], ess=G["ess"], flex=G["flex"],
                exist=G["exist"], exist_mo=int(G["exist_mo"]),
                savings=G["savings"], sal_day=int(G["sal_day"]),
                pay_day=int(G["pay_day"]),
                start_str=str(G["loan_start"]),
                proj=proj_mode, proj_pct=proj_pct, infl_exp=infl_exp,
                stable=stable, purpose=purpose, knows=knows,
                can_delay=can_delay, is_refi=is_refi, sav_mo=int(sav_mo),
            )
            fc = build_forecast(str(G["loan_start"]), G["term_months"], clf, le)
            if len(fc) < G["term_months"]:
                st.error(t("err_forecast")); st.stop()
            R  = simulate(inp, fc)
            st.session_state["result"]   = R
            st.session_state["inp_obj"]  = inp
            st.session_state["sim_done"] = True
            st.session_state["step"]     = 4
            st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════
# STEP 4 — RESULT
# ══════════════════════════════════════════════════════════════
if st.session_state["step"] == 4 and st.session_state.get("sim_done"):
    R   = st.session_state["result"]
    inp: Inp = st.session_state["inp_obj"]
    sim = R["sim"]
    lv  = R["result"]

    st.markdown(wbar(4), unsafe_allow_html=True)
    st.markdown(f"#### {t('s4_h')}")
    st.markdown(f"<p style='color:#94a3b8;font-size:.85rem;margin-top:-6px'>{t('r_sub')}</p>",
                unsafe_allow_html=True)

    # ── Main result card ─────────────────────────────────────
    bxmap = {"Low": ("rbox-low","r_low_h","r_low_b"),
             "Medium":("rbox-med","r_med_h","r_med_b"),
             "High":  ("rbox-high","r_high_h","r_high_b"),
             "Critical":("rbox-crit","r_crit_h","r_crit_b")}
    bx, th, tb = bxmap.get(lv, bxmap["Medium"])
    st.markdown(f"""
<div class="{bx}">
  <h4 style="margin:0 0 6px">{t(th)}</h4>
  <p style="margin:0;font-size:.9rem">{t(tb)}</p>
</div>""", unsafe_allow_html=True)

    # ── 4 key metrics ────────────────────────────────────────
    income_pct = f"{R['avg_pti']:.0f}%" if not np.isnan(R["avg_pti"]) else "—"
    frm        = str(R["first_risky"]) if R["first_risky"] else t("r_na")
    m1,m2,m3,m4 = st.columns(4)
    with m1: mcard(t("r_lbl_pmt"),  fmt(R["mp"]))
    with m2: mcard(t("r_lbl_pct"),  income_pct)
    with m3: mcard(t("r_lbl_left"), fmt(R["min_cash"]))
    with m4: mcard(t("r_lbl_first"),frm)

    # ── Monte Carlo default probability ──────────────────────
    dp = R.get("default_prob")
    if dp is not None:
        dp_pct = dp * 100
        if dp_pct < 20:   dp_color, dp_label = "#10b981", t("lbl_low")
        elif dp_pct < 50: dp_color, dp_label = "#f59e0b", t("lbl_med")
        elif dp_pct < 75: dp_color, dp_label = "#f97316", t("lbl_high")
        else:             dp_color, dp_label = "#ef4444", t("lbl_crit")
        filled = int(dp_pct / 5)
        gauge_html = "".join(
            f'<span style="color:{dp_color}">█</span>' if i < filled
            else '<span style="color:#e2e8f0">█</span>'
            for i in range(20)
        )
        st.markdown(f"""
<div class="mcard" style="margin-top:8px">
  <div class="mlbl">{t('mc_default_lbl')} &nbsp;
    <span style="font-size:.72rem;color:#94a3b8">{t('mc_default_sub')}</span>
  </div>
  <div style="display:flex;align-items:center;gap:12px;margin-top:6px">
    <div style="font-size:1.4rem;font-weight:850;color:{dp_color}">{dp_pct:.1f}%</div>
    <div style="font-size:10px;letter-spacing:.5px;line-height:1">{gauge_html}</div>
    <div style="font-size:.78rem;color:{dp_color};font-weight:650">{dp_label}</div>
  </div>
</div>""", unsafe_allow_html=True)
    elif load_mc_model() is None and _JOBLIB_AVAILABLE:
        st.markdown(f'<div class="ibox" style="font-size:.8rem">🤖 {t("mc_no_model")}</div>',
                    unsafe_allow_html=True)

    # ── Burden distribution ──────────────────────────────────
    st.markdown(f"<div style='margin-top:14px;font-weight:700;font-size:.88rem;color:#334155'>{t('dist_h')}</div>",
                unsafe_allow_html=True)
    d1,d2,d3,d4 = st.columns(4)
    for col, level_key, cnt_key, lbl_key in [
        (d1,"Low","low","dist_low"),(d2,"Medium","med","dist_med"),
        (d3,"High","high","dist_high"),(d4,"Critical","crit","dist_crit")
    ]:
        with col:
            st.markdown(pill(level_key), unsafe_allow_html=True)
            st.metric(t(lbl_key), R[cnt_key])

    st.markdown("---")

    # ── Structured recommendations ───────────────────────────
    # BEFORE applying
    before: list[str] = []
    if inp.pay_day < inp.sal_day:     before.append(t("chk_date"))
    if not inp.knows:                 before.append(t("chk_terms"))
    before.append(t("chk_total"))
    if inp.sav_mo < 1:                before.append(t("chk_buffer"))
    if not inp.can_delay:             before.append(t("chk_delay"))
    if before:
        st.markdown(f'<div class="rech">✅ {t("rec_before_h")}</div>', unsafe_allow_html=True)
        for b in before: st.write(f"- {b}")

    # REDUCE pressure
    reduce: list[str] = []
    if R["avg_pti"] >= 30:            reduce.append(t("red_amount"))
    if R["avg_pti"] >= 25:            reduce.append(t("red_term"))
    if inp.flex > 0:                  reduce.append(t("red_flex"))
    if R["max_tdb"] >= 35:            reduce.append(t("red_no_extra"))
    if reduce:
        st.markdown(f'<div class="rech">📉 {t("rec_reduce_h")}</div>', unsafe_allow_html=True)
        for r in reduce: st.write(f"- {r}")

    # TIMING risk
    st.markdown(f'<div class="rech">📅 {t("rec_timing_h")}</div>', unsafe_allow_html=True)
    if inp.pay_day < inp.sal_day:
        st.warning(t("tim_date").format(p=inp.pay_day, s=inp.sal_day))
    else:
        st.success(t("tim_ok").format(p=inp.pay_day, s=inp.sal_day))

    # WHEN to be careful
    careful: list[str] = []
    if not inp.stable:              careful.append(t("car_salary"))
    non_urg = ["Техника","Отдых","Electronics","Leisure","Texnika","Dam olish"]
    if any(k.lower() in inp.purpose.lower() for k in non_urg):
        careful.append(t("car_non_urg"))
    if inp.is_refi:                 careful.append(t("car_refi"))
    avg_sal = float(sim["avg_sal"].mean())
    fin_inc = float(sim["income"].iloc[-1])
    if avg_sal > 0 and fin_inc/avg_sal < 0.80:
        careful.append(t("car_low_inc"))
    if (sim["macro"]=="High").sum()/len(sim) > 0.30:
        careful.append(t("car_macro"))
    if inp.income > 0 and inp.flex/inp.income > 0.20:
        careful.append(t("car_flex"))
    if careful:
        st.markdown(f'<div class="rech">⚠️ {t("rec_careful_h")}</div>', unsafe_allow_html=True)
        for c in careful: st.write(f"- {c}")

    # ── Expanders ────────────────────────────────────────────
    CLRS = {"Low":"#10b981","Medium":"#f59e0b","High":"#f97316","Critical":"#ef4444"}

    with st.expander(f"📈 {t('exp_charts')}"):
        # Chart 1: PTI
        st.markdown(f"**{t('ch1_title')}**")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=sim["date"], y=sim["pti_pct"],
            mode="lines", name="",
            line=dict(color="#2563eb",width=2.5),
            hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
            fill="tozeroy", fillcolor="rgba(37,99,235,.08)"))
        for thr, clr, lbl in [(25,"#10b981","25%"),(40,"#f59e0b","40%"),(50,"#ef4444","50%")]:
            fig1.add_hline(y=thr, line_dash="dot", line_color=clr,
                           annotation_text=lbl, annotation_position="right")
        fig1.update_layout(height=260, margin=dict(t=10,b=30,l=40,r=70),
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            showlegend=False, xaxis_title="", yaxis_title="%", hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: Cash
        st.markdown(f"**{t('ch2_title')}**")
        bar_c = [CLRS.get(l,"#2563eb") for l in sim["level"]]
        fig2  = go.Figure()
        fig2.add_trace(go.Bar(x=sim["date"], y=sim["cash"],
            marker_color=bar_c,
            hovertemplate="%{x|%b %Y}: %{y:,.0f} UZS<extra></extra>", name=""))
        fig2.add_hline(y=0, line_color="#ef4444", line_width=1.5)
        fig2.update_layout(height=240, margin=dict(t=10,b=30,l=40,r=20),
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            showlegend=False, xaxis_title="", yaxis_title="UZS", hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander(f"🗂️ {t('exp_table')}"):
        tbl = sim[["date","month","payment","income","ess","flex",
                   "pti_pct","tdb_pct","cash","balance","level"]].copy()
        lvl_map = {"Low":t("lbl_low"),"Medium":t("lbl_med"),
                   "High":t("lbl_high"),"Critical":t("lbl_crit")}
        tbl["level"] = tbl["level"].map(lvl_map)
        tbl.columns = [t("tbl_date"),t("tbl_month"),t("tbl_pmt"),t("tbl_inc"),
                       t("tbl_ess"),t("tbl_flex"),t("tbl_pti"),t("tbl_tdb"),
                       t("tbl_cash"),t("tbl_bal"),t("tbl_lvl")]
        num_c = [t("tbl_pmt"),t("tbl_inc"),t("tbl_ess"),t("tbl_flex"),t("tbl_cash"),t("tbl_bal")]
        pct_c = [t("tbl_pti"),t("tbl_tdb")]
        fmt_d = {c:"{:,.0f}" for c in num_c if c in tbl.columns}
        fmt_d.update({c:"{:.1f}" for c in pct_c if c in tbl.columns})
        st.dataframe(tbl.style.format(fmt_d), use_container_width=True, hide_index=True)
        csv = sim.to_csv(index=False).encode("utf-8")
        st.download_button(t("download"), data=csv,
                           file_name="loan_simulation.csv", mime="text/csv")

    with st.expander(f"🌐 {t('exp_macro')}"):
        fc = R["fc"]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=fc["date"], y=fc["inflation_yoy_pct"],
            mode="lines", name="Инфляция %", line=dict(color="#f97316",width=2)))
        fig3.add_trace(go.Scatter(x=fc["date"], y=fc["policy_rate_pct"],
            mode="lines", name="Ставка %", line=dict(color="#7c3aed",width=2)))
        fig3.add_trace(go.Scatter(x=fc["date"], y=fc["real_wage_growth_pct"],
            name="Рост зарплат %",
            line=dict(color="#10b981",width=2,dash="dash")))
        fig3.update_layout(height=240, margin=dict(t=10,b=30,l=40,r=20),
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            legend=dict(orientation="h",y=-0.28),
            xaxis_title="", yaxis_title="%", hovermode="x unified")
        st.plotly_chart(fig3, use_container_width=True)

        # ML probability chart — RF model confidence per month
        st.markdown(f"**🤖 {t('ml_proba_title')}**")
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=sim["date"], y=sim["prob_high"]*100,
            name=t("lbl_high"), marker_color="#ef4444",
            hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>"))
        fig4.add_trace(go.Bar(
            x=sim["date"], y=sim["prob_med"]*100,
            name=t("lbl_med"), marker_color="#f59e0b",
            hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>"))
        fig4.add_trace(go.Bar(
            x=sim["date"], y=sim["prob_low"]*100,
            name=t("lbl_low"), marker_color="#10b981",
            hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>"))
        fig4.update_layout(
            barmode="stack", height=220,
            margin=dict(t=10,b=30,l=40,r=20),
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            legend=dict(orientation="h",y=-0.30),
            xaxis_title="", yaxis_title="%", hovermode="x unified",
            yaxis=dict(range=[0,100]))
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown(f"<p style='color:#94a3b8;font-size:.78rem'>{t('macro_note')}</p>",
                    unsafe_allow_html=True)

    with st.expander(f"🔬 {t('exp_method')}"):
        st.markdown(t("method_body"))

    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # AI FEATURES
    # ══════════════════════════════════════════════════════════
    _client_ok = _get_ai_client() is not None
    if not _client_ok:
        st.markdown(f'<div class="ibox">⚙️ {t("ai_no_key")}</div>', unsafe_allow_html=True)
    else:
        _ctx = _build_context(R, inp, LANG)

        # ── 1. AI Insight ────────────────────────────────────
        st.markdown(f"#### {t('ai_insight_title')}")
        if st.button(t("ai_insight_btn"), key="btn_insight"):
            with st.spinner(t("ai_insight_spin")):
                _prompt = (
                    "In 3–4 sentences give a personalized, specific insight about this user's loan situation. "
                    "Focus on the single most important risk or opportunity you see in their numbers. "
                    "Be concrete — mention actual figures (amounts, percentages, months)."
                )
                _resp = ai_stream_response(_prompt, _ctx)
                if _resp is None:
                    st.error(t("ai_error"))

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── 2. AI Scenario Optimizer ─────────────────────────
        st.markdown(f"#### {t('ai_opt_title')}")
        if st.button(t("ai_opt_btn"), key="btn_opt"):
            with st.spinner(t("ai_opt_spin")):
                _prompt = (
                    "Suggest exactly 3 alternative loan parameter combinations that would reduce the burden. "
                    "For each: state the new amount and/or term, estimate the new monthly payment, "
                    "explain why it helps. Be specific with numbers. Format as 3 numbered options."
                )
                _resp = ai_stream_response(_prompt, _ctx)
                if _resp is None:
                    st.error(t("ai_error"))

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── 3. AI Chat ───────────────────────────────────────
        st.markdown(f"#### {t('ai_chat_title')}")
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for msg in st.session_state["chat_history"]:
            role_icon = "🧑" if msg["role"] == "user" else "🤖"
            st.markdown(f"**{role_icon}** {msg['content']}")

        _user_q = st.text_input(
            "", placeholder=t("ai_chat_ph"),
            key="chat_input", label_visibility="collapsed"
        )
        cb1, cb2 = st.columns([3, 1])
        with cb1:
            _send = st.button(t("ai_chat_btn"), key="btn_chat", type="primary",
                              use_container_width=True)
        with cb2:
            if st.button(t("ai_chat_clear"), key="btn_chat_clear",
                         use_container_width=True):
                st.session_state["chat_history"] = []
                st.rerun()

        if _send and _user_q.strip():
            st.session_state["chat_history"].append(
                {"role": "user", "content": _user_q.strip()}
            )
            _history_msgs = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["chat_history"]
            ]
            with st.spinner(t("ai_chat_spin")):
                client = _get_ai_client()
                if client:
                    full_reply = ""
                    with st.empty() as ph:
                        try:
                            with client.messages.stream(
                                model="claude-sonnet-4-20250514",
                                max_tokens=500,
                                system=_ctx,
                                messages=_history_msgs,
                            ) as stream:
                                for chunk in stream.text_stream:
                                    full_reply += chunk
                                    ph.markdown(f"**🤖** {full_reply}▌")
                            ph.markdown(f"**🤖** {full_reply}")
                        except Exception:
                            ph.markdown("")
                            full_reply = t("ai_error")
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": full_reply}
                    )
                    st.rerun()

    st.markdown("---")
    if st.button(t("btn_restart"), use_container_width=False):
        for k in ["step","sim_done","result","inp_obj",
                  "amount","rate","term_years","term_months","loan_start",
                  "income","ess","flex","exist","exist_mo","savings",
                  "sal_day","pay_day","stable","knows","is_refi",
                  "purpose","can_delay","sav_mo","proj","proj_pct","infl_exp",
                  "chat_history"]:
            st.session_state.pop(k, None)
        st.session_state["step"] = 0; st.rerun()

    st.markdown(f"<div class='disc'>{t('disclaimer')}</div>", unsafe_allow_html=True)
