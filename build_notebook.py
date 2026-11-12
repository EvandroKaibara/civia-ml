"""Build citizentech_ml.ipynb from the cell list below, then execute it."""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# CIVIA: risco de barreira de acessibilidade para pessoas cegas e surdas

Este é o modelo de Machine Learning que fiz para o CIVIA, nosso projeto para a CitizenTech.

**Problema.** Quem é cego ou surdo e anda pela cidade encontra trechos que parecem comuns para os outros e são perigosos para ele. Um cruzamento sem sinal sonoro é uma barreira para uma pessoa cega. Uma rua escura, de muito tráfego e sem aviso visual é uma barreira para uma pessoa surda, que não ouve os carros.

**Pergunta do modelo.** Dadas as características de um trecho urbano e o perfil do pedestre (cego ou surdo), qual o nível de barreira: `low`, `medium` ou `high`?

**Quem usa a previsão.** O app da CitizenTech, ao sugerir rotas. Cada trecho recebe um nível e a rota herda o pior deles.

**Como rodar.** `pip install -r requirements.txt`, depois executar o notebook de cima para baixo. Ele gera os dados, treina e salva `model.joblib` e `metrics.json`."""),

md("""## 1. Dados e origem

Não encontrei uma base pública que traga, por trecho de calçada, sinal sonoro, piso tátil, ruído, iluminação e tráfego junto com um rótulo de dificuldade por perfil de deficiência. Por isso os dados são **simulados** (a atividade permite), com semente fixa (`seed=42`) em `generate_data.py`.

A regra de simulação vem de como cada perfil se orienta. Para pessoas cegas contam piso tátil, sinal sonoro em travessias, sinalização em braille, obstáculos e ruído que cobre os sons do ambiente. Para pessoas surdas contam sinal visual, avisos visuais, iluminação e tráfego intenso, já que elas não ouvem os veículos. Largura, inclinação e escadas pesam para os dois. Entra ruído gaussiano na pontuação, e 4% dos valores de ruído e iluminação ficam vazios, como acontece quando um sensor falha.

**Limite que precisa ficar claro:** como o rótulo sai de uma regra que eu mesmo escrevi, a métrica abaixo mede se o modelo recupera essa regra a partir dos dados, não se ele acerta a realidade de uma cidade. Com dados de campo (mapeamentos de calçadas, relatos de usuários) o mesmo pipeline serviria, mas o número final seria menor."""),

code("""import json
import pathlib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from generate_data import generate

SEED = 42
df = generate()
pathlib.Path("data").mkdir(exist_ok=True)
df.to_csv("data/accessibility_segments.csv", index=False)
df.head()"""),

code("""print(df.shape)
print(df.isna().sum()[lambda s: s > 0])
print(df["barrier_level"].value_counts(normalize=True).round(3))"""),

md("""## 2. Análise e preparação

Três perguntas guiam a análise: as classes estão equilibradas, os perfis sofrem barreiras diferentes, e quais variáveis separam os níveis?"""),

code("""fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
pd.crosstab(df["profile"], df["barrier_level"], normalize="index")[["low", "medium", "high"]].plot.bar(ax=ax[0], rot=0)
ax[0].set_title("Nível de barreira por perfil")
df.groupby("barrier_level")["traffic_vehicles_per_min"].mean().reindex(["low", "medium", "high"]).plot.bar(ax=ax[1], rot=0)
ax[1].set_title("Tráfego médio por nível")
plt.tight_layout(); plt.show()

print(df.groupby(["profile", "tactile_paving"])["barrier_level"].apply(lambda s: (s == "high").mean()).round(2))"""),

md("""O que a tabela mostra: sem piso tátil, a fatia de trechos `high` para pessoas cegas é bem maior, e para pessoas surdas esse atributo pouco muda o resultado. É isso que justifica usar o perfil como variável do modelo e não treinar um modelo único sem ele.

**Preparação.** Valores vazios de ruído e iluminação são preenchidos com a mediana. Variáveis numéricas são padronizadas (necessário para a regressão logística, inofensivo para a floresta) e as categóricas viram colunas binárias. Tudo fica dentro de um `Pipeline`, o que evita vazamento: a mediana e a escala saem só do conjunto de treino."""),

code("""target = "barrier_level"
X, y = df.drop(columns=target), df[target]
cat_cols = ["profile", "segment_type"]
num_cols = [c for c in X.columns if c not in cat_cols]

prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(len(X_train), len(X_test))"""),

md("""## 3. Modelo e treino

Comparo três modelos contra um baseline que sempre prevê a classe mais frequente. A regressão logística é simples e explicável. A floresta aleatória e o gradient boosting capturam interações do tipo "sinal sonoro só importa em travessia e só para pessoa cega", que um modelo linear não enxerga. A escolha é por validação cruzada de 5 partes no treino, usando F1 macro, porque a acurácia esconde o desempenho nas classes menores."""),

code("""candidates = {
    "baseline (classe mais comum)": DummyClassifier(strategy="most_frequent"),
    "regressão logística": LogisticRegression(max_iter=2000),
    "random forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    "gradient boosting": HistGradientBoostingClassifier(random_state=SEED),
}
cv_scores = {}
for name, est in candidates.items():
    pipe = Pipeline([("prep", prep), ("model", est)])
    s = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1_macro")
    cv_scores[name] = (s.mean(), s.std())
    print(f"{name:30s} F1 macro = {s.mean():.3f} +- {s.std():.3f}")"""),

code("""best_name = max(cv_scores, key=lambda k: cv_scores[k][0])
print("escolhido:", best_name)
model = Pipeline([("prep", prep), ("model", candidates[best_name])]).fit(X_train, y_train)"""),

md("""## 4. Avaliação

A métrica principal é o **F1 macro** no conjunto de teste, que o modelo não viu em nenhum momento. Ela pesa as três classes igualmente. Olho também a matriz de confusão, porque os erros não custam o mesmo: dizer `low` para um trecho que é `high` leva uma pessoa a um lugar perigoso, e o erro contrário só faz a rota ficar mais longa."""),

code("""pred = model.predict(X_test)
macro_f1 = f1_score(y_test, pred, average="macro")
base_f1 = f1_score(y_test, DummyClassifier(strategy="most_frequent").fit(X_train, y_train).predict(X_test), average="macro")
print(classification_report(y_test, pred, labels=["low", "medium", "high"]))
print(f"F1 macro teste = {macro_f1:.3f} | baseline = {base_f1:.3f}")
ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=["low", "medium", "high"], cmap="Blues")
plt.show()

dangerous = ((y_test == "high") & (pred == "low")).sum()
print("trechos high previstos como low (erro perigoso):", dangerous, "de", (y_test == "high").sum())"""),

code("""imp = permutation_importance(model, X_test, y_test, scoring="f1_macro", n_repeats=10, random_state=SEED)
order = np.argsort(imp.importances_mean)[::-1][:8]
for i in order:
    print(f"{X.columns[i]:28s} {imp.importances_mean[i]:.3f}")"""),

md("""A importância por permutação confirma que o modelo usa o que a regra de simulação previa (piso tátil, tráfego, obstáculos, sinais). Se uma variável irrelevante aparecesse no topo, isso indicaria vazamento ou ruído sendo decorado."""),

code("""joblib.dump(model, "model.joblib")
with open("metrics.json", "w") as f:
    json.dump({"model": best_name, "f1_macro_test": round(macro_f1, 4), "f1_macro_baseline": round(base_f1, 4),
               "cv_f1_macro": {k: round(v[0], 4) for k, v in cv_scores.items()},
               "dangerous_errors": int(dangerous)}, f, indent=2, ensure_ascii=False)
print(open("metrics.json").read())"""),

md("""## 5. Integração na solução

`predict.py` carrega o modelo salvo e avalia uma rota trecho a trecho para um perfil. A rota recebe o pior nível entre os trechos, porque basta uma travessia perigosa para a rota inteira ser ruim. É essa função que a tela de navegação da CitizenTech chamaria."""),

code("""import predict
route_a = [
    {"segment_type": "sidewalk", "tactile_paving": 1, "audible_signal": 0, "visual_signal": 0, "braille_signage": 0,
     "visual_alerts": 0, "sidewalk_width_m": 2.4, "obstacles_per_100m": 1, "slope_pct": 3.0,
     "ambient_noise_db": 62.0, "lighting_lux": 50.0, "traffic_vehicles_per_min": 8},
    {"segment_type": "crossing", "tactile_paving": 1, "audible_signal": 1, "visual_signal": 1, "braille_signage": 1,
     "visual_alerts": 1, "sidewalk_width_m": 2.0, "obstacles_per_100m": 0, "slope_pct": 2.0,
     "ambient_noise_db": 66.0, "lighting_lux": 60.0, "traffic_vehicles_per_min": 10},
]
route_b = [
    {"segment_type": "crossing", "tactile_paving": 0, "audible_signal": 0, "visual_signal": 0, "braille_signage": 0,
     "visual_alerts": 0, "sidewalk_width_m": 1.0, "obstacles_per_100m": 6, "slope_pct": 12.0,
     "ambient_noise_db": 85.0, "lighting_lux": 8.0, "traffic_vehicles_per_min": 25},
]
for name, route in [("A", route_a), ("B", route_b)]:
    for profile in ("blind", "deaf"):
        print(name, predict.rate_route(route, profile))"""),

md("""## 6. Limites e próximos passos

* Os dados são simulados e o rótulo vem de uma regra minha. A métrica prova que o pipeline funciona, não que o modelo acerta numa cidade real.
* O próximo passo com mais valor é trocar a simulação por dados coletados: mapeamento de calçadas e semáforos (por exemplo OpenStreetMap) e relatos de usuários da própria CitizenTech.
* Pessoas com surdocegueira dependem de outros sinais e ficam fora deste modelo.
* O erro mais caro é prever `low` para um trecho `high`. Num produto real eu ajustaria o limiar de decisão para errar para o lado seguro.""")
]

nb = nbf.v4.new_notebook(cells=cells)
ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": "."}})
nbf.write(nb, "citizentech_ml.ipynb")
print("notebook executed")
