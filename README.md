## CIVIA: modelo de risco de acessibilidade para pessoas cegas e surdas

Este é o modelo que fiz para o CIVIA, nosso projeto para a CitizenTech. É um modelo de Machine Learning que classifica um trecho urbano em `low`, `medium` ou `high` quanto à barreira de acessibilidade, levando em conta o perfil do pedestre (cego ou surdo). O app usa o resultado para avaliar rotas.

### Integrantes

* Ana Luiza Santana da Rocha, RM561194
* Evandro Yuji Kaibara de Oliveira, RM559274
* Guilherme Rodrigues Leite, RM560131
* Luís Filipe Crivellaro, RM560877
* Pedro Henrique Batista Bergamin, RM559443

### Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python build_notebook.py     # gera os dados, executa o notebook, salva model.joblib e metrics.json
python predict.py            # demonstração da integração com duas rotas
pytest tests                 # 11 testes: dados, modelo e integração
```

Para só ler o notebook, abra `citizentech_ml.ipynb`. Ele já está executado.

### Arquivos

| Arquivo | Conteúdo |
| --- | --- |
| `citizentech_ml.ipynb` | Problema, dados, análise, treino, avaliação e integração |
| `generate_data.py` | Simulação do dataset (semente 42) |
| `data/accessibility_segments.csv` | Dataset com 3000 trechos |
| `model.joblib` | Pipeline treinado (preparação e modelo) |
| `metrics.json` | Métricas de validação cruzada e de teste |
| `predict.py` | Função `rate_route`, ponto de uso do modelo na solução |

### Origem dos dados

Os dados são simulados. Não achei uma base pública com piso tátil, sinal sonoro, ruído, iluminação e tráfego por trecho, junto com um rótulo de dificuldade por perfil. O rótulo sai de uma pontuação que escrevi em `generate_data.py`, com ruído gaussiano e 4% de valores vazios em ruído e iluminação.

Para pessoas cegas pesam piso tátil, sinal sonoro em travessias, braille, obstáculos e ruído alto. Para pessoas surdas pesam sinal visual, avisos visuais, iluminação e tráfego intenso. Largura, inclinação e escadas afetam os dois perfis.

### Decisões

* **Perfil como variável.** O mesmo piso tátil muda muito o risco para quem é cego e quase nada para quem é surdo, então um modelo sem o perfil perderia essa diferença.
* **Pipeline único.** Imputação por mediana, padronização e one-hot ficam dentro do `Pipeline`, ajustados só no treino. Isso evita vazamento e permite que `predict.py` receba dados crus.
* **F1 macro como métrica.** As classes não são iguais em tamanho e a acurácia esconde o desempenho nas menores. Antes de equilibrar os cortes das classes, `high` tinha só 3% dos casos e o F1 macro refletia isso, então os cortes passaram a ser por quantis (cerca de 25%, 45% e 30%).
* **Escolha do modelo por validação cruzada** (5 partes, só no treino). O conjunto de teste ficou intocado até a avaliação final.

### Resultados

| Modelo | F1 macro (validação cruzada) |
| --- | --- |
| Baseline (classe mais comum) | 0,207 |
| Regressão logística | 0,603 |
| Gradient boosting | 0,621 |
| Random forest (escolhido) | 0,642 |

No teste (600 trechos), o random forest chegou a **F1 macro de 0,649** e acurácia de 0,66. Nenhum dos 150 trechos `high` foi previsto como `low`, que é o erro mais perigoso para o usuário.

O valor não chega perto de 1 porque a pontuação simulada tem ruído (desvio de 6 pontos), que o modelo não consegue prever.

### Limites

* A métrica mede se o modelo recupera a regra simulada. Não prova desempenho numa cidade real.
* Com dados reais (mapeamento de calçadas, semáforos, relatos de usuários) o mesmo pipeline serve, mas o resultado precisa ser reavaliado.
* Pessoas com surdocegueira têm necessidades diferentes e não estão cobertas.
* Num produto real, o limiar de decisão deveria errar para o lado seguro.
