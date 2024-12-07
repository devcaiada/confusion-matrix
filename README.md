# Projeto de Cálculo de Métricas de Avaliação de Aprendizado

Este projeto demonstra como treinar um modelo de Machine Learning, calcular várias métricas de avaliação e plotar a Curva de ROC utilizando o conjunto de dados "Breast Cancer Wisconsin (Diagnostic) Dataset", sample do próprio **scikit learn**.

## Estrutura do Repositório

~~~
.
├── README.md
└── evaluate_model.py
└── assets
~~~

## Pré-requisitos

Certifique-se de ter o **Python** e as seguintes bibliotecas instaladas:

- **numpy**

- **pandas**

- **scikit-learn**

- **matplotlib**

Você pode instalar essas bibliotecas utilizando o **pip**:

~~~python
pip install numpy pandas scikit-learn matplotlib
~~~

## Descrição do Código

### 1. Carregamento e Preparação dos Dados

~~~python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Carregar o dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
~~~

Utilizamos o dataset "Breast Cancer Wisconsin (Diagnostic) Dataset" do scikit-learn, que contém características de células cancerígenas e não cancerígenas. Os dados são divididos em conjuntos de treino e teste.

### 2. Treinamento do Modelo

~~~python
# Treinar um modelo de Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
~~~

Treinamos um modelo de Random Forest para classificar as amostras como cancerígenas ou não cancerígenas.

### 3. Plotagem da Matriz de Confusão

~~~python
# Plotar a matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusão')
plt.colorbar()
classes = data.target_names
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.ylabel('Classe Real')
plt.xlabel('Classe Predita')

# Printar a matriz de confusão
print("Matriz de Confusão:")
print(cm)
~~~

A matriz de confusão mostra o desempenho do modelo de classificação, indicando os verdadeiros positivos (VP), verdadeiros negativos (VN), falsos positivos (FP) e falsos negativos (FN).

![confusion-matrix](https://github.com/devcaiada/confusion-matrix/blob/main/assets/confusion-matrix.png?raw=true)

### 4. Cálculo das Métricas de Avaliação

~~~python
# Calcular as métricas
VP = cm[1, 1]
VN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

sensibilidade = VP / (VP + FN)
especificidade = VN / (FP + VN)
acurácia = (VP + VN) / np.sum(cm)
precisão = VP / (VP + FP)
f_score = 2 * (precisão * sensibilidade) / (precisão + sensibilidade)

# Printar os resultados das métricas
print(f"Sensibilidade: {sensibilidade:.2f}")
print(f"Especificidade: {especificidade:.2f}")
print(f"Acurácia: {acurácia:.2f}")
print(f"Precisão: {precisão:.2f}")
print(f"F-Score: {f_score:.2f}")
~~~

![print](https://github.com/devcaiada/confusion-matrix/blob/main/assets/print.png?raw=true)

**Sensibilidade**: Medida da capacidade do modelo de identificar corretamente os positivos reais.

![sensibilidade](https://github.com/devcaiada/confusion-matrix/blob/main/assets/sensibilidade.png?raw=true)

**Especificidade**: Medida da capacidade do modelo de identificar corretamente os negativos reais.

![especificidade](https://github.com/devcaiada/confusion-matrix/blob/main/assets/especificidade.png?raw=true)

**Acurácia**: Proporção de verdadeiros resultados (positivos e negativos) entre o total de casos examinados.

![acuracia](https://github.com/devcaiada/confusion-matrix/blob/main/assets/acuracia.png?raw=true)

**Precisão**: Proporção de positivos verdadeiros entre os resultados positivos previstos pelo modelo.

![precisao](https://github.com/devcaiada/confusion-matrix/blob/main/assets/precisao.png?raw=true)

**F-Score**: Média harmônica da precisão e da sensibilidade, útil para avaliar modelos com classes desbalanceadas.

![f-score](https://github.com/devcaiada/confusion-matrix/blob/main/assets/f-score.png?raw=true)

### 5. Plotagem da Curva de ROC

~~~python
# Plotar a Curva de ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva de ROC')
plt.legend(loc="lower right")
plt.show()
~~~

A **Curva de ROC (Receiver Operating Characteristic)** é um gráfico que mostra a relação entre a taxa de verdadeiros positivos (Sensibilidade) e a taxa de falsos positivos. A área sob a curva (AUC) é uma medida de quão bem o modelo é capaz de distinguir entre classes positivas e negativas.

![ROC-curve](https://github.com/devcaiada/confusion-matrix/blob/main/assets/ROC-curve.png?raw=true)


## Como Executar

1. Clone este repositório:
~~~Sh
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
~~~

2. Instale as dependências necessárias:

~~~Sh
pip install numpy pandas scikit-learn matplotlib
~~~

3. Execute o script:
~~~Sh
python evaluate_model.py
~~~

## Contribuição <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Rocket.png" alt="Rocket" width="25" height="25" />

Sinta-se à vontade para contribuir com este projeto. Você pode abrir issues para relatar problemas ou fazer pull requests para melhorias.

