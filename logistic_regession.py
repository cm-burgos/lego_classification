import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

sc = StandardScaler()
# leer csv
df = pd.read_csv('lego_data.csv', sep=',', header=None)

y = df[df.columns[-1]].values  # columna de etiquetas

X = df.iloc[:, 0:64]  # columnas de características
X = X.values

# dividir datos y normalizar X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# configuración de regresor y entrenamiento
log_reg = LogisticRegression(solver='saga', max_iter=500, random_state=0)
log_reg.fit(X_train, y_train)

# predición y prueba
y_pred = log_reg.predict(X_test)
y_test_scores = log_reg.predict_proba(X_test)

# métricas de evaluación
MCC = matthews_corrcoef(y_test, y_pred)
print("matthews_corrcoef", MCC)
ACC = accuracy_score(y_test, y_pred)
print("Accuracy", ACC)


from sklearn.metrics import plot_confusion_matrix
matrix = plot_confusion_matrix(log_reg, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize=None)
plt.title('Matriz de confusión para regresión logística')
plt.show()