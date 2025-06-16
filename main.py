import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# 1. Wczytaj dane
dane = pd.read_csv("dane3.csv") 
X = dane.drop("target", axis=1)  
y = dane["target"]

# 2. Normalizacja / Standaryzacja
skalowanie = StandardScaler()
X_scaled = skalowanie.fit_transform(X)

# 3. Selekcja wektorów
selekcja = SelectKBest(score_func=f_classif, k=5)
X_selected = selekcja.fit_transform(X_scaled, y)

# 4. Modele
# Dodano random_state dla powtarzalności modeli, które mają losowe komponenty
model1 = DecisionTreeClassifier(random_state=42)
model2 = KNeighborsClassifier() 
model3 = SVC(probability=True, random_state=42)

komitet = VotingClassifier(estimators=[
    ('drzewo', model1),
    ('knn', model2),
    ('svc', model3)
], voting='soft')

# 5. Kroswalidacja
# Liczba foldów CV jest dynamicznie ustalana na minimum z 10 lub najmniejszej liczby wystąpień klasy
min_class_count = y.value_counts().min()
cv_folds = min(10, min_class_count)

dokladnosci = cross_val_score(komitet, X_selected, y, cv=cv_folds, scoring='accuracy')
srednia_dokladnosc_cv = dokladnosci.mean() # Zapisujemy średnią dokładność CV
print(f"Średnia dokładność kroswalidacji ({cv_folds} foldów): {srednia_dokladnosc_cv:.4f}")

# 6. Podział na zbiór treningowy i testowy dla macierzy pomyłek
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
komitet.fit(X_train, y_train)
y_pred = komitet.predict(X_test)

# Obliczanie dokładności na zbiorze testowym
dokladnosc_testowa = accuracy_score(y_test, y_pred)
print(f"Dokładność na zbiorze testowym: {dokladnosc_testowa:.4f}")

# 7. Macierz pomyłek
macierz = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6)) # Ustawienie rozmiaru wykresu dla lepszej czytelności
sns.heatmap(macierz, annot=True, fmt="d", cmap="Blues",
            xticklabels=komitet.classes_, yticklabels=komitet.classes_) # Etykiety klas
plt.title("Macierz Pomyłek")
plt.xlabel("Przewidziane")
plt.ylabel("Rzeczywiste")
plt.show()

# 8. Wykres dokładności z kroswalidacji
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(dokladnosci) + 1), dokladnosci, marker='o', linestyle='-')
plt.title(f"Dokładność w kroswalidacji ({len(dokladnosci)}x)")
plt.xlabel("Numer Folds")
plt.ylabel("Dokładność")
plt.grid(True) 
plt.xticks(range(1, len(dokladnosci) + 1)) 
plt.tight_layout() 
plt.show()

# --- Zapisywanie wyników do pliku ---

nazwa_pliku_wynikow = "wynik.txt"
next_experiment_number = 1

# Logika do określenia następnego numeru eksperymentu
try:
    with open(nazwa_pliku_wynikow, 'r') as f:
        for line in f:
            if line.startswith("Próba "):
                try:
                    num_str = line.split(':')[0].split(' ')[1]
                    current_próba_num = int(num_str)
                    if current_próba_num >= next_experiment_number:
                        next_experiment_number = current_próba_num + 1
                except (ValueError, IndexError):
                    pass
except FileNotFoundError:
    pass

# Pobierz aktualną datę i godzinę
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(nazwa_pliku_wynikow, 'a') as f:
    f.write(f"Próba {next_experiment_number}:\n")
    f.write(f"Plik: dane3\n")
    f.write(f"--- Uruchomienie: {timestamp} ---\n")
    f.write(f"Średnia dokładność kroswalidacji: {srednia_dokladnosc_cv:.4f}\n")
    f.write(f"Dokładność na zbiorze testowym: {dokladnosc_testowa:.4f}\n")
    f.write(f"Liczba foldów CV: {len(dokladnosci)}\n")
    f.write("-" * 30 + "\n\n")

print(f"\nWyniki zapisano do pliku: {nazwa_pliku_wynikow}")
