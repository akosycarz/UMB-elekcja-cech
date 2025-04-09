import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import make_scorer, f1_score
import pandas as pd

def oblicz_f_score(X, y):
    """
    Oblicza wartość F-score dla każdej cechy
    """
    klasy = np.unique(y)
    n_cech = X.shape[1]
    f_scores = np.zeros(n_cech)
    
    for cecha in range(n_cech):
        licznik = 0
        mianownik = 0
        srednia_calosc = np.mean(X[:, cecha])
        
        for klasa in klasy:
            dane_klasy = X[y == klasa, cecha]
            srednia_klasy = np.mean(dane_klasy)
            wariancja_klasy = np.var(dane_klasy)
            
            licznik += len(dane_klasy) * (srednia_klasy - srednia_calosc) ** 2
            mianownik += len(dane_klasy) * wariancja_klasy
        
        f_scores[cecha] = licznik / mianownik if mianownik != 0 else 0
        
    return f_scores

def selekcja_sekwencyjna_cv(X, y, n_cech, kierunek='forward', n_splits=5):
    """
    Wykonuje sekwencyjną selekcję cech z walidacją krzyżową
    """
    # Inicjalizacja klasyfikatora SVM
    clf = SVC(kernel='linear', random_state=42)
    
    # Inicjalizacja walidacji krzyżowej ze stratyfikacją
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Inicjalizacja selektora cech
    sfs = SequentialFeatureSelector(
        clf,
        n_features_to_select=n_cech,
        direction=kierunek,
        scoring='f1',
        cv=cv,  # Używamy zdefiniowanego obiektu cv
        n_jobs=-1  # Wykorzystanie wszystkich dostępnych rdzeni
    )
    
    # Dopasowanie selektora
    sfs.fit(X, y)
    
    return np.where(sfs.get_support())[0]

def ocen_cechy_cv(X, y, wybrane_cechy, n_splits=5):
    """
    Ocenia wybrane cechy używając walidacji krzyżowej
    """
    clf = SVC(kernel='linear', random_state=42)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Używamy tylko wybranych cech
    X_selected = X[:, wybrane_cechy]
    
    # Obliczamy wyniki dla każdego foldu
    scores = cross_val_score(clf, X_selected, y, cv=cv, scoring='f1')
    
    return scores

def porownaj_metody_selekcji(X, y, n_cech=5, n_splits=5):
    """
    Porównuje różne metody selekcji cech używając walidacji krzyżowej
    """
    # Standaryzacja danych
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Selekcja sekwencyjna w przód z CV
    print("\nWykonywanie selekcji sekwencyjnej w przód...")
    sfs_indeksy = selekcja_sekwencyjna_cv(X_scaled, y, n_cech, 'forward', n_splits)
    
    # Selekcja sekwencyjna wstecz z CV
    print("Wykonywanie selekcji sekwencyjnej wstecz...")
    sbs_indeksy = selekcja_sekwencyjna_cv(X_scaled, y, n_cech, 'backward', n_splits)
    
    # Selekcja na podstawie F-score
    print("Obliczanie F-score...")
    f_scores = oblicz_f_score(X_scaled, y)
    fscore_indeksy = np.argsort(f_scores)[::-1][:n_cech]
    
    # Ocena wyników dla każdej metody
    wyniki = {}
    for nazwa, indeksy in [('SFS', sfs_indeksy), 
                          ('SBS', sbs_indeksy), 
                          ('F-score', fscore_indeksy)]:
        scores = ocen_cechy_cv(X_scaled, y, indeksy, n_splits)
        wyniki[nazwa] = {
            'indeksy': indeksy,
            'wyniki_foldow': scores,
            'sredni_f1': np.mean(scores),
            'std_f1': np.std(scores)
        }
    
    # Wizualizacja wyników
    plt.figure(figsize=(15, 12))
    
    # Wykres F-scores
    plt.subplot(3, 1, 1)
    plt.bar(range(len(f_scores)), f_scores)
    plt.title('Wartości F-score dla wszystkich cech')
    plt.xlabel('Indeks cechy')
    plt.ylabel('Wartość F-score')
    
    # Wykres porównawczy metod (boxplot)
    plt.subplot(3, 1, 2)
    boxplot_data = [wyniki[m]['wyniki_foldow'] for m in ['SFS', 'SBS', 'F-score']]
    plt.boxplot(boxplot_data, labels=['SFS', 'SBS', 'F-score'])
    plt.title(f'Rozkład F1-scores dla różnych metod ({n_splits}-fold CV)')
    plt.ylabel('F1-score')
    
    # Wykres słupkowy średnich wyników
    plt.subplot(3, 1, 3)
    x = np.arange(3)
    srednie = [wyniki[m]['sredni_f1'] for m in ['SFS', 'SBS', 'F-score']]
    std = [wyniki[m]['std_f1'] for m in ['SFS', 'SBS', 'F-score']]
    
    plt.bar(x, srednie, yerr=std, capsize=5)
    plt.xticks(x, ['SFS', 'SBS', 'F-score'])
    plt.title('Porównanie metod selekcji cech')
    plt.ylabel('Średni F1-score')
    
    plt.tight_layout()
    # plt.savefig('porownanie_metod_selekcji_cv.png')
    plt.show()
    
    # Wyświetlenie szczegółowych wyników
    print("\nWyniki selekcji cech z {}-krotną walidacją krzyżową:".format(n_splits))
    print("-" * 70)
    print("Metoda\t\tWybrane cechy\t\tF1-score (średnia ± std)\t\tWyniki dla foldów")
    print("-" * 70)
    
    for nazwa in ['SFS', 'SBS', 'F-score']:
        print(f"{nazwa}\t\t{wyniki[nazwa]['indeksy']}\t\t"
              f"{wyniki[nazwa]['sredni_f1']:.3f} ± {wyniki[nazwa]['std_f1']:.3f}\t\t"
              f"{wyniki[nazwa]['wyniki_foldow']}")
    
    return wyniki

# Generowanie przykładowych danych
print("Generowanie danych...")
X, y = make_classification(
    n_samples=100,    # liczba próbek
    n_features=20,    # liczba cech
    n_informative=10, # liczba informatywnych cech
    n_redundant=5,    # liczba redundantnych cech
    n_classes=2,      # liczba klas
    random_state=42
)

#
# Porównanie metod z 5-krotną walidacją krzyżową

wyniki =  porownaj_metody_selekcji(X, y, n_cech=int(5), n_splits=int(5))