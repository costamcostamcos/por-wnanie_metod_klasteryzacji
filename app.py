import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.neighbors import KernelDensity

# Algorytmy Partycjonujące (Tabela 2)
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

# Algorytmy Gęstościowe (Tabela 3)
from sklearn.cluster import DBSCAN, OPTICS, MeanShift
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

st.set_page_config(page_title="Klasteryzacja Widm EPR", layout="wide")

# ==========================================
# NATYWNE IMPLEMENTACJE - TABELA 2
# ==========================================
def uruchom_clara(X, liczba_grup, random_state=42):
    np.random.seed(random_state)
    n_total = len(X)
    n_samples = min(n_total, 40 + 2 * liczba_grup)
    najlepsze_medoidy = None
    najmniejszy_koszt = float('inf')
    dist_matrix_full = pairwise_distances(X)
    
    for _ in range(5): 
        probka_idx = np.random.choice(n_total, n_samples, replace=False)
        probka_X = X[probka_idx]
        pam = KMedoids(n_clusters=liczba_grup, method='pam', init='heuristic').fit(probka_X)
        medoidy_w_probce = probka_idx[pam.medoid_indices_] 
        koszt = np.sum(np.min(dist_matrix_full[:, medoidy_w_probce], axis=1))
        if koszt < najmniejszy_koszt:
            najmniejszy_koszt = koszt
            najlepsze_medoidy = medoidy_w_probce
            
    labels = np.argmin(dist_matrix_full[:, najlepsze_medoidy], axis=1)
    return labels

def uruchom_clarans(X, liczba_grup, numlocal=3, maxneighbor=5, random_state=42):
    np.random.seed(random_state)
    n_total = len(X)
    dist_matrix = pairwise_distances(X)
    najlepsze_medoidy = None
    najmniejszy_koszt = float('inf')
    
    for _ in range(numlocal):
        obecne_medoidy = list(np.random.choice(n_total, liczba_grup, replace=False))
        def oblicz_koszt(medoidy): return np.sum(np.min(dist_matrix[:, medoidy], axis=1))
        obecny_koszt = oblicz_koszt(obecne_medoidy)
        
        j = 0
        while j < maxneighbor:
            m_idx = np.random.randint(liczba_grup)
            stary_medoid = obecne_medoidy[m_idx]
            nowy_medoid = np.random.randint(n_total)
            while nowy_medoid in obecne_medoidy: nowy_medoid = np.random.randint(n_total)
                
            obecne_medoidy[m_idx] = nowy_medoid
            nowy_koszt = oblicz_koszt(obecne_medoidy)
            
            if nowy_koszt < obecny_koszt:
                obecny_koszt = nowy_koszt
                j = 0
            else:
                obecne_medoidy[m_idx] = stary_medoid
                j += 1
                
        if obecny_koszt < najmniejszy_koszt:
            najmniejszy_koszt = obecny_koszt
            najlepsze_medoidy = list(obecne_medoidy)
            
    labels = np.argmin(dist_matrix[:, najlepsze_medoidy], axis=1)
    return labels

# ==========================================
# NATYWNE IMPLEMENTACJE - TABELA 3
# ==========================================
def uruchom_denclue(X, bandwidth=0.5, threshold=0.05):
    """Natywna implementacja DENCLUE wykorzystująca Kernel Density Estimation (KDE)."""
    # 1. Estymacja gęstości przestrzennej dla każdego punktu
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
    log_dens = kde.score_samples(X)
    dens = np.exp(log_dens)
    
    # 2. Odfiltrowanie szumu (punkty poniżej progu gęstości)
    core_mask = dens > threshold
    labels = np.full(X.shape[0], -1) # Domyślnie szum
    
    if np.sum(core_mask) > 0:
        X_core = X[core_mask]
        dist_matrix = pairwise_distances(X_core)
        # 3. Łączenie gęstych obszarów (Atraktorów)
        adj_matrix = (dist_matrix <= bandwidth).astype(int)
        n_components, labels_core = connected_components(csgraph=sp.csr_matrix(adj_matrix), directed=False)
        labels[core_mask] = labels_core
        
    return labels

def uruchom_rdbc(X, eps=0.5, min_samples=5):
    """Natywna implementacja RDBC (Reachability Distance Based Clustering)."""
    dist_matrix = pairwise_distances(X)
    n = X.shape[0]
    
    # Punkty rdzeniowe
    adj = (dist_matrix <= eps).astype(int)
    core_points = np.sum(adj, axis=1) >= min_samples
    
    labels = np.full(n, -1)
    if np.sum(core_points) > 0:
        adj_core = adj[core_points][:, core_points]
        n_components, labels_core = connected_components(csgraph=sp.csr_matrix(adj_core), directed=False)
        labels[core_points] = labels_core
        
        # Przypisanie punktów brzegowych do najbliższego rdzenia
        core_indices = np.where(core_points)[0]
        for i in range(n):
            if not core_points[i]:
                distances_to_cores = dist_matrix[i, core_indices]
                min_dist_idx = np.argmin(distances_to_cores)
                if distances_to_cores[min_dist_idx] <= eps:
                    labels[i] = labels_core[min_dist_idx]
    return labels

# ==========================================
# GŁÓWNA LOGIKA APLIKACJI
# ==========================================

def wczytaj_i_przygotuj_dane(plik_excel, pomin_kolumne, transponuj, gt_indeks_kolumny):
    df = pd.read_excel(plik_excel, sheet_name=0)
    
    if pomin_kolumne: X_df = df.iloc[:, 1:]
    else: X_df = df
        
    if transponuj:
        X_df = X_df.T
        identyfikatory_widm = X_df.index.astype(str).tolist()
    else:
        if pomin_kolumne: identyfikatory_widm = df.iloc[:, 0].astype(str).tolist()
        else: identyfikatory_widm = [f"Widmo_{i+1}" for i in range(X_df.shape[0])]
        
    X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    X = X_df.values
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ground_truth_labels = None
    df_gt_preview = None
    try:
        df_gt = pd.read_excel(plik_excel, sheet_name='Ground Truth')
        df_gt_preview = df_gt
        ground_truth_labels = df_gt.iloc[:, gt_indeks_kolumny].values 
    except Exception: pass 
    
    return X, X_scaled, df, ground_truth_labels, df_gt_preview, identyfikatory_widm

def analizuj_partycjonujace(X_scaled, liczba_grup):
    wyniki = {}
    wyniki['K-Means'] = KMeans(n_clusters=liczba_grup, random_state=42).fit_predict(X_scaled)
    wyniki['K-Medoids'] = KMedoids(n_clusters=liczba_grup, method='alternate', random_state=42).fit_predict(X_scaled)
    wyniki['PAM'] = KMedoids(n_clusters=liczba_grup, method='pam', random_state=42).fit_predict(X_scaled)
    wyniki['CLARA'] = uruchom_clara(X_scaled, liczba_grup)
    wyniki['CLARANS'] = uruchom_clarans(X_scaled, liczba_grup)
    return wyniki

def analizuj_gestosciowe(X_scaled, eps, min_samples, bandwidth):
    wyniki = {}
    wyniki['DBSCAN'] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
    wyniki['OPTICS'] = OPTICS(min_samples=min_samples).fit_predict(X_scaled)
    # Mean-Shift nie wymaga z góry podanej liczby klastrów, korzysta z szerokości pasma (bandwidth)
    wyniki['Mean-Shift'] = MeanShift(bandwidth=bandwidth).fit_predict(X_scaled)
    wyniki['DENCLUE'] = uruchom_denclue(X_scaled, bandwidth=bandwidth, threshold=0.01)
    wyniki['RDBC'] = uruchom_rdbc(X_scaled, eps=eps, min_samples=min_samples)
    return wyniki

def generuj_wykres_srednich(X, etykiety, nazwa_algorytmu, limit_y=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    unikalne_etykiety = np.unique(etykiety)
    os_x = np.arange(X.shape[1])
    
    for etykieta in unikalne_etykiety:
        maska = (etykiety == etykieta)
        liczba_widm = np.sum(maska)
        if liczba_widm == 0: continue
            
        srednie_widmo = np.mean(X[maska], axis=0)
        odchylenie = np.std(X[maska], axis=0)
        
        if etykieta == -1:
            line = ax.plot(os_x, srednie_widmo, color='gray', linestyle='--', label=f'Szum (-1) [n={liczba_widm}]')
            ax.fill_between(os_x, srednie_widmo - odchylenie, srednie_widmo + odchylenie, color='gray', alpha=0.2)
        else:
            line = ax.plot(os_x, srednie_widmo, label=f'Klaster {etykieta} [n={liczba_widm}]')
            kolor = line[0].get_color()
            ax.fill_between(os_x, srednie_widmo - odchylenie, srednie_widmo + odchylenie, color=kolor, alpha=0.2)
            
    ax.set_title(f'Średnia reprezentacja klastrów: {nazwa_algorytmu}')
    ax.set_xlabel('Punkt pomiarowy')
    ax.set_ylabel('Średnie natężenie')
    if limit_y > 0.0:
        min_y = np.min(X)
        dolna_granica = min_y - (0.05 * abs(min_y)) if min_y < 0 else -0.05
        ax.set_ylim(bottom=dolna_granica, top=limit_y)
        
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    return fig

# --- INTERFEJS STREAMLIT ---
st.title("🔬 Analiza i Klasteryzacja Widm EPR")

st.sidebar.header("Wybór Metodyologii")
rodzina_algorytmow = st.sidebar.radio(
    "Którą rodzinę algorytmów chcesz uruchomić?",
    ("Partycjonujące (Tab 2)", "Oparte na Gęstości (Tab 3)")
)

st.sidebar.markdown("---")
st.sidebar.header("Parametry algorytmów")

if rodzina_algorytmow == "Partycjonujące (Tab 2)":
    liczba_grup = st.sidebar.number_input("Liczba klastrów (K):", min_value=2, max_value=20, value=3)
else:
    st.sidebar.markdown("*Uwaga: Metody gęstościowe same znajdują optymalną liczbę klastrów.*")
    eps_val = st.sidebar.slider("Promień poszukiwań (eps) - DBSCAN/RDBC:", min_value=0.1, max_value=20.0, value=5.0, step=0.1)
    min_samples_val = st.sidebar.number_input("Minimalna liczba punktów (min_samples) - DBSCAN/OPTICS:", min_value=2, max_value=50, value=3)
    bandwidth_val = st.sidebar.slider("Szerokość pasma (bandwidth) - Mean-Shift/DENCLUE:", min_value=0.1, max_value=20.0, value=2.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("Ustawienia danych głównego arkusza")
pomin_kolumne = st.sidebar.checkbox("Zignoruj pierwszą kolumnę (np. oś X widma)", value=False)
transpozycja = st.sidebar.checkbox("Transpozycja danych (widma w kolumnach)", value=False)
limit_osi_y = st.sidebar.number_input("Maksymalna wartość osi Y na wykresach (0 = auto):", min_value=0.0, max_value=1000.0, value=0.25, step=0.05)

st.sidebar.markdown("---")
st.sidebar.header("Ustawienia Ground Truth")
gt_indeks = st.sidebar.number_input("Indeks kolumny z etykietami klastrów (0 = pierwsza):", min_value=0, max_value=10, value=0)

wgrany_plik = st.file_uploader("Wybierz plik Excel (.xlsx)", type=['xlsx'])

if wgrany_plik is not None:
    try:
        with st.spinner('Wczytywanie i przygotowywanie danych...'):
            X, X_scaled, oryginalny_df, gt_labels, df_gt_preview, identyfikatory = wczytaj_i_przygotuj_dane(
                wgrany_plik, pomin_kolumne, transpozycja, gt_indeks
            )
            
        st.success(f"Dane wczytano! Główne dane: {X.shape[0]} widm, {X.shape[1]} punktów pomiarowych.")
        
        if gt_labels is not None:
            st.info("✅ Wykryto arkusz 'Ground Truth'.")
            with st.expander("Kliknij, aby rozwinąć PODGLĄD GROUND TRUTH"):
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(df_gt_preview.head(10))
                with col2:
                    st.write(gt_labels[:10])
                    st.markdown(f"*Liczba unikalnych etykiet: {len(np.unique(gt_labels))}*")
        
        if st.button("Uruchom Klasteryzację", type="primary"):
            if gt_labels is not None and len(gt_labels) != X.shape[0]:
                st.error(f"❌ Błąd zgodności! Ground Truth (ilość: {len(gt_labels)}) nie pasuje do badanych widm ({X.shape[0]}).")
            else:
                with st.spinner('Trwa obliczanie klastrów...'):
                    
                    if rodzina_algorytmow == "Partycjonujące (Tab 2)":
                        wyniki = analizuj_partycjonujace(X_scaled, liczba_grup)
                    else:
                        wyniki = analizuj_gestosciowe(X_scaled, eps_val, min_samples_val, bandwidth_val)
                        
                    wykresy = {}
                    wyniki_ewaluacji = []
                    
                    st.subheader("Skład poszczególnych klastrów")
                    
                    for nazwa_algorytmu, etykiety in wyniki.items():
                        with st.expander(f"Rozkład widm: {nazwa_algorytmu}"):
                            unikalne = np.unique(etykiety)
                            for etyk in unikalne:
                                indeksy_w_klastrze = np.where(etykiety == etyk)[0]
                                id_widm_w_klastrze = [identyfikatory[i] for i in indeksy_w_klastrze]
                                nazwa_kat = f"Klaster {etyk}" if etyk != -1 else "Szum (-1)"
                                st.markdown(f"**{nazwa_kat}** (Sztuk: {len(indeksy_w_klastrze)}): {', '.join(id_widm_w_klastrze)}")
                        
                        fig = generuj_wykres_srednich(X, etykiety, nazwa_algorytmu, limit_osi_y)
                        wykresy[nazwa_algorytmu] = fig
                        
                        if gt_labels is not None:
                            ari_score = adjusted_rand_score(gt_labels, etykiety)
                            wyniki_ewaluacji.append({
                                "Algorytm": nazwa_algorytmu,
                                "ARI (Adjusted Rand Index)": round(ari_score, 4)
                            })
                    
                    if gt_labels is not None:
                        st.subheader("Wyniki Ewaluacji (porównanie z Ground Truth)")
                        df_ewaluacja = pd.DataFrame(wyniki_ewaluacji).sort_values(by="ARI (Adjusted Rand Index)", ascending=False)
                        st.dataframe(df_ewaluacja, use_container_width=True)
    
                    st.subheader("Wizualizacja średnich widm z pasmem błędu")
                    for nazwa, fig in wykresy.items():
                        st.pyplot(fig)
                
                bufor = io.BytesIO()
                with pd.ExcelWriter(bufor, engine='xlsxwriter') as writer:
                    wyniki_df = pd.DataFrame(X)
                    if transpozycja: wyniki_df.index = identyfikatory
                    
                    for algorytm, etyk in wyniki.items():
                        wyniki_df[f'Klaster_{algorytm}'] = etyk
                    
                    wyniki_df.to_excel(writer, sheet_name='Sklasyfikowane_Dane')
                    if gt_labels is not None: df_ewaluacja.to_excel(writer, sheet_name='Ewaluacja', index=False)
                    
                    workbook  = writer.book
                    worksheet = workbook.add_worksheet('Wykresy_Klastrow')
                    wiersz_start = 1
                    for nazwa, fig in wykresy.items():
                        img_data = io.BytesIO()
                        fig.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
                        img_data.seek(0)
                        worksheet.insert_image(f'B{wiersz_start}', nazwa, {'image_data': img_data})
                        wiersz_start += 28
    
                st.download_button(
                    label="⬇️ Pobierz plik Excel (Wyniki)",
                    data=bufor.getvalue(),
                    file_name="sklasyfikowane_widma.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
    except Exception as e:
        st.error(f"Wystąpił błąd podczas przetwarzania pliku: {e}")
