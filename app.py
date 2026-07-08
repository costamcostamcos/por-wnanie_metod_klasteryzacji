import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Algorytmy Partycjonujące i Gęstościowe
from sklearn.cluster import KMeans, DBSCAN, OPTICS, MeanShift, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import gaussian_filter, maximum_filter

st.set_page_config(page_title="Klasteryzacja Widm EPR", layout="wide")

# ==========================================
# NATYWNE IMPLEMENTACJE - TABELA 2 (Partycjonujące)
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
# NATYWNE IMPLEMENTACJE - TABELA 3 (Gęstościowe)
# ==========================================
def uruchom_denclue(X, bandwidth=0.5, threshold=0.05):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
    dens = np.exp(kde.score_samples(X))
    core_mask = dens > threshold
    labels = np.full(X.shape[0], -1) 
    
    if np.sum(core_mask) > 0:
        dist_matrix = pairwise_distances(X[core_mask])
        adj_matrix = (dist_matrix <= bandwidth).astype(int)
        n_components, labels_core = connected_components(csgraph=sp.csr_matrix(adj_matrix), directed=False)
        labels[core_mask] = labels_core
    return labels

def uruchom_rdbc(X, eps=0.5, min_samples=5):
    dist_matrix = pairwise_distances(X)
    adj = (dist_matrix <= eps).astype(int)
    core_points = np.sum(adj, axis=1) >= min_samples
    labels = np.full(X.shape[0], -1)
    if np.sum(core_points) > 0:
        adj_core = adj[core_points][:, core_points]
        n_components, labels_core = connected_components(csgraph=sp.csr_matrix(adj_core), directed=False)
        labels[core_points] = labels_core
        core_indices = np.where(core_points)[0]
        for i in range(X.shape[0]):
            if not core_points[i]:
                distances_to_cores = dist_matrix[i, core_indices]
                min_dist_idx = np.argmin(distances_to_cores)
                if distances_to_cores[min_dist_idx] <= eps:
                    labels[i] = labels_core[min_dist_idx]
    return labels

# ==========================================
# NATYWNE IMPLEMENTACJE - TABELA 4 (Siatkowe)
# ==========================================
def pca_grid_base(X, bins=15):
    if X.shape[1] >= 2: X_pca = PCA(n_components=2, random_state=42).fit_transform(X)
    else: X_pca = np.column_stack((X, np.zeros_like(X)))
    X_min, X_max = X_pca.min(axis=0), X_pca.max(axis=0)
    X_norm = (X_pca - X_min) / (X_max - X_min + 1e-9)
    coords = np.floor(X_norm * bins).astype(int)
    coords = np.clip(coords, 0, bins - 1)
    flat_coords = coords[:, 0] * bins + coords[:, 1]
    grid_2d = np.bincount(flat_coords, minlength=bins*bins).reshape((bins, bins))
    return coords, flat_coords, grid_2d

def uruchom_sting(X, bins=15):
    coords, flat_coords, grid_2d = pca_grid_base(X, bins)
    threshold = (X.shape[0] / (bins*bins)) * 0.5
    dense_cells = np.argwhere(grid_2d > threshold)
    if len(dense_cells) == 0: return np.full(X.shape[0], -1)
    cell_labels = DBSCAN(eps=1.5, min_samples=1).fit_predict(dense_cells)
    flat_to_label = np.full(bins*bins, -1)
    for idx, (i, j) in enumerate(dense_cells): flat_to_label[i * bins + j] = cell_labels[idx]
    return flat_to_label[flat_coords]

def uruchom_clique(X, bins=15):
    coords, flat_coords, grid_2d = pca_grid_base(X, bins)
    c0 = np.bincount(coords[:, 0], minlength=bins)
    c1 = np.bincount(coords[:, 1], minlength=bins)
    t0, t1 = np.mean(c0), np.mean(c1)
    labels = np.full(X.shape[0], -1)
    cluster_id = 0
    for i in np.where(c0 > t0)[0]:
        for j in np.where(c1 > t1)[0]:
            mask = (coords[:, 0] == i) & (coords[:, 1] == j)
            if np.any(mask):
                labels[mask] = cluster_id
                cluster_id += 1
    return labels

def uruchom_optigrid(X, bins=15):
    coords, flat_coords, grid_2d = pca_grid_base(X, bins)
    c0 = np.bincount(coords[:, 0], minlength=bins)
    c1 = np.bincount(coords[:, 1], minlength=bins)
    cut0, cut1 = np.argmin(c0), np.argmin(c1)
    labels = np.zeros(X.shape[0], dtype=int)
    labels[coords[:, 0] > cut0] += 1
    labels[coords[:, 1] > cut1] += 2
    return labels

def uruchom_gridclus(X, bins=15):
    coords, flat_coords, grid_2d = pca_grid_base(X, bins)
    populated = np.argwhere(grid_2d > 0)
    if len(populated) < 2: return np.zeros(X.shape[0])
    cell_labels = AgglomerativeClustering(n_clusters=min(3, len(populated))).fit_predict(populated)
    flat_to_label = np.full(bins*bins, -1)
    for idx, (i, j) in enumerate(populated): flat_to_label[i * bins + j] = cell_labels[idx]
    return flat_to_label[flat_coords]

def uruchom_gdilc(X, bins=15):
    coords, flat_coords, grid_2d = pca_grid_base(X, bins)
    local_max = maximum_filter(grid_2d, size=3) == grid_2d
    peaks = np.argwhere(local_max & (grid_2d > 0))
    if len(peaks) == 0: return np.full(X.shape[0], -1)
    populated = np.argwhere(grid_2d > 0)
    from sklearn.metrics import pairwise_distances_argmin
    cell_labels = pairwise_distances_argmin(populated, peaks)
    flat_to_label = np.full(bins*bins, -1)
    for idx, (i, j) in enumerate(populated): flat_to_label[i * bins + j] = cell_labels[idx]
    return flat_to_label[flat_coords]

def uruchom_wavecluster(X, bins=15):
    coords, flat_coords, grid_2d = pca_grid_base(X, bins)
    blurred = gaussian_filter(grid_2d.astype(float), sigma=1.0)
    dense_cells = np.argwhere(blurred > np.mean(blurred))
    if len(dense_cells) == 0: return np.full(X.shape[0], -1)
    cell_labels = DBSCAN(eps=1.5, min_samples=1).fit_predict(dense_cells)
    flat_to_label = np.full(bins*bins, -1)
    for idx, (i, j) in enumerate(dense_cells): flat_to_label[i * bins + j] = cell_labels[idx]
    return flat_to_label[flat_coords]

# ==========================================
# NATYWNE IMPLEMENTACJE - TABELA 5 (Rozmyte/Fuzzy)
# ==========================================
def uruchom_fcm(X, n_clusters, m=2.0, metric='euclidean', max_iter=150, tol=1e-5):
    """Implementacja Fuzzy C-Means (oraz wariantów). Zwraca twarde etykiety (defuzyfikacja)."""
    np.random.seed(42)
    n_samples = X.shape[0]
    
    # Inicjalizacja macierzy przynależności U (wartości od 0 do 1)
    U = np.random.dirichlet(np.ones(n_clusters), size=n_samples)
    
    for _ in range(max_iter):
        U_m = U ** m
        # Obliczanie nowych centrów rozmytych
        centers = (U_m.T @ X) / np.sum(U_m, axis=0)[:, None]
        
        # Odległość do centrów
        dist = pairwise_distances(X, centers, metric=metric)
        dist = np.fmax(dist, np.finfo(np.float64).eps) # Zabezpieczenie przed dzieleniem przez 0
        
        # Aktualizacja macierzy U
        temp = dist ** (-2 / (m - 1))
        U_new = temp / np.sum(temp, axis=1)[:, None]
        
        if np.linalg.norm(U_new - U) < tol:
            break
        U = U_new
        
    # Defuzyfikacja (wybieramy klaster o najwyższym prawdopodobieństwie)
    return np.argmax(U, axis=1)

def uruchom_mec(X, n_clusters, beta=1.0, max_iter=150):
    """Maximum Entropy Clustering (MEC) z regularyzacją entropijną (Soft K-Means)."""
    np.random.seed(42)
    # Losowa inicjalizacja centrów
    centers_idx = np.random.choice(X.shape[0], n_clusters, replace=False)
    centers = X[centers_idx]
    
    for _ in range(max_iter):
        dist = pairwise_distances(X, centers, metric='sqeuclidean')
        # Zastosowanie rozkładu Boltzmanna / Softmax
        temp = np.exp(-beta * dist)
        U = temp / np.sum(temp, axis=1)[:, None]
        
        new_centers = (U.T @ X) / np.sum(U, axis=0)[:, None]
        if np.allclose(centers, new_centers, atol=1e-5):
            break
        centers = new_centers
        
    return np.argmax(U, axis=1)

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
    wyniki['Mean-Shift'] = MeanShift(bandwidth=bandwidth).fit_predict(X_scaled)
    wyniki['DENCLUE'] = uruchom_denclue(X_scaled, bandwidth=bandwidth, threshold=0.01)
    wyniki['RDBC'] = uruchom_rdbc(X_scaled, eps=eps, min_samples=min_samples)
    return wyniki

def analizuj_siatkowe(X_scaled, bins):
    wyniki = {}
    wyniki['STING'] = uruchom_sting(X_scaled, bins=bins)
    wyniki['CLIQUE'] = uruchom_clique(X_scaled, bins=bins)
    wyniki['OptiGrid'] = uruchom_optigrid(X_scaled, bins=bins)
    wyniki['GRIDCLUS'] = uruchom_gridclus(X_scaled, bins=bins)
    wyniki['GDILC'] = uruchom_gdilc(X_scaled, bins=bins)
    wyniki['WaveCluster'] = uruchom_wavecluster(X_scaled, bins=bins)
    return wyniki

def analizuj_rozmyte(X_scaled, liczba_grup, m_fuzziness, beta_mec):
    wyniki = {}
    # Fuzzy k-means
    wyniki['Fuzzy k-means'] = uruchom_fcm(X_scaled, liczba_grup, m=m_fuzziness, metric='euclidean')
    # Fuzzy k-modes (przybliżenie przez zastosowanie odległości miejskiej L1 dla danych numerycznych)
    wyniki['Fuzzy k-modes'] = uruchom_fcm(X_scaled, liczba_grup, m=m_fuzziness, metric='manhattan')
    # FCM (Fuzzy C-Means to standardowa nazwa k-means)
    wyniki['FCM'] = uruchom_fcm(X_scaled, liczba_grup, m=m_fuzziness, metric='euclidean')
    
    # FCS (Fuzzy C-Spherical / Scatter Matrix) - Aproksymacja przez Probabilistyczny Gaussian Mixture (Spherical)
    gmm_fcs = GaussianMixture(n_components=liczba_grup, covariance_type='spherical', random_state=42)
    wyniki['FCS'] = gmm_fcs.fit_predict(X_scaled)
    
    # MM (Markov Model / Mixture Model) - GMM z diagonalną macierzą kowariancji
    gmm_mm = GaussianMixture(n_components=liczba_grup, covariance_type='diag', random_state=42)
    wyniki['MM'] = gmm_mm.fit_predict(X_scaled)
    
    # MEC (Maximum Entropy Clustering)
    wyniki['MEC'] = uruchom_mec(X_scaled, liczba_grup, beta=beta_mec)
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

st.sidebar.header("Wybór Metodologii")
rodzina_algorytmow = st.sidebar.radio(
    "Którą rodzinę algorytmów chcesz uruchomić?",
    ("Partycjonujące (Tab 2)", "Oparte na Gęstości (Tab 3)", "Oparte na Siatce (Tab 4)", "Rozmyte / Fuzzy (Tab 5)")
)

st.sidebar.markdown("---")
st.sidebar.header("Parametry algorytmów")

if rodzina_algorytmow == "Partycjonujące (Tab 2)":
    liczba_grup = st.sidebar.number_input("Liczba klastrów (K):", min_value=2, max_value=20, value=3)

elif rodzina_algorytmow == "Oparte na Gęstości (Tab 3)":
    st.sidebar.markdown("*Metody gęstościowe same znajdują optymalną liczbę klastrów.*")
    eps_val = st.sidebar.slider("Promień poszukiwań (eps):", min_value=0.1, max_value=20.0, value=5.0, step=0.1)
    min_samples_val = st.sidebar.number_input("Minimalna liczba punktów (min_samples):", min_value=2, max_value=50, value=3)
    bandwidth_val = st.sidebar.slider("Szerokość pasma (bandwidth):", min_value=0.1, max_value=20.0, value=2.0, step=0.1)

elif rodzina_algorytmow == "Oparte na Siatce (Tab 4)":
    st.sidebar.markdown("*Metody siatkowe redukują wymiarowość widm (PCA) i dzielą przestrzeń na bloki.*")
    bins_val = st.sidebar.slider("Rozdzielczość siatki (komórki):", min_value=5, max_value=50, value=15)

elif rodzina_algorytmow == "Rozmyte / Fuzzy (Tab 5)":
    st.sidebar.markdown("*Algorytmy rozmyte określają prawdopodobieństwo przynależności.*")
    liczba_grup = st.sidebar.number_input("Liczba klastrów (K):", min_value=2, max_value=20, value=3)
    m_fuzziness = st.sidebar.slider("Współczynnik rozmycia (m) dla FCM:", min_value=1.1, max_value=5.0, value=2.0, step=0.1)
    beta_mec = st.sidebar.slider("Parametr temperatury (\u03B2) dla MEC:", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("Ustawienia danych")
pomin_kolumne = st.sidebar.checkbox("Zignoruj pierwszą kolumnę (np. oś X widma)", value=False)
transpozycja = st.sidebar.checkbox("Transpozycja danych (widma w kolumnach)", value=False)
limit_osi_y = st.sidebar.number_input("Limit osi Y na wykresach (0 = auto):", min_value=0.0, max_value=1000.0, value=0.25, step=0.05)

st.sidebar.markdown("---")
st.sidebar.header("Ground Truth")
gt_indeks = st.sidebar.number_input("Indeks kolumny etykiet (0 = pierwsza):", min_value=0, max_value=10, value=0)

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
                    elif rodzina_algorytmow == "Oparte na Gęstości (Tab 3)":
                        wyniki = analizuj_gestosciowe(X_scaled, eps_val, min_samples_val, bandwidth_val)
                    elif rodzina_algorytmow == "Oparte na Siatce (Tab 4)":
                        wyniki = analizuj_siatkowe(X_scaled, bins_val)
                    elif rodzina_algorytmow == "Rozmyte / Fuzzy (Tab 5)":
                        wyniki = analizuj_rozmyte(X_scaled, liczba_grup, m_fuzziness, beta_mec)
                        
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
