import streamlit as st
import pandas as pd
import numpy as np
import io
import traceback
import matplotlib
matplotlib.use("Agg")  # backend bez GUI – bezpieczny w Streamlit i przy zapisie do Excela
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Algorytmy Partycjonujące i Gęstościowe
from sklearn.cluster import KMeans, DBSCAN, OPTICS, MeanShift, AgglomerativeClustering, Birch
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import gaussian_filter, maximum_filter

# ==========================================
# BEZPIECZNY IMPORT KMEDOIDS
# scikit-learn-extra (0.3.0, marzec 2023) jest niekompatybilny z NumPy >= 2.0
# (rozszerzenia C skompilowane pod NumPy 1.x). Na nowym środowisku import wywala
# całą aplikację. Poniżej: próba importu, a w razie porażki natywny fallback
# zgodny z interfejsem używanym w aplikacji (fit / fit_predict / medoid_indices_).
# ==========================================
try:
    from sklearn_extra.cluster import KMedoids
    ZRODLO_KMEDOIDS = "sklearn_extra"
except Exception:
    ZRODLO_KMEDOIDS = "natywna (fallback)"

    class KMedoids:
        """Lekka natywna implementacja k-medoids (metody 'pam' i 'alternate').
        Interfejs zgodny z sklearn_extra w zakresie używanym przez aplikację."""

        def __init__(self, n_clusters=8, metric="euclidean", method="alternate",
                     init="heuristic", max_iter=300, random_state=None):
            self.n_clusters = n_clusters
            self.metric = metric
            self.method = method
            self.init = init
            self.max_iter = max_iter
            self.random_state = random_state

        def _init_medoidy(self, D, rng):
            n = D.shape[0]
            if self.init == "heuristic":
                # punkty najbliżej "środka" – o najmniejszej sumie odległości
                return list(np.argsort(D.sum(axis=1))[: self.n_clusters])
            return list(rng.choice(n, self.n_clusters, replace=False))

        @staticmethod
        def _koszt(D, medoidy):
            lab = np.argmin(D[:, medoidy], axis=1)
            n = D.shape[0]
            return D[np.arange(n), np.array(medoidy)[lab]].sum(), lab

        def fit(self, X):
            rng = np.random.RandomState(self.random_state)
            D = pairwise_distances(X, metric=self.metric)
            n = D.shape[0]
            medoidy = self._init_medoidy(D, rng)
            najlepszy_koszt, labels = self._koszt(D, medoidy)

            for _ in range(self.max_iter):
                zmiana = False
                if self.method == "pam":
                    for mi in range(self.n_clusters):
                        for kandydat in range(n):
                            if kandydat in medoidy:
                                continue
                            nowe = list(medoidy)
                            nowe[mi] = kandydat
                            koszt, lab = self._koszt(D, nowe)
                            if koszt < najlepszy_koszt - 1e-12:
                                najlepszy_koszt, medoidy, labels = koszt, nowe, lab
                                zmiana = True
                else:  # 'alternate' – medoid = punkt minimalizujący sumę odległości w klastrze
                    labels = np.argmin(D[:, medoidy], axis=1)
                    nowe = list(medoidy)
                    for k in range(self.n_clusters):
                        idx = np.where(labels == k)[0]
                        if len(idx) == 0:
                            continue
                        nowe[k] = idx[np.argmin(D[np.ix_(idx, idx)].sum(axis=1))]
                    if nowe != medoidy:
                        medoidy = nowe
                        zmiana = True
                    najlepszy_koszt, labels = self._koszt(D, medoidy)
                if not zmiana:
                    break

            self.medoid_indices_ = np.array(medoidy)
            self.labels_ = np.argmin(D[:, medoidy], axis=1)
            self.inertia_ = najlepszy_koszt
            return self

        def fit_predict(self, X):
            return self.fit(X).labels_


st.set_page_config(page_title="Klasteryzacja Widm EPR - Ranking Globalny", layout="wide")

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
def uruchom_fcm(X, n_clusters, m=2.0, metric='euclidean', max_iter=150, tol=1e-5, seed=42):
    np.random.seed(seed)
    n_samples = X.shape[0]
    U = np.random.dirichlet(np.ones(n_clusters), size=n_samples)

    for _ in range(max_iter):
        U_m = U ** m
        centers = (U_m.T @ X) / np.fmax(np.sum(U_m, axis=0)[:, None], 1e-10)
        dist = pairwise_distances(X, centers, metric=metric)

        U_new = np.zeros_like(U)
        for i in range(n_samples):
            if np.any(dist[i] < 1e-8):
                U_new[i, np.argmin(dist[i])] = 1.0
            else:
                inv_dist = dist[i] ** (-2 / (m - 1))
                U_new[i] = inv_dist / np.sum(inv_dist)

        if np.linalg.norm(U_new - U) < tol:
            break
        U = U_new

    return np.argmax(U, axis=1)

def uruchom_mec(X, n_clusters, beta=1.0, max_iter=150):
    np.random.seed(42)
    centers_idx = np.random.choice(X.shape[0], n_clusters, replace=False)
    centers = X[centers_idx]

    for _ in range(max_iter):
        dist = pairwise_distances(X, centers, metric='sqeuclidean')
        dist_min = np.min(dist, axis=1, keepdims=True)
        temp = np.exp(-beta * (dist - dist_min))

        U = temp / np.fmax(np.sum(temp, axis=1)[:, None], 1e-10)
        new_centers = (U.T @ X) / np.fmax(np.sum(U, axis=0)[:, None], 1e-10)

        if np.allclose(centers, new_centers, atol=1e-5):
            break
        centers = new_centers

    return np.argmax(U, axis=1)

# ==========================================
# NATYWNE IMPLEMENTACJE - TABELA 1 (Hierarchiczne)
# SLINK / CLINK / BIRCH -> sklearn (single/complete linkage, Birch).
# CURE / ROCK / Chameleon / DIANA / DISMEA -> implementacje natywne poniżej.
# Uwaga: metody hierarchiczne wymagają zadanej liczby klastrów K.
# ==========================================
def uruchom_cure(X, liczba_grup, n_repr=5, shrink=0.3):
    """CURE: każdy klaster reprezentowany przez zbiór rozproszonych punktów
    obkurczonych w stronę centroidu; łączenie pary o najbliższych reprezentantach."""
    n = X.shape[0]
    clusters = {i: [i] for i in range(n)}
    reps = {i: X[i:i+1].astype(float).copy() for i in range(n)}
    while len(clusters) > liczba_grup:
        keys = list(clusters.keys())
        min_d, pair = np.inf, None
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                d = pairwise_distances(reps[keys[i]], reps[keys[j]]).min()
                if d < min_d:
                    min_d, pair = d, (keys[i], keys[j])
        a, b = pair
        merged = clusters[a] + clusters[b]
        pts = X[merged]
        center = pts.mean(axis=0)
        # farthest-first: wybór rozproszonych reprezentantów
        chosen = [merged[int(np.argmax(np.linalg.norm(pts - center, axis=1)))]]
        while len(chosen) < min(n_repr, len(merged)):
            d2 = np.min(pairwise_distances(X[merged], X[chosen]), axis=1)
            chosen.append(merged[int(np.argmax(d2))])
        rep_pts = X[chosen].astype(float)
        rep_pts = rep_pts + shrink * (center - rep_pts)  # obkurczanie w stronę centroidu
        nk = max(clusters) + 1
        clusters[nk] = merged
        reps[nk] = rep_pts
        del clusters[a]; del clusters[b]; del reps[a]; del reps[b]
    labels = np.empty(n, dtype=int)
    for lab, members in enumerate(clusters.values()):
        for m in members:
            labels[m] = lab
    return labels

def uruchom_rock(X, liczba_grup, theta=0.5, k_neighbors=None):
    """ROCK (adaptacja do danych numerycznych): sąsiedztwo = graf k-NN,
    link(a,b) = liczba wspólnych sąsiadów, łączenie wg miary 'goodness' z f(theta).
    Oryginalnie metoda dla danych Boolean/Categorical - dla widm używamy grafu k-NN,
    co daje strukturę linków stabilną niezależnie od skali danych."""
    n = X.shape[0]
    if k_neighbors is None:
        k_neighbors = max(2, int(np.sqrt(n)))
    k_neighbors = min(k_neighbors, n - 1)
    A = kneighbors_graph(X, n_neighbors=k_neighbors, mode='connectivity').toarray()
    neigh = np.maximum(A, A.T).astype(int)  # symetryzacja grafu sąsiedztwa
    link = neigh @ neigh.T                  # liczba wspólnych sąsiadów
    np.fill_diagonal(link, 0)
    clusters = {i: [i] for i in range(n)}
    f = 1 + 2 * (1 - theta) / (1 + theta)
    def goodness(a, b):
        la, lb = clusters[a], clusters[b]
        cross = link[np.ix_(la, lb)].sum()
        na, nb = len(la), len(lb)
        denom = (na + nb) ** f - na ** f - nb ** f
        return cross / denom if denom > 0 else 0.0
    while len(clusters) > liczba_grup:
        keys = list(clusters.keys())
        best_g, pair = 0.0, None
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                g = goodness(keys[i], keys[j])
                if g > best_g:
                    best_g, pair = g, (keys[i], keys[j])
        if pair is None:
            break  # brak linków - zatrzymanie (może pozostać > K klastrów, zgodnie z naturą ROCK)
        a, b = pair
        nk = max(clusters) + 1
        clusters[nk] = clusters[a] + clusters[b]
        del clusters[a]; del clusters[b]
    labels = np.full(n, -1)
    for lab, members in enumerate(clusters.values()):
        for m in members:
            labels[m] = lab
    return labels

def uruchom_chameleon(X, liczba_grup, k_nn=5, alpha=2.0):
    """Chameleon (uproszczony): wstępna partycja na drobne klastry, następnie
    aglomeracyjne łączenie wg Względnej Wzajemnej Łączności (RI) i Względnej
    Bliskości (RC) liczonych na grafie k-NN."""
    n = X.shape[0]
    k_nn = min(k_nn, n - 1)
    knn = kneighbors_graph(X, n_neighbors=k_nn, mode='distance').toarray()
    W = np.where(knn > 0, 1.0 / (1.0 + knn), 0.0)
    W = np.maximum(W, W.T)  # symetryczne wagi (podobieństwo)
    m = min(max(liczba_grup * 3, liczba_grup + 1), n)
    sub = AgglomerativeClustering(n_clusters=m).fit_predict(X)
    clusters = {c: list(np.where(sub == c)[0]) for c in np.unique(sub)}

    def cross_w(a, b):
        blk = W[np.ix_(clusters[a], clusters[b])]
        return blk[blk > 0]
    def intern_w(members):
        blk = W[np.ix_(members, members)]
        iu = np.triu_indices(len(members), 1)
        w = blk[iu]
        return w[w > 0]

    while len(clusters) > liczba_grup:
        keys = list(clusters.keys())
        best, pair = -1.0, None
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                cw = cross_w(a, b)
                if len(cw) == 0:
                    continue
                ia, ib = intern_w(clusters[a]), intern_w(clusters[b])
                ECa = ia.sum() if len(ia) else 1e-9
                ECb = ib.sum() if len(ib) else 1e-9
                RI = cw.sum() / ((ECa + ECb) / 2 + 1e-9)         # Względna Łączność
                Sab = cw.mean()
                Sa = ia.mean() if len(ia) else Sab
                Sb = ib.mean() if len(ib) else Sab
                na, nb = len(clusters[a]), len(clusters[b])
                RC = Sab / ((na / (na + nb)) * Sa + (nb / (na + nb)) * Sb + 1e-9)  # Względna Bliskość
                score = RI * (RC ** alpha)
                if score > best:
                    best, pair = score, (a, b)
        if pair is None:
            # brak krawędzi między klastrami w grafie - łączymy najbliższe geometrycznie
            keys = list(clusters.keys())
            cen = {c: X[clusters[c]].mean(axis=0) for c in keys}
            bd, pair = np.inf, (keys[0], keys[1])
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    d = np.linalg.norm(cen[keys[i]] - cen[keys[j]])
                    if d < bd:
                        bd, pair = d, (keys[i], keys[j])
        a, b = pair
        nk = max(clusters) + 1
        clusters[nk] = clusters[a] + clusters[b]
        del clusters[a]; del clusters[b]
    labels = np.empty(n, dtype=int)
    for lab, members in enumerate(clusters.values()):
        for mm in members:
            labels[mm] = lab
    return labels

def uruchom_diana(X, liczba_grup):
    """DIANA (Divisive Analysis): iteracyjny podział klastra o największej średnicy
    metodą 'splinter group' Macnaughton-Smitha."""
    D = pairwise_distances(X)
    labels = np.zeros(X.shape[0], dtype=int)
    while len(np.unique(labels)) < liczba_grup:
        best_c, best_diam = None, -1.0
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            if len(idx) < 2:
                continue
            diam = D[np.ix_(idx, idx)].max()
            if diam > best_diam:
                best_diam, best_c = diam, c
        if best_c is None:
            break
        idx = np.where(labels == best_c)[0]
        sub_D = D[np.ix_(idx, idx)]
        splinter = [int(np.argmax(sub_D.mean(axis=1)))]  # punkt najbardziej odstający
        reszta = [i for i in range(len(idx)) if i not in splinter]
        moved = True
        while moved and len(reszta) > 1:
            moved = False
            best_p, best_diff = None, 1e-12
            for p in reszta:
                d_spl = np.mean([sub_D[p, s] for s in splinter])
                d_rest = np.mean([sub_D[p, r] for r in reszta if r != p])
                if (d_rest - d_spl) > best_diff:
                    best_diff, best_p = d_rest - d_spl, p
            if best_p is not None:
                splinter.append(best_p)
                reszta.remove(best_p)
                moved = True
        labels[idx[splinter]] = labels.max() + 1
    return labels

def uruchom_dismea(X, liczba_grup, random_state=42):
    """DISMEA: podziałowy - iteracyjnie dzieli najbardziej rozproszony klaster
    (największe SSE) algorytmem 2-średnich, aż do uzyskania K klastrów."""
    labels = np.zeros(X.shape[0], dtype=int)
    while len(np.unique(labels)) < liczba_grup:
        best_c, best_sse = None, -1.0
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            if len(idx) < 2:
                continue
            sse = ((X[idx] - X[idx].mean(axis=0)) ** 2).sum()
            if sse > best_sse:
                best_sse, best_c = sse, c
        if best_c is None:
            break
        idx = np.where(labels == best_c)[0]
        sub = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit_predict(X[idx])
        labels[idx[sub == 1]] = labels.max() + 1
    return labels

# ==========================================
# GŁÓWNA LOGIKA APLIKACJI
# ==========================================

# ==========================================
# AUTOMATYCZNE WYKRYWANIE USTAWIEŃ DANYCH
# ==========================================
_SLOWA_OSI = ('b', 'field', 'pole', 'gauss', 'mt', 'g', 'x', 'os', 'oś', 'ppm',
              'index', 'indeks', 'channel', 'kanal', 'kanał', 'point', 'punkt',
              'freq', 'czest', 'id', 'sample', 'probka', 'próbka', 'nr', 'no')

def _czy_os_pomiarowa(seria):
    """Czy wektor wygląda jak oś pomiarowa/indeks: ściśle monotoniczny
    i (niemal) równomiernie rozłożony (stały krok)."""
    v = pd.to_numeric(seria, errors='coerce').dropna().values
    if len(v) < 3:
        return False
    diffs = np.diff(v)
    if np.all(diffs > 0) or np.all(diffs < 0):
        krok = np.abs(diffs)
        cv = krok.std() / (krok.mean() + 1e-12)  # współczynnik zmienności kroku
        return cv < 0.05
    return False

def wykryj_ustawienia_danych(df):
    """Na podstawie surowego arkusza wykrywa: czy pominąć pierwszą kolumnę
    (oś X / ID) oraz czy transponować (widma w kolumnach). Zwraca (pomin, transp, powody)."""
    powody = []
    n_wierszy, n_kolumn = df.shape

    naglowek = str(df.columns[0]).strip().lower()
    naglowek_ok = any(naglowek == s or naglowek.startswith(s) for s in _SLOWA_OSI)
    kol_monotoniczna = _czy_os_pomiarowa(df.iloc[:, 0])
    pomin = bool(kol_monotoniczna or naglowek_ok)

    if kol_monotoniczna:
        powody.append("pierwsza kolumna jest monotoniczna i równomierna → traktowana jako oś X / indeks")
    elif naglowek_ok:
        powody.append(f"nagłówek pierwszej kolumny ('{df.columns[0]}') wskazuje na oś lub identyfikator")
    else:
        powody.append("pierwsza kolumna zawiera dane widmowe → zachowana")

    dane_kolumny = n_kolumn - (1 if pomin else 0)
    # Oś pomiarowa (dużo punktów) powinna trafić do KOLUMN. Jeśli wierszy jest
    # wyraźnie więcej niż kolumn danych, widma są prawdopodobnie w kolumnach.
    if n_wierszy > dane_kolumny * 1.5:
        transp = True
        powody.append(f"{n_wierszy} wierszy > {dane_kolumny} kolumn danych → widma w kolumnach (transpozycja)")
    else:
        transp = False
        powody.append(f"{dane_kolumny} kolumn danych ≥ {n_wierszy} wierszy → widma w wierszach (bez transpozycji)")

    return pomin, transp, powody

def sugeruj_limit_y(X):
    """Robustowy górny limit osi Y: 99. percentyl wartości z zapasem, odporny na
    pojedyncze wartości odstające. Zwraca 0.0 (= autoskalowanie) gdy brak sensownej granicy."""
    vals = X[np.isfinite(X)]
    if vals.size == 0:
        return 0.0
    gora = np.percentile(vals, 99.0)
    if gora <= 0:
        return 0.0
    return float(round(gora * 1.15, 4))

def wczytaj_i_przygotuj_dane(plik_excel, pomin_kolumne, transponuj, gt_indeks_kolumny):
    try:
        plik_excel.seek(0)
    except Exception:
        pass
    df = pd.read_excel(plik_excel, sheet_name=0)
    if pomin_kolumne: X_df = df.iloc[:, 1:]
    else: X_df = df

    if transponuj:
        X_df = X_df.T
        identyfikatory_widm = X_df.index.astype(str).tolist()
    else:
        if pomin_kolumne: identyfikatory_widm = df.iloc[:, 0].astype(str).tolist()
        else: identyfikatory_widm = [f"Widmo_{i+1}" for i in range(X_df.shape[0])]

    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    X = X_df.values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    ground_truth_labels = None
    df_gt_preview = None
    ostrzezenie_gt = None
    try:
        try:
            plik_excel.seek(0)
        except Exception:
            pass
        df_gt = pd.read_excel(plik_excel, sheet_name='Ground Truth')
        df_gt_preview = df_gt
        # Walidacja indeksu kolumny etykiet – wcześniej błąd był cicho połykany,
        # przez co GT znikało bez informacji dla użytkownika.
        if gt_indeks_kolumny >= df_gt.shape[1]:
            ostrzezenie_gt = (
                f"Arkusz 'Ground Truth' ma {df_gt.shape[1]} kolumn, "
                f"a wybrany indeks etykiet to {gt_indeks_kolumny}. Etykiety zignorowane."
            )
        else:
            ground_truth_labels = df_gt.iloc[:, gt_indeks_kolumny].values
    except ValueError:
        pass  # brak arkusza 'Ground Truth' – to normalna, dozwolona sytuacja
    except Exception as e:
        ostrzezenie_gt = f"Nie udało się wczytać 'Ground Truth': {e}"

    return X, X_scaled, df, ground_truth_labels, df_gt_preview, identyfikatory_widm, ostrzezenie_gt

def analizuj_hierarchiczne(X_scaled, liczba_grup, birch_threshold=0.5, cure_repr=5,
                            cure_shrink=0.3, rock_theta=0.5, cham_knn=5):
    wyniki = {}
    wyniki['SLINK'] = AgglomerativeClustering(n_clusters=liczba_grup, linkage='single').fit_predict(X_scaled)
    wyniki['CLINK'] = AgglomerativeClustering(n_clusters=liczba_grup, linkage='complete').fit_predict(X_scaled)
    wyniki['BIRCH'] = Birch(n_clusters=liczba_grup, threshold=birch_threshold).fit_predict(X_scaled)
    wyniki['CURE'] = uruchom_cure(X_scaled, liczba_grup, n_repr=cure_repr, shrink=cure_shrink)
    wyniki['ROCK'] = uruchom_rock(X_scaled, liczba_grup, theta=rock_theta)
    wyniki['Chameleon'] = uruchom_chameleon(X_scaled, liczba_grup, k_nn=cham_knn)
    wyniki['DIANA'] = uruchom_diana(X_scaled, liczba_grup)
    wyniki['DISMEA'] = uruchom_dismea(X_scaled, liczba_grup)
    return wyniki

def analizuj_partycjonujace(X_scaled, liczba_grup):
    wyniki = {}
    wyniki['K-Means'] = KMeans(n_clusters=liczba_grup, random_state=42, n_init=10).fit_predict(X_scaled)
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
    # UWAGA metodologiczna: FCM (Fuzzy C-Means) to ten sam algorytm co "Fuzzy k-means".
    # W oryginale oba liczono IDENTYCZNIE (euclidean, ten sam seed) -> dwa takie same
    # wiersze w rankingu. Aby wiersze nie były bit-w-bit identyczne (co myli ewaluację),
    # FCM inicjalizowany jest innym ziarnem. Jeśli chcesz je scalić - usuń jeden wpis.
    wyniki['Fuzzy k-means'] = uruchom_fcm(X_scaled, liczba_grup, m=m_fuzziness, metric='euclidean', seed=42)
    wyniki['Fuzzy k-modes'] = uruchom_fcm(X_scaled, liczba_grup, m=m_fuzziness, metric='manhattan', seed=42)
    wyniki['FCM'] = uruchom_fcm(X_scaled, liczba_grup, m=m_fuzziness, metric='euclidean', seed=7)
    gmm_fcs = GaussianMixture(n_components=liczba_grup, covariance_type='spherical', random_state=42, reg_covar=1e-4)
    wyniki['FCS'] = gmm_fcs.fit_predict(X_scaled)
    gmm_mm = GaussianMixture(n_components=liczba_grup, covariance_type='diag', random_state=42, reg_covar=1e-4)
    wyniki['MM'] = gmm_mm.fit_predict(X_scaled)
    wyniki['MEC'] = uruchom_mec(X_scaled, liczba_grup, beta=beta_mec)
    return wyniki

def generuj_wykres_srednich(X, etykiety, nazwa_algorytmu, limit_y=0.0):
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
    # Zabezpieczenie: limit_y może być None -> unikamy TypeError przy porównaniu.
    if limit_y is not None and limit_y > 0.0:
        min_y = np.min(X)
        dolna_granica = min_y - (0.05 * abs(min_y)) if min_y < 0 else -0.05
        ax.set_ylim(bottom=dolna_granica, top=limit_y)

    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    return fig

# --- INTERFEJS STREAMLIT ---
st.title("🔬 Analiza i Klasteryzacja Widm EPR")
st.caption(f"Źródło implementacji K-Medoids/PAM/CLARA: **{ZRODLO_KMEDOIDS}**")

st.sidebar.header("Wybór Metodologii")
rodzina_algorytmow = st.sidebar.radio(
    "Którą procedurę chcesz uruchomić?",
    (
        "Hierarchiczne (Tab 1)",
        "Partycjonujące (Tab 2)",
        "Oparte na Gęstości (Tab 3)",
        "Oparte na Siatce (Tab 4)",
        "Rozmyte / Fuzzy (Tab 5)",
        "🔥 WSZYSTKIE NA RAZ (Globalny Ranking)"
    )
)

st.sidebar.markdown("---")
st.sidebar.header("Parametry algorytmów")

# Zmienne domyślne dla bezpieczeństwa
liczba_grup = 5
eps_val, min_samples_val, bandwidth_val = 5.0, 3, 2.0
bins_val = 15
m_fuzziness, beta_mec = 2.0, 1.0
birch_threshold, cure_repr, cure_shrink, rock_theta, cham_knn = 0.5, 5, 0.3, 0.5, 5

if rodzina_algorytmow == "Hierarchiczne (Tab 1)":
    st.sidebar.markdown("*Metody hierarchiczne (aglomeracyjne i podziałowe) budują drzewo klastrów; wymagają zadanej liczby K.*")
    liczba_grup = st.sidebar.number_input("Liczba klastrów (K):", min_value=2, max_value=20, value=3)
    with st.sidebar.expander("Parametry szczegółowe (BIRCH / CURE / ROCK / Chameleon)"):
        birch_threshold = st.slider("BIRCH – próg (threshold):", 0.1, 5.0, 0.5, 0.1)
        cure_repr = st.number_input("CURE – liczba reprezentantów:", 1, 20, 5)
        cure_shrink = st.slider("CURE – współczynnik obkurczania:", 0.0, 1.0, 0.3, 0.05)
        rock_theta = st.slider("ROCK – próg podobieństwa (\u03B8):", 0.1, 0.95, 0.5, 0.05)
        cham_knn = st.slider("Chameleon – liczba sąsiadów k-NN:", 2, 30, 5)

elif rodzina_algorytmow == "Partycjonujące (Tab 2)":
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

elif rodzina_algorytmow == "🔥 WSZYSTKIE NA RAZ (Globalny Ranking)":
    st.sidebar.markdown("**Zestawienie Globalne (Dostęp do wszystkich parametrów)**")
    liczba_grup = st.sidebar.number_input("Liczba klastrów (K) dla Metod Hierarchicznych, Partycjonujących i Rozmytych:", min_value=2, max_value=20, value=5)

    with st.sidebar.expander("Parametry Metod Hierarchicznych"):
        birch_threshold = st.slider("BIRCH – próg (threshold):", 0.1, 5.0, 0.5, 0.1)
        cure_repr = st.number_input("CURE – liczba reprezentantów:", 1, 20, 5)
        cure_shrink = st.slider("CURE – współczynnik obkurczania:", 0.0, 1.0, 0.3, 0.05)
        rock_theta = st.slider("ROCK – próg podobieństwa (\u03B8):", 0.1, 0.95, 0.5, 0.05)
        cham_knn = st.slider("Chameleon – liczba sąsiadów k-NN:", 2, 30, 5)

    with st.sidebar.expander("Parametry Metod Gęstościowych"):
        eps_val = st.slider("Promień poszukiwań (eps):", 0.1, 20.0, 5.0, 0.1)
        min_samples_val = st.number_input("Minimalna liczba punktów:", 2, 50, 3)
        bandwidth_val = st.slider("Szerokość pasma (bandwidth):", 0.1, 20.0, 2.0, 0.1)

    with st.sidebar.expander("Parametry Metod Siatkowych"):
        bins_val = st.slider("Rozdzielczość siatki (komórki):", 5, 50, 15)

    with st.sidebar.expander("Parametry Metod Rozmytych (poza K)"):
        m_fuzziness = st.slider("Współczynnik rozmycia (m) dla FCM:", 1.1, 5.0, 2.0, 0.1)
        beta_mec = st.slider("Parametr temperatury (\u03B2) dla MEC:", 0.1, 10.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("Ustawienia danych")
tryb_ustawien_auto = st.sidebar.checkbox(
    "🤖 Automatycznie dopasuj ustawienia danych",
    value=True,
    help="Aplikacja sama wykryje oś X w pierwszej kolumnie, orientację widm "
         "(transpozycję) i zaproponuje limit osi Y. Odznacz, aby ustawić ręcznie."
)

# Wartości domyślne; w trybie auto zostaną nadpisane po wczytaniu pliku.
pomin_kolumne, transpozycja, limit_osi_y = False, False, 0.25

if not tryb_ustawien_auto:
    pomin_kolumne = st.sidebar.checkbox("Zignoruj pierwszą kolumnę (np. oś X widma)", value=False)
    transpozycja = st.sidebar.checkbox("Transpozycja danych (widma w kolumnach)", value=False)
    limit_osi_y = st.sidebar.number_input("Limit osi Y na wykresach (0 = auto):", min_value=0.0, max_value=1000.0, value=0.25, step=0.05)
else:
    st.sidebar.caption("Ustawienia zostaną wykryte po wczytaniu pliku i pokazane w panelu głównym.")

st.sidebar.markdown("---")
st.sidebar.header("Ground Truth")
gt_indeks = st.sidebar.number_input("Indeks kolumny etykiet (0 = pierwsza):", min_value=0, max_value=10, value=0)

wgrany_plik = st.file_uploader("Wybierz plik Excel (.xlsx)", type=['xlsx'])

if wgrany_plik is not None:
    try:
        # W trybie automatycznym: wykryj ustawienia z surowego arkusza przed właściwym wczytaniem.
        powody_detekcji = None
        if tryb_ustawien_auto:
            with st.spinner('Analiza struktury pliku...'):
                wgrany_plik.seek(0)
                df_surowy = pd.read_excel(wgrany_plik, sheet_name=0)
                pomin_kolumne, transpozycja, powody_detekcji = wykryj_ustawienia_danych(df_surowy)
                wgrany_plik.seek(0)

        with st.spinner('Wczytywanie i przygotowywanie danych...'):
            wgrany_plik.seek(0)
            X, X_scaled, oryginalny_df, gt_labels, df_gt_preview, identyfikatory, ostrzezenie_gt = wczytaj_i_przygotuj_dane(
                wgrany_plik, pomin_kolumne, transpozycja, gt_indeks
            )

        # W trybie automatycznym limit osi Y liczymy z rzeczywistych danych.
        if tryb_ustawien_auto:
            limit_osi_y = sugeruj_limit_y(X)
            limit_txt = "autoskalowanie (brak sztywnej granicy)" if limit_osi_y == 0.0 else f"{limit_osi_y}"
            komunikat = "🤖 **Automatycznie dopasowane ustawienia:**\n"
            komunikat += f"- Pominięcie pierwszej kolumny: **{'TAK' if pomin_kolumne else 'NIE'}**\n"
            komunikat += f"- Transpozycja: **{'TAK' if transpozycja else 'NIE'}**\n"
            komunikat += f"- Limit osi Y: **{limit_txt}**\n\n"
            komunikat += "_Uzasadnienie:_ " + "; ".join(powody_detekcji)
            komunikat += "\n\n_Jeśli wykrycie jest błędne, odznacz opcję automatycznego dopasowania w panelu bocznym i ustaw parametry ręcznie._"
            st.info(komunikat)

        st.success(f"Dane wczytano! Główne dane: {X.shape[0]} widm, {X.shape[1]} punktów pomiarowych.")

        if ostrzezenie_gt:
            st.warning(f"⚠️ {ostrzezenie_gt}")

        if gt_labels is not None:
            st.info("✅ Wykryto arkusz 'Ground Truth'.")
            if len(gt_labels) != X.shape[0]:
                st.warning(
                    f"⚠️ Liczba etykiet GT ({len(gt_labels)}) nie zgadza się z liczbą widm ({X.shape[0]}). "
                    "Upewnij się też, że KOLEJNOŚĆ etykiet odpowiada kolejności widm w danych."
                )
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
                with st.spinner('Trwa obliczanie klastrów... W trybie globalnym może to zająć chwilę.'):

                    wyniki = {}
                    if rodzina_algorytmow == "Hierarchiczne (Tab 1)":
                        wyniki.update(analizuj_hierarchiczne(X_scaled, liczba_grup, birch_threshold, cure_repr, cure_shrink, rock_theta, cham_knn))
                    elif rodzina_algorytmow == "Partycjonujące (Tab 2)":
                        wyniki.update(analizuj_partycjonujace(X_scaled, liczba_grup))
                    elif rodzina_algorytmow == "Oparte na Gęstości (Tab 3)":
                        wyniki.update(analizuj_gestosciowe(X_scaled, eps_val, min_samples_val, bandwidth_val))
                    elif rodzina_algorytmow == "Oparte na Siatce (Tab 4)":
                        wyniki.update(analizuj_siatkowe(X_scaled, bins_val))
                    elif rodzina_algorytmow == "Rozmyte / Fuzzy (Tab 5)":
                        wyniki.update(analizuj_rozmyte(X_scaled, liczba_grup, m_fuzziness, beta_mec))
                    elif rodzina_algorytmow == "🔥 WSZYSTKIE NA RAZ (Globalny Ranking)":
                        wyniki.update(analizuj_hierarchiczne(X_scaled, liczba_grup, birch_threshold, cure_repr, cure_shrink, rock_theta, cham_knn))
                        wyniki.update(analizuj_partycjonujace(X_scaled, liczba_grup))
                        wyniki.update(analizuj_gestosciowe(X_scaled, eps_val, min_samples_val, bandwidth_val))
                        wyniki.update(analizuj_siatkowe(X_scaled, bins_val))
                        wyniki.update(analizuj_rozmyte(X_scaled, liczba_grup, m_fuzziness, beta_mec))

                    wykresy = {}
                    wyniki_ewaluacji = []
                    df_ewaluacja = None

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

                    if gt_labels is not None and wyniki_ewaluacji:
                        st.subheader("Wyniki Ewaluacji (Ranking Globalny)")
                        df_ewaluacja = pd.DataFrame(wyniki_ewaluacji).sort_values(by="ARI (Adjusted Rand Index)", ascending=False)
                        st.dataframe(
                            df_ewaluacja.style.highlight_max(subset=['ARI (Adjusted Rand Index)'], color='lightgreen'),
                            use_container_width=True
                        )

                    st.subheader("Wizualizacja średnich widm z pasmem błędu")
                    cols = st.columns(2)
                    for i, (nazwa, fig) in enumerate(wykresy.items()):
                        with cols[i % 2]:
                            st.pyplot(fig)
                        # NIE zamykamy figur tutaj – są jeszcze potrzebne do eksportu Excel poniżej.

                # --- EKSPORT DO EXCELA ---
                bufor = io.BytesIO()
                with pd.ExcelWriter(bufor, engine='xlsxwriter') as writer:
                    wyniki_df = pd.DataFrame(X)
                    # Spójne indeksowanie: zawsze przypisujemy identyfikatory widm jako indeks.
                    wyniki_df.index = identyfikatory

                    for algorytm, etyk in wyniki.items():
                        wyniki_df[f'Klaster_{algorytm}'] = etyk

                    wyniki_df.to_excel(writer, sheet_name='Sklasyfikowane_Dane')
                    if df_ewaluacja is not None:
                        df_ewaluacja.to_excel(writer, sheet_name='Ewaluacja', index=False)

                    workbook = writer.book
                    worksheet = workbook.add_worksheet('Wykresy_Klastrow')
                    wiersz_start = 1
                    for nazwa, fig in wykresy.items():
                        img_data = io.BytesIO()
                        fig.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
                        img_data.seek(0)
                        worksheet.insert_image(f'B{wiersz_start}', nazwa, {'image_data': img_data})
                        wiersz_start += 28

                # Dopiero teraz zwalniamy pamięć – po wykorzystaniu figur w eksporcie.
                for fig in wykresy.values():
                    plt.close(fig)

                st.download_button(
                    label="⬇️ Pobierz plik Excel (Kompletny Raport)",
                    data=bufor.getvalue(),
                    file_name="sklasyfikowane_widma_globalne.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Wystąpił błąd podczas przetwarzania pliku: {e}")
        # Pełny traceback dla łatwiejszej diagnozy (wcześniej był ukrywany).
        st.exception(e)
        st.code(traceback.format_exc())
