import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import adjusted_rand_score
from pyclustering.cluster.clara import clara
from pyclustering.cluster.clarans import clarans

st.set_page_config(page_title="Klasteryzacja Widm EPR - Metody Partycjonujące", layout="wide")

def wczytaj_i_przygotuj_dane(plik_excel, pomin_kolumne, transponuj, gt_indeks_kolumny):
    df = pd.read_excel(plik_excel, sheet_name=0)
    
    if pomin_kolumne:
        X_df = df.iloc[:, 1:]
    else:
        X_df = df
        
    if transponuj:
        X_df = X_df.T
        identyfikatory_widm = X_df.index.astype(str).tolist()
    else:
        if pomin_kolumne:
            identyfikatory_widm = df.iloc[:, 0].astype(str).tolist()
        else:
            identyfikatory_widm = [f"Widmo_{i+1}" for i in range(X_df.shape[0])]
        
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
    except Exception:
        pass 
    
    return X, X_scaled, df, ground_truth_labels, df_gt_preview, identyfikatory_widm

def analizuj_widma_epr(X_scaled, liczba_grup):
    wyniki_klasteryzacji = {}
    
    # 1. K-Means
    kmeans = KMeans(n_clusters=liczba_grup, random_state=42)
    wyniki_klasteryzacji['K-Means'] = kmeans.fit_predict(X_scaled)
    
    # 2. K-Medoids (standardowa metoda alternatywna wg implementacji sklearn_extra)
    kmedoids = KMedoids(n_clusters=liczba_grup, method='alternate', random_state=42)
    wyniki_klasteryzacji['K-Medoids'] = kmedoids.fit_predict(X_scaled)

    # 3. PAM (Partitioning Around Medoids)
    pam = KMedoids(n_clusters=liczba_grup, method='pam', random_state=42)
    wyniki_klasteryzacji['PAM'] = pam.fit_predict(X_scaled)
    
    # Przygotowanie danych dla biblioteki pyclustering (wymaga formatu listy list)
    dane_pyclustering = X_scaled.tolist()
    
    # 4. CLARA
    # Parametr 'number_samples' określa rozmiar podpróbki. Zazwyczaj 40 + 2*k jest dobrym punktem wyjścia.
    rozmiar_probki_clara = min(len(dane_pyclustering), 40 + 2 * liczba_grup)
    clara_instance = clara(dane_pyclustering, liczba_grup, rozmiar_probki_clara)
    clara_instance.process()
    clara_clusters = clara_instance.get_clusters()
    
    # Konwersja formatu pyclustering na płaską tablicę etykiet (taką jak ze sklearn)
    etykiety_clara = np.zeros(X_scaled.shape[0], dtype=int)
    for id_klastra, klaster in enumerate(clara_clusters):
        for id_punktu in klaster:
            etykiety_clara[id_punktu] = id_klastra
    wyniki_klasteryzacji['CLARA'] = etykiety_clara

    # 5. CLARANS
    # Parametry numlocal i maxneighbor determinują dokładność i czas działania. Ustawiamy stabilne domyślne.
    clarans_instance = clarans(dane_pyclustering, liczba_grup, numlocal=3, maxneighbor=4)
    clarans_instance.process()
    clarans_clusters = clarans_instance.get_clusters()
    
    etykiety_clarans = np.zeros(X_scaled.shape[0], dtype=int)
    for id_klastra, klaster in enumerate(clarans_clusters):
        for id_punktu in klaster:
            etykiety_clarans[id_punktu] = id_klastra
    wyniki_klasteryzacji['CLARANS'] = etykiety_clarans

    return wyniki_klasteryzacji

def generuj_wykres_srednich(X, etykiety, nazwa_algorytmu, limit_y=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    unikalne_etykiety = np.unique(etykiety)
    os_x = np.arange(X.shape[1])
    
    for etykieta in unikalne_etykiety:
        maska = (etykiety == etykieta)
        liczba_widm = np.sum(maska)
        if liczba_widm == 0:
            continue
            
        srednie_widmo = np.mean(X[maska], axis=0)
        odchylenie = np.std(X[maska], axis=0)
        
        if etykieta == -1:
            line = ax.plot(os_x, srednie_widmo, color='gray', linestyle='--', label=f'Szum (-1) [n={liczba_widm}]')
            ax.fill_between(os_x, srednie_widmo - odchylenie, srednie_widmo + odchylenie, color='gray', alpha=0.2)
        else:
            line = ax.plot(os_x, srednie_widmo, label=f'Klaster {etykieta} [n={liczba_widm}]')
            kolor = line[0].get_color()
            ax.fill_between(os_x, srednie_widmo - odchylenie, srednie_widmo + odchylenie, color=kolor, alpha=0.2)
            
    ax.set_title(f'Średnia reprezentacja klastrów: {nazwa_algorytmu} (wstęga = $\pm$1 odchylenie std)')
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
st.title("🔬 Analiza i Klasteryzacja Widm EPR (Metody Partycjonujące)")
st.markdown("Algorytmy: **K-means, K-medoids, PAM, CLARA, CLARANS**")

st.sidebar.header("Parametry algorytmów")
liczba_grup = st.sidebar.number_input("Liczba klastrów (K):", min_value=2, max_value=20, value=3)

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
            with st.expander("Kliknij, aby rozwinąć PODGLĄD GROUND TRUTH (Sprawdź poprawność etykiet)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Cały arkusz Ground Truth:**")
                    st.dataframe(df_gt_preview.head(10))
                with col2:
                    st.markdown(f"**Etykiety wczytane z kolumny {gt_indeks}:**")
                    st.write(gt_labels[:10])
                    st.markdown(f"*Liczba unikalnych etykiet: {len(np.unique(gt_labels))}*")
                    if len(np.unique(gt_labels)) > 20:
                        st.error("⚠️ UWAGA! Wykryto bardzo dużo unikalnych etykiet. Zmień indeks kolumny w panelu bocznym!")
        
        if st.button("Uruchom Klasteryzację", type="primary"):
            if gt_labels is not None and len(gt_labels) != X.shape[0]:
                st.error(f"❌ Błąd zgodności! Ground Truth (ilość: {len(gt_labels)}) nie pasuje do badanych widm ({X.shape[0]}).")
            else:
                with st.spinner('Trwa obliczanie klastrów (CLARANS może chwilę potrwać) i ewaluacja...'):
                    wyniki = analizuj_widma_epr(X_scaled, liczba_grup)
                    wykresy = {}
                    wyniki_ewaluacji = []
                    
                    st.subheader("Skład poszczególnych klastrów")
                    
                    for nazwa_algorytmu, etykiety in wyniki.items():
                        with st.expander(f"Rozkład widm: {nazwa_algorytmu}"):
                            unikalne = np.unique(etykiety)
                            for etyk in unikalne:
                                indeksy_w_klastrze = np.where(etykiety == etyk)[0]
                                id_widm_w_klastrze = [identyfikatory[i] for i in indeksy_w_klastrze]
                                st.markdown(f"**Klaster {etyk}** (Sztuk: {len(indeksy_w_klastrze)}): {', '.join(id_widm_w_klastrze)}")
                        
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
                    if transpozycja:
                        wyniki_df.index = identyfikatory
                    
                    for algorytm, etyk in wyniki.items():
                        wyniki_df[f'Klaster_{algorytm}'] = etyk
                    
                    wyniki_df.to_excel(writer, sheet_name='Sklasyfikowane_Dane')
                    
                    if gt_labels is not None:
                        df_ewaluacja.to_excel(writer, sheet_name='Ewaluacja', index=False)
                    
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
                    file_name="sklasyfikowane_widma_partycjonujace.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
    except Exception as e:
        st.error(f"Wystąpił błąd podczas przetwarzania pliku: {e}")
