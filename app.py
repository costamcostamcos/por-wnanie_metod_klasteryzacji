import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering, DBSCAN, OPTICS, MeanShift
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import adjusted_rand_score

st.set_page_config(page_title="Klasteryzacja Widm EPR", layout="wide")

def wczytaj_i_przygotuj_dane(plik_excel, pomin_kolumne, transponuj):
    # Wczytanie głównego arkusza
    df = pd.read_excel(plik_excel, sheet_name=0)
    
    # KROK 1: Usunięcie pierwszej kolumny (np. oś X widma) jeśli zaznaczono
    if pomin_kolumne:
        X_df = df.iloc[:, 1:]
    else:
        X_df = df
        
    # KROK 2: Transpozycja (zamiana wierszy z kolumnami), jeśli widma są ułożone w kolumnach
    if transponuj:
        X_df = X_df.T
        
    # KROK 3: Czyszczenie danych (odporność na '#REF!' i puste komórki)
    X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    X = X_df.values
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Wczytanie Ground Truth
    ground_truth_labels = None
    try:
        df_gt = pd.read_excel(plik_excel, sheet_name='Ground Truth')
        ground_truth_labels = df_gt.iloc[:, 0].values 
    except ValueError:
        pass 
    
    return X, X_scaled, df, ground_truth_labels

def analizuj_widma_epr(X_scaled, liczba_grup, eps_dbscan, min_samples_dbscan):
    wyniki_klasteryzacji = {}
    
    slink = AgglomerativeClustering(n_clusters=liczba_grup, linkage='single')
    wyniki_klasteryzacji['SLINK'] = slink.fit_predict(X_scaled)
    
    birch = Birch(n_clusters=liczba_grup)
    wyniki_klasteryzacji['BIRCH'] = birch.fit_predict(X_scaled)

    kmeans = KMeans(n_clusters=liczba_grup, random_state=42)
    wyniki_klasteryzacji['K-Means'] = kmeans.fit_predict(X_scaled)
    
    pam = KMedoids(n_clusters=liczba_grup, method='pam', random_state=42)
    wyniki_klasteryzacji['PAM'] = pam.fit_predict(X_scaled)
    
    dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
    wyniki_klasteryzacji['DBSCAN'] = dbscan.fit_predict(X_scaled)
    
    optics = OPTICS(min_samples=min_samples_dbscan)
    wyniki_klasteryzacji['OPTICS'] = optics.fit_predict(X_scaled)
    
    meanshift = MeanShift()
    wyniki_klasteryzacji['Mean-Shift'] = meanshift.fit_predict(X_scaled)

    return wyniki_klasteryzacji

def generuj_wykres_srednich(X, etykiety, nazwa_algorytmu, limit_y=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    unikalne_etykiety = np.unique(etykiety)
    
    for etykieta in unikalne_etykiety:
        maska = (etykiety == etykieta)
        if not np.any(maska):
            continue
            
        srednie_widmo = np.mean(X[maska], axis=0)
        
        if etykieta == -1:
            ax.plot(srednie_widmo, color='gray', linestyle='--', alpha=0.6, label='Szum (-1)')
        else:
            ax.plot(srednie_widmo, label=f'Klaster {etykieta}')
            
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

st.sidebar.header("Parametry algorytmów")
liczba_grup = st.sidebar.number_input("Liczba klastrów (K-Means, PAM, Birch, SLINK):", min_value=2, max_value=20, value=3)
eps_dbscan = st.sidebar.slider("DBSCAN eps (promień):", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
min_samples_dbscan = st.sidebar.number_input("DBSCAN/OPTICS min_samples:", min_value=2, max_value=50, value=5)

st.sidebar.markdown("---")
st.sidebar.header("Ustawienia danych i wykresów")
pomin_kolumne = st.sidebar.checkbox("Zignoruj pierwszą kolumnę (np. oś X widma)", value=False)
transpozycja = st.sidebar.checkbox("Transpozycja danych (zaznacz, jeśli widma są ułożone w kolumnach)", value=False)
limit_osi_y = st.sidebar.number_input("Maksymalna wartość osi Y (0 aby wyłączyć):", min_value=0.0, max_value=1000.0, value=0.25, step=0.05)

wgrany_plik = st.file_uploader("Wybierz plik Excel (.xlsx)", type=['xlsx'])

if wgrany_plik is not None:
    try:
        with st.spinner('Wczytywanie i skalowanie danych...'):
            X, X_scaled, oryginalny_df, gt_labels = wczytaj_i_przygotuj_dane(wgrany_plik, pomin_kolumne, transpozycja)
            
        st.success(f"Dane wczytano! Po przekształceniach mamy do analizy {X.shape[0]} widm (wierszy), a każde ma {X.shape[1]} punktów pomiarowych.")
        
        if gt_labels is not None:
            st.info(f"✅ Wykryto arkusz 'Ground Truth' z {len(gt_labels)} etykietami. Ewaluacja zostanie przeprowadzona.")
        else:
            st.warning("⚠️ Brak arkusza 'Ground Truth'. Ewaluacja pominęta.")
        
        if st.button("Uruchom Klasteryzację", type="primary"):
            # Zabezpieczenie przed błędem z wymiarami
            if gt_labels is not None and len(gt_labels) != X.shape[0]:
                st.error(f"❌ Błąd zgodności! Twoje Ground Truth ma etykiety dla {len(gt_labels)} widm, ale główny arkusz ma {X.shape[0]} wierszy analizy. Zmień ustawienie opcji 'Transpozycja danych' lub 'Zignoruj pierwszą kolumnę' w panelu bocznym!")
            else:
                with st.spinner('Trwa obliczanie klastrów i ewaluacja...'):
                    wyniki = analizuj_widma_epr(X_scaled, liczba_grup, eps_dbscan, min_samples_dbscan)
                    wykresy = {}
                    wyniki_ewaluacji = []
                    
                    for nazwa_algorytmu, etykiety in wyniki.items():
                        # Jeśli użyto transpozycji, musimy ostrożnie dopisać wyniki, żeby nie zepsuć oryginalnego dataframe
                        oryginalny_df[f'Klaster_{nazwa_algorytmu}'] = "Zapis w nowym arkuszu" # Placeholder
                        
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
                        st.markdown("*Wskaźnik ARI: 1.0 oznacza idealne pokrycie z Ground Truth, 0.0 oznacza wynik losowy.*")
    
                    st.subheader("Wizualizacja średnich widm w klastrach")
                    for nazwa, fig in wykresy.items():
                        st.pyplot(fig)
                
                bufor = io.BytesIO()
                with pd.ExcelWriter(bufor, engine='xlsxwriter') as writer:
                    # Zapis przetransformowanych wyników 
                    wyniki_df = pd.DataFrame(X)
                    for algorytm, etyk in wyniki.items():
                        wyniki_df[f'Klaster_{algorytm}'] = etyk
                    
                    wyniki_df.to_excel(writer, sheet_name='Sklasyfikowane_Dane', index=False)
                    
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
                    label="⬇️ Pobierz plik Excel (Dane + Ewaluacja + Wykresy)",
                    data=bufor.getvalue(),
                    file_name="sklasyfikowane_widma.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
    except Exception as e:
        st.error(f"Wystąpił błąd podczas przetwarzania pliku: {e}")
