import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering, DBSCAN, OPTICS, MeanShift
from sklearn_extra.cluster import KMedoids

# Konfiguracja strony
st.set_page_config(page_title="Klasteryzacja Widm EPR", layout="wide")

def wczytaj_i_przygotuj_dane(plik_excel):
    """Wczytuje i skaluje dane."""
    df = pd.read_excel(plik_excel)
    X = df.values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df

def analizuj_widma_epr(X_scaled, liczba_grup, eps_dbscan, min_samples_dbscan):
    """Przeprowadza klasteryzację za pomocą różnych algorytmów."""
    wyniki_klasteryzacji = {}
    
    # TABELA 1: METODY HIERARCHICZNE
    slink = AgglomerativeClustering(n_clusters=liczba_grup, linkage='single')
    wyniki_klasteryzacji['SLINK'] = slink.fit_predict(X_scaled)
    
    birch = Birch(n_clusters=liczba_grup)
    wyniki_klasteryzacji['BIRCH'] = birch.fit_predict(X_scaled)

    # TABELA 2: METODY PARTYCJONUJĄCE
    kmeans = KMeans(n_clusters=liczba_grup, random_state=42)
    wyniki_klasteryzacji['K-Means'] = kmeans.fit_predict(X_scaled)
    
    pam = KMedoids(n_clusters=liczba_grup, method='pam', random_state=42)
    wyniki_klasteryzacji['PAM'] = pam.fit_predict(X_scaled)

    # TABELA 3: METODY OPARTE NA GĘSTOŚCI
    dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
    wyniki_klasteryzacji['DBSCAN'] = dbscan.fit_predict(X_scaled)
    
    optics = OPTICS(min_samples=min_samples_dbscan)
    wyniki_klasteryzacji['OPTICS'] = optics.fit_predict(X_scaled)
    
    meanshift = MeanShift()
    wyniki_klasteryzacji['Mean-Shift'] = meanshift.fit_predict(X_scaled)

    return wyniki_klasteryzacji

# --- INTERFEJS STREAMLIT ---
st.title("🔬 Analiza i Klasteryzacja Widm EPR")
st.markdown("Wgraj plik Excel zawierający dane widm EPR, dostosuj parametry i pobierz sklasyfikowane wyniki.")

# Pasek boczny z parametrami
st.sidebar.header("Parametry algorytmów")
liczba_grup = st.sidebar.number_input("Liczba klastrów (K-Means, PAM, Birch, SLINK):", min_value=2, max_value=20, value=3)
eps_dbscan = st.sidebar.slider("DBSCAN eps (promień):", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
min_samples_dbscan = st.sidebar.number_input("DBSCAN/OPTICS min_samples:", min_value=2, max_value=20, value=5)

# Wgrywanie pliku
wgrany_plik = st.file_uploader("Wybierz plik Excel (.xlsx)", type=['xlsx'])

if wgrany_plik is not None:
    try:
        with st.spinner('Wczytywanie i skalowanie danych...'):
            X_scaled, oryginalny_df = wczytaj_i_przygotuj_dane(wgrany_plik)
            
        st.success("Dane wczytano pomyślnie!")
        st.write("Podgląd surowych danych (pierwsze 5 wierszy):")
        st.dataframe(oryginalny_df.head())
        
        if st.button("Uruchom Klasteryzację", type="primary"):
            with st.spinner('Trwa obliczanie klastrów...'):
                wyniki = analizuj_widma_epr(X_scaled, liczba_grup, eps_dbscan, min_samples_dbscan)
                
                # Przypisywanie etykiet do ramki danych
                for nazwa_algorytmu, etykiety in wyniki.items():
                    oryginalny_df[f'Klaster_{nazwa_algorytmu}'] = etykiety
            
            st.success("Klasteryzacja zakończona!")
            st.write("Podgląd sklasyfikowanych danych:")
            st.dataframe(oryginalny_df)
            
            # Zapisywanie do pamięci RAM, aby umożliwić pobranie w Streamlit
            bufor = io.BytesIO()
            with pd.ExcelWriter(bufor, engine='openpyxl') as writer:
                oryginalny_df.to_excel(writer, index=False)
            
            st.download_button(
                label="⬇️ Pobierz plik ze sklasyfikowanymi widmami",
                data=bufor.getvalue(),
                file_name="sklasyfikowane_widma_epr.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
    except Exception as e:
        st.error(f"Wystąpił błąd podczas przetwarzania pliku: {e}")
