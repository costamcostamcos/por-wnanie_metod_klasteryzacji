import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering, DBSCAN, OPTICS, MeanShift

# Konfiguracja strony
st.set_page_config(page_title="Klasteryzacja Widm EPR", layout="wide")

def wczytaj_i_przygotuj_dane(plik_excel):
    """Wczytuje i skaluje dane."""
    df = pd.read_excel(plik_excel)
    X = df.values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, X_scaled, df

def analizuj_widma_epr(X_scaled, liczba_grup, eps_dbscan, min_samples_dbscan):
    """Przeprowadza klasteryzację za pomocą różnych algorytmów (bez problematycznego PAM)."""
    wyniki_klasteryzacji = {}
    
    # METODY HIERARCHICZNE
    slink = AgglomerativeClustering(n_clusters=liczba_grup, linkage='single')
    wyniki_klasteryzacji['SLINK'] = slink.fit_predict(X_scaled)
    
    birch = Birch(n_clusters=liczba_grup)
    wyniki_klasteryzacji['BIRCH'] = birch.fit_predict(X_scaled)

    # METODY PARTYCJONUJĄCE
    kmeans = KMeans(n_clusters=liczba_grup, random_state=42)
    wyniki_klasteryzacji['K-Means'] = kmeans.fit_predict(X_scaled)
    
    # METODY OPARTE NA GĘSTOŚCI
    dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
    wyniki_klasteryzacji['DBSCAN'] = dbscan.fit_predict(X_scaled)
    
    optics = OPTICS(min_samples=min_samples_dbscan)
    wyniki_klasteryzacji['OPTICS'] = optics.fit_predict(X_scaled)
    
    meanshift = MeanShift()
    wyniki_klasteryzacji['Mean-Shift'] = meanshift.fit_predict(X_scaled)

    return wyniki_klasteryzacji

def generuj_wykres_srednich(X, etykiety, nazwa_algorytmu):
    """Generuje wykres uśrednionych widm dla każdego klastra."""
    fig, ax = plt.subplots(figsize=(10, 5))
    unikalne_etykiety = np.unique(etykiety)
    
    for etykieta in unikalne_etykiety:
        # Zabezpieczenie przed błędami w przypadku braku danych w klastrze
        maska = (etykiety == etykieta)
        if not np.any(maska):
            continue
            
        srednie_widmo = np.mean(X[maska], axis=0)
        
        if etykieta == -1:
            # Klaster -1 oznacza szum (np. w DBSCAN/OPTICS) - rysujemy na szaro
            ax.plot(srednie_widmo, color='gray', linestyle='--', alpha=0.6, label='Szum (-1)')
        else:
            ax.plot(srednie_widmo, label=f'Klaster {etykieta}')
            
    ax.set_title(f'Średnia reprezentacja klastrów: {nazwa_algorytmu}')
    ax.set_xlabel('Punkt pomiarowy / Pole magnetyczne')
    ax.set_ylabel('Średnie natężenie')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    return fig

# --- INTERFEJS STREAMLIT ---
st.title("🔬 Analiza i Klasteryzacja Widm EPR")
st.markdown("Wgraj plik Excel, dobierz parametry, a aplikacja wygeneruje klastry oraz narysuje średnie widmo dla każdej grupy.")

# Pasek boczny
st.sidebar.header("Parametry algorytmów")
liczba_grup = st.sidebar.number_input("Liczba klastrów (K-Means, Birch, SLINK):", min_value=2, max_value=20, value=3)
eps_dbscan = st.sidebar.slider("DBSCAN eps (promień):", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
min_samples_dbscan = st.sidebar.number_input("DBSCAN/OPTICS min_samples:", min_value=2, max_value=50, value=5)

wgrany_plik = st.file_uploader("Wybierz plik Excel (.xlsx)", type=['xlsx'])

if wgrany_plik is not None:
    try:
        with st.spinner('Wczytywanie i skalowanie danych...'):
            X, X_scaled, oryginalny_df = wczytaj_i_przygotuj_dane(wgrany_plik)
            
        st.success("Dane wczytano pomyślnie!")
        
        if st.button("Uruchom Klasteryzację i Rysuj Wykresy", type="primary"):
            with st.spinner('Trwa obliczanie klastrów i generowanie wykresów...'):
                wyniki = analizuj_widma_epr(X_scaled, liczba_grup, eps_dbscan, min_samples_dbscan)
                
                wykresy = {}
                st.subheader("Wizualizacja średnich widm w klastrach")
                
                # Przetwarzanie i rysowanie wykresów dla każdego algorytmu
                for nazwa_algorytmu, etykiety in wyniki.items():
                    oryginalny_df[f'Klaster_{nazwa_algorytmu}'] = etykiety
                    
                    fig = generuj_wykres_srednich(X, etykiety, nazwa_algorytmu)
                    wykresy[nazwa_algorytmu] = fig
                    
                    # Wyświetlenie wykresu w Streamlit
                    st.pyplot(fig)
            
            # --- ZAPIS DO EXCELA (Dane + Wykresy) ---
            bufor = io.BytesIO()
            with pd.ExcelWriter(bufor, engine='xlsxwriter') as writer:
                # 1. Zapis danych do pierwszego arkusza
                oryginalny_df.to_excel(writer, sheet_name='Sklasyfikowane_Dane', index=False)
                
                # 2. Tworzenie drugiego arkusza na wykresy
                workbook  = writer.book
                worksheet = workbook.add_worksheet('Wykresy_Klastrow')
                
                wiersz_start = 1
                for nazwa, fig in wykresy.items():
                    # Zapisywanie wykresu do bufora obrazu w pamięci
                    img_data = io.BytesIO()
                    fig.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
                    img_data.seek(0)
                    
                    # Wstawianie obrazu do Excela (przesuwamy się w dół o 25 wierszy z każdym wykresem)
                    worksheet.insert_image(f'B{wiersz_start}', nazwa, {'image_data': img_data})
                    wiersz_start += 28

            st.success("Analiza zakończona! Możesz teraz pobrać plik Excel z danymi i wykresami.")
            
            st.download_button(
                label="⬇️ Pobierz plik Excel (Dane + Wykresy)",
                data=bufor.getvalue(),
                file_name="sklasyfikowane_widma_epr_z_wykresami.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
    except Exception as e:
        st.error(f"Wystąpił błąd podczas przetwarzania pliku: {e}")
