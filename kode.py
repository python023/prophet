import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Membaca data dari file
file_path = 'data.csv'  # Sesuaikan dengan nama file yang diupload
df = pd.read_csv(file_path, sep=',')

# Mengubah nama kolom sesuai dengan data yang benar
df.columns = ['Provinsi', 'SP1971', 'SP1980', 'SP1990', 'SP2000', 'SP2010', 'SP2020']

# Mengganti nilai '-' dengan NaN dan mengonversi kolom yang relevan menjadi numerik
df.replace('-', np.nan, inplace=True)
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Menghapus baris yang mengandung NaN
df.dropna(inplace=True)

# Mengubah data ke format panjang
df_melted = df.melt(id_vars=['Provinsi'],
                    value_vars=['SP1971', 'SP1980', 'SP1990', 'SP2000', 'SP2010', 'SP2020'],
                    var_name='Year',
                    value_name='TFR')

# Mengubah tahun menjadi format datetime
df_melted['Year'] = df_melted['Year'].str.extract('(\d+)').astype(int)
df_melted['Year'] = pd.to_datetime(df_melted['Year'], format='%Y')
df_melted = df_melted.rename(columns={'Year': 'ds', 'TFR': 'y'})

# Mengonversi kolom TFR ke tipe numerik
df_melted['y'] = df_melted['y'].astype(float)

def analyze_province(province_name):
    # Filter data untuk provinsi yang dipilih
    province_data = df_melted[df_melted['Provinsi'] == province_name].copy()

    if province_data.empty:
        st.write(f"Data untuk provinsi '{province_name}' tidak ditemukan.")
        return None

    st.write("Data provinsi yang dianalisis:")
    st.write(province_data)

    # Melatih model menggunakan Prophet dengan yearly seasonality
    model = Prophet(growth='linear', yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    model.fit(province_data)

    # Membuat dataframe untuk prediksi
    future_dates = model.make_future_dataframe(periods=10, freq='Y')
    forecast = model.predict(future_dates)

    st.write("\nHasil prediksi:")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Mengambil prediksi untuk rentang 2030
    future_dates_2030 = pd.date_range(start='2030-01-01', end='2030-12-31', freq='D')
    future_2030 = pd.DataFrame({'ds': future_dates_2030})
    forecast_2030 = model.predict(future_2030)

    # Mengambil prediksi untuk tanggal 2030-01-30
    prediksi_2030 = forecast_2030[forecast_2030['ds'] == '2030-01-30']
    future_prediction = prediksi_2030['yhat'].values[0] if not prediksi_2030.empty else np.nan

    st.write(f"\nPrediksi untuk 2030-01-30 : {future_prediction}")

    # Plot hasil prediksi
    fig, ax = plt.subplots()
    model.plot(forecast, ax=ax)
    plt.title(f'Analisis Prophet Angka Kelahiran di {province_name}')
    plt.xlabel('Tahun')
    plt.ylabel('Angka Kelahiran')

    # Menambahkan teks untuk setiap titik data
    for i in range(len(province_data)):
        ax.text(province_data['ds'].iloc[i], province_data['y'].iloc[i],
                f'{province_data["y"].iloc[i]:.2f}', fontsize=10, ha='right')

    # Menambahkan prediksi 2030 ke grafik
    if not np.isnan(future_prediction):
        ax.scatter(pd.Timestamp('2030-01-30'), future_prediction, color='green', s=100, label='Prediksi 2030')
        # Menggambarkan garis prediksi menuju tahun 2030
        last_year = pd.Timestamp(province_data['ds'].max())
        ax.plot([last_year, pd.Timestamp('2030-01-30')],
                [province_data['y'].iloc[-1], future_prediction],
                color='red', linestyle='--')

    ax.legend()

    # Menambahkan informasi metrik ke plot
    y_true = province_data['y'].values
    y_pred = forecast.loc[forecast['ds'].isin(province_data['ds']), 'yhat'].values

    if len(y_pred) > len(y_true):
        y_pred = y_pred[:len(y_true)]
    elif len(y_true) > len(y_pred):
        y_true = y_true[:len(y_pred)]

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Plot komponen
    fig2 = model.plot_components(forecast)

    # Cross Validation
    initial_period = '3650 days'  # 10 tahun data untuk pelatihan awal
    period = '365 days'  # Frekuensi prediksi tahunan
    horizon = '730 days'  # Jangka waktu prediksi 2 tahun

    df_cv = cross_validation(model, initial=initial_period, period=period, horizon=horizon)
    df_p = performance_metrics(df_cv)

    # Visualisasi Cross Validation
    fig3, ax3 = plt.subplots()
    ax3.plot(df_cv['ds'], df_cv['y'], 'k-', label='Actual')
    ax3.plot(df_cv['ds'], df_cv['yhat'], 'r-', label='Fitted')
    ax3.set_title('Hasil Cross Validation')
    ax3.set_xlabel('Tahun')
    ax3.set_ylabel('Angka Kelahiran')
    ax3.legend()

    return {
        'model': model,
        'forecast': forecast,
        'plot': fig,
        'components_plot': fig2,
        'cross_val_plot': fig3,
        'metrics': {
            'MSE': mse,
            'RMSE': rmse
        },
        'performance_metrics': df_p,
        'future_prediction_2030': future_prediction
    }

# Streamlit UI
st.title("Prediksi Angka Kelahiran di Indonesia Berdasarkan Provinsi")

province_name = st.selectbox("Silahkan Pilih Provinsi", ['Cari Provinsi'] + list(df['Provinsi'].unique()))

if province_name and province_name != 'Cari Provinsi':
    result = analyze_province(province_name)
    if result:
        st.text(f"MSE           : {result['metrics']['MSE']:.4f}")
        st.text(f"RMSE          : {result['metrics']['RMSE']:.4f}")
        
        st.pyplot(result['plot'])  # Menampilkan plot hasil prediksi
        st.pyplot(result['components_plot'])  # Menampilkan plot komponen
        st.pyplot(result['cross_val_plot'])  # Menampilkan plot cross-validation
else:
    st.write("Silahkan pilih provinsi di atas untuk melihat Prediksi Angka Kelahiran di Indonesia Berdasarkan Provinsi.")