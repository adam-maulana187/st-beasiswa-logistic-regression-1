import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib
import os

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Analisis & Prediksi Penerimaan Beasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS kustom
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .card {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-text {
        color: #10B981;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .warning-text {
        color: #F59E0B;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header aplikasi
st.markdown('<h1 class="main-header">🎓 Aplikasi Analisis & Prediksi Penerimaan Beasiswa</h1>', unsafe_allow_html=True)
st.markdown("""
    Aplikasi ini menggunakan model **Logistic Regression** untuk menganalisis dan memprediksi kelayakan penerimaan beasiswa 
    berdasarkan berbagai faktor seperti IPK, pendapatan orang tua, prestasi, dan lainnya.
""")

# Sidebar untuk navigasi
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/student-center.png", width=80)
    st.title("Navigasi")
    menu = st.radio(
        "Pilih Menu:",
        ["📊 Dashboard Analisis", "🔍 Eksplorasi Data", "🤖 Model & Prediksi", "📈 Evaluasi Model", "🎯 Prediksi Baru", "📋 Tentang"]
    )
    
    st.markdown("---")
    st.markdown("### Dataset")
    uploaded_file = st.file_uploader("Unggah dataset CSV", type=['csv'])
    
    st.markdown("---")
    st.markdown("### Parameter Model")
    test_size = st.slider("Ukuran Data Testing (%)", 10, 40, 20)
    random_state = st.number_input("Random State", 0, 100, 42)
    
    st.markdown("---")
    st.markdown("### Dibuat oleh:")
    st.info("Sistem Prediksi Beasiswa\nv1.0")

# Load dataset
@st.cache_data
def load_data(file_path=None, uploaded_file=None):
    """Load dataset dari file atau upload"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif file_path and os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        # Gunakan dataset default yang tersedia
        df = pd.read_csv('template_dataset_beasiswa.csv')
    
    # Preprocessing dasar
    df_clean = df.copy()
    
    return df, df_clean

# Fungsi preprocessing
@st.cache_data
def preprocess_data(df):
    """Preprocess data untuk modeling"""
    df_processed = df.copy()
    
    # Encode variabel kategorikal
    categorical_cols = ['Asal_Sekolah', 'Lokasi_Domisili', 'Gender', 'Status_Disabilitas']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    # Pisahkan fitur dan target
    X = df_processed.drop('Diterima_Beasiswa', axis=1)
    y = df_processed['Diterima_Beasiswa']
    
    return X, y, label_encoders

# Fungsi training model
@st.cache_resource
def train_model(X_train, X_test, y_train, y_test):
    """Train Logistic Regression model"""
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    return model, scaler, y_pred, y_pred_proba

# Main app logic
def main():
    # Load data
    try:
        df, df_clean = load_data('template_dataset_beasiswa.csv', uploaded_file)
        
        if menu == "📊 Dashboard Analisis":
            show_dashboard(df)
        
        elif menu == "🔍 Eksplorasi Data":
            explore_data(df)
        
        elif menu == "🤖 Model & Prediksi":
            show_model_info(df)
        
        elif menu == "📈 Evaluasi Model":
            evaluate_model(df)
        
        elif menu == "🎯 Prediksi Baru":
            predict_new(df)
        
        elif menu == "📋 Tentang":
            show_about()
    
    except Exception as e:
        st.error(f"Terjadi error: {str(e)}")
        st.info("Pastikan dataset memiliki format yang benar atau gunakan dataset default.")

# ==================== FUNGSI UNTUK SETIAP MENU ====================

def show_dashboard(df):
    """Menampilkan dashboard analisis"""
    st.markdown('<h2 class="sub-header">📊 Dashboard Analisis Data</h2>', unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Data", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        diterima = df['Diterima_Beasiswa'].sum()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Diterima", f"{diterima} ({diterima/len(df)*100:.1f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        avg_ipk = df['IPK'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Rata-rata IPK", f"{avg_ipk:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        avg_pendapatan = df['Pendapatan_Orang_Tua'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Rata-rata Pendapatan", f"{avg_pendapatan:.1f} juta")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualisasi utama
    st.markdown("### 📈 Visualisasi Utama")
    
    # Tab untuk berbagai visualisasi
    tab1, tab2, tab3 = st.tabs(["Distribusi Data", "Korelasi", "Perbandingan"])
    
    with tab1:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribusi IPK', 'Distribusi Pendapatan Orang Tua', 
                          'Status Penerimaan', 'Jumlah Organisasi')
        )
        
        # Distribusi IPK
        fig.add_trace(
            go.Histogram(x=df['IPK'], nbinsx=20, name='IPK',
                        marker_color='#3B82F6'),
            row=1, col=1
        )
        
        # Distribusi Pendapatan
        fig.add_trace(
            go.Histogram(x=df['Pendapatan_Orang_Tua'], nbinsx=20, name='Pendapatan',
                        marker_color='#10B981'),
            row=1, col=2
        )
        
        # Status Penerimaan
        status_counts = df['Diterima_Beasiswa'].value_counts()
        fig.add_trace(
            go.Bar(x=['Tidak Diterima', 'Diterima'], y=status_counts.values,
                  marker_color=['#EF4444', '#10B981'], name='Status'),
            row=2, col=1
        )
        
        # Organisasi
        org_counts = df['Keikutsertaan_Organisasi'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=org_counts.index.astype(str), y=org_counts.values,
                  marker_color='#8B5CF6', name='Organisasi'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Heatmap korelasi
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Heatmap Korelasi Fitur Numerik',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Boxplot IPK berdasarkan status
            fig = px.box(df, x='Diterima_Beasiswa', y='IPK',
                        color='Diterima_Beasiswa',
                        title='Perbandingan IPK berdasarkan Status Beasiswa',
                        labels={'Diterima_Beasiswa': 'Status', 'IPK': 'IPK'},
                        color_discrete_map={0: '#EF4444', 1: '#10B981'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Boxplot Pendapatan berdasarkan status
            fig = px.box(df, x='Diterima_Beasiswa', y='Pendapatan_Orang_Tua',
                        color='Diterima_Beasiswa',
                        title='Perbandingan Pendapatan berdasarkan Status Beasiswa',
                        labels={'Diterima_Beasiswa': 'Status', 'Pendapatan_Orang_Tua': 'Pendapatan'},
                        color_discrete_map={0: '#EF4444', 1: '#10B981'})
            st.plotly_chart(fig, use_container_width=True)

def explore_data(df):
    """Menampilkan eksplorasi data interaktif"""
    st.markdown('<h2 class="sub-header">🔍 Eksplorasi Data Interaktif</h2>', unsafe_allow_html=True)
    
    # Filter interaktif
    st.markdown("### 🔎 Filter Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_ipk = st.slider("IPK Minimum", float(df['IPK'].min()), float(df['IPK'].max()), 
                           float(df['IPK'].min()), 0.1)
    
    with col2:
        max_pendapatan = st.slider("Pendapatan Maksimum (juta)", 
                                  float(df['Pendapatan_Orang_Tua'].min()), 
                                  float(df['Pendapatan_Orang_Tua'].max()),
                                  float(df['Pendapatan_Orang_Tua'].max()), 1.0)
    
    with col3:
        status_filter = st.selectbox("Status Beasiswa", 
                                    ["Semua", "Diterima", "Tidak Diterima"])
    
    # Terapkan filter
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df['IPK'] >= min_ipk]
    filtered_df = filtered_df[filtered_df['Pendapatan_Orang_Tua'] <= max_pendapatan]
    
    if status_filter == "Diterima":
        filtered_df = filtered_df[filtered_df['Diterima_Beasiswa'] == 1]
    elif status_filter == "Tidak Diterima":
        filtered_df = filtered_df[filtered_df['Diterima_Beasiswa'] == 0]
    
    # Tampilkan statistik filter
    st.info(f"**{len(filtered_df)}** data sesuai dengan filter (dari total {len(df)} data)")
    
    # Tampilkan data
    st.markdown("### 📋 Tabel Data")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Eksplorasi visual
    st.markdown("### 📊 Visualisasi Interaktif")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox("Pilih sumbu X", 
                             ['IPK', 'Pendapatan_Orang_Tua', 'Keikutsertaan_Organisasi', 
                              'Pengalaman_Sosial', 'Prestasi_Akademik', 'Prestasi_Non_Akademik'])
    
    with col2:
        y_axis = st.selectbox("Pilih sumbu Y", 
                             ['Pendapatan_Orang_Tua', 'IPK', 'Keikutsertaan_Organisasi', 
                              'Pengalaman_Sosial', 'Prestasi_Akademik', 'Prestasi_Non_Akademik'])
    
    # Scatter plot interaktif
    fig = px.scatter(filtered_df, x=x_axis, y=y_axis,
                     color='Diterima_Beasiswa',
                     hover_data=['Asal_Sekolah', 'Lokasi_Domisili', 'Gender'],
                     title=f'Hubungan {x_axis} vs {y_axis}',
                     color_discrete_map={0: '#EF4444', 1: '#10B981'},
                     labels={'Diterima_Beasiswa': 'Status Beasiswa'})
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Analisis kategorikal
    st.markdown("### 📈 Analisis Variabel Kategorikal")
    
    cat_col = st.selectbox("Pilih variabel kategorikal untuk dianalisis",
                          ['Asal_Sekolah', 'Lokasi_Domisili', 'Gender', 'Status_Disabilitas'])
    
    if cat_col:
        # Hitung persentase penerimaan per kategori
        cat_analysis = df.groupby(cat_col)['Diterima_Beasiswa'].agg(['count', 'sum']).reset_index()
        cat_analysis['Persentase_Diterima'] = (cat_analysis['sum'] / cat_analysis['count'] * 100).round(1)
        
        # Buat bar chart
        fig = px.bar(cat_analysis, x=cat_col, y='Persentase_Diterima',
                    title=f'Persentase Diterima berdasarkan {cat_col}',
                    color='Persentase_Diterima',
                    color_continuous_scale='Viridis',
                    text='Persentase_Diterima')
        
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(yaxis_title='Persentase Diterima (%)')
        
        st.plotly_chart(fig, use_container_width=True)

def show_model_info(df):
    """Menampilkan informasi tentang model"""
    st.markdown('<h2 class="sub-header">🤖 Model Logistic Regression</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <h3>📚 Tentang Model</h3>
    <p>Model <strong>Logistic Regression</strong> digunakan untuk memprediksi probabilitas penerimaan beasiswa 
    berdasarkan berbagai faktor. Model ini dipilih karena:</p>
    <ul>
        <li>Mudah diinterpretasi</li>
        <li>Memberikan probabilitas, bukan hanya klasifikasi biner</li>
        <li>Cocok untuk masalah klasifikasi biner</li>
        <li>Koefisien dapat diinterpretasi sebagai pengaruh faktor terhadap penerimaan</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Preprocess data untuk menampilkan fitur
    X, y, _ = preprocess_data(df)
    
    # Tampilkan informasi fitur
    st.markdown("### 🔧 Fitur yang Digunakan dalam Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Fitur Numerik:")
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        for feature in numeric_features:
            st.write(f"• {feature}")
    
    with col2:
        st.markdown("#### Fitur Kategorikal (encoded):")
        categorical_features = ['Asal_Sekolah', 'Lokasi_Domisili', 'Gender', 'Status_Disabilitas']
        for feature in categorical_features:
            if feature in X.columns:
                st.write(f"• {feature}")
    
    # Split data dan train model untuk demo
    st.markdown("### 🚀 Latih Model")
    
    if st.button("🚀 Latih Model Sekarang", type="primary"):
        with st.spinner("Melatih model... Harap tunggu"):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=random_state, stratify=y
            )
            
            # Train model
            model, scaler, y_pred, y_pred_proba = train_model(X_train, X_test, y_train, y_test)
            
            # Hitung akurasi
            accuracy = accuracy_score(y_test, y_pred)
            
            # Simpan model
            joblib.dump(model, 'model_beasiswa.pkl')
            joblib.dump(scaler, 'scaler_beasiswa.pkl')
            
            # Tampilkan hasil
            st.success(f"✅ Model berhasil dilatih dengan akurasi: **{accuracy:.2%}**")
            
            # Tampilkan koefisien model
            st.markdown("### 📊 Koefisien Model")
            
            coefficients = pd.DataFrame({
                'Fitur': X.columns,
                'Koefisien': model.coef_[0]
            }).sort_values('Koefisien', ascending=False)
            
            # Visualisasi koefisien
            fig = px.bar(coefficients, x='Koefisien', y='Fitur',
                        color='Koefisien',
                        color_continuous_scale='RdYlGn',
                        title='Koefisien Logistic Regression',
                        orientation='h')
            
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretasi koefisien
            st.markdown("### 📝 Interpretasi Koefisien")
            
            st.markdown("""
            **Koefisien positif** → Meningkatkan peluang diterima
            **Koefisien negatif** → Mengurangi peluang diterima
            
            **Contoh interpretasi:**
            - Jika koefisien IPK = 1.5, maka setiap kenaikan 1 poin IPK meningkatkan log-odds diterima sebesar 1.5
            - Jika koefisien Pendapatan = -0.3, maka setiap kenaikan 1 juta pendapatan mengurangi log-odds diterima sebesar 0.3
            """)

def evaluate_model(df):
    """Evaluasi performa model"""
    st.markdown('<h2 class="sub-header">📈 Evaluasi Model</h2>', unsafe_allow_html=True)
    
    # Preprocess data
    X, y, _ = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=random_state, stratify=y
    )
    
    # Train model
    model, scaler, y_pred, y_pred_proba = train_model(X_train, X_test, y_train, y_test)
    
    # Hitung metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Tampilkan metrics
    st.markdown("### 📊 Metrik Performa")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Akurasi", f"{accuracy:.2%}")
    
    with col2:
        st.metric("Precision", f"{report['1']['precision']:.2%}")
    
    with col3:
        st.metric("Recall", f"{report['1']['recall']:.2%}")
    
    with col4:
        st.metric("F1-Score", f"{report['1']['f1-score']:.2%}")
    
    # Visualisasi dalam tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix", "ROC Curve", "Classification Report", "Distribusi Probabilitas"])
    
    with tab1:
        # Confusion Matrix
        fig = px.imshow(cm,
                       labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                       x=['Tidak Diterima', 'Diterima'],
                       y=['Tidak Diterima', 'Diterima'],
                       text_auto=True,
                       color_continuous_scale='Blues')
        
        fig.update_layout(title='Confusion Matrix')
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretasi confusion matrix
        st.markdown("""
        **Interpretasi:**
        - **True Negative (TN):** Diprediksi tidak diterima dan benar tidak diterima
        - **False Positive (FP):** Diprediksi diterima tapi sebenarnya tidak (Type I Error)
        - **False Negative (FN):** Diprediksi tidak diterima tapi sebenarnya diterima (Type II Error)
        - **True Positive (TP):** Diprediksi diterima dan benar diterima
        """)
    
    with tab2:
        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='darkorange', width=3)
        ))
        
        # Diagonal
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='navy', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretasi AUC
        st.markdown(f"""
        **Interpretasi AUC = {roc_auc:.3f}:**
        - **0.90-1.00:** Sangat baik
        - **0.80-0.90:** Baik
        - **0.70-0.80:** Cukup
        - **0.60-0.70:** Buruk
        - **0.50-0.60:** Gagal
        
        Model ini memiliki **klasifikasi {'sangat baik' if roc_auc >= 0.9 else 'baik' if roc_auc >= 0.8 else 'cukup'}**.
        """)
    
    with tab3:
        # Classification Report
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.3f}").background_gradient(cmap='YlOrBr'), 
                    use_container_width=True)
    
    with tab4:
        # Distribusi probabilitas
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=('Distribusi Probabilitas Diterima', 
                                         'Threshold Analysis'))
        
        # Histogram probabilitas
        fig.add_trace(
            go.Histogram(x=y_pred_proba[y_test == 0], name='Tidak Diterima',
                        marker_color='#EF4444', opacity=0.7, nbinsx=20),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=y_pred_proba[y_test == 1], name='Diterima',
                        marker_color='#10B981', opacity=0.7, nbinsx=20),
            row=1, col=1
        )
        
        # Threshold analysis
        thresholds = np.arange(0.1, 1.0, 0.1)
        accuracies = []
        
        for threshold in thresholds:
            y_pred_threshold = (y_pred_proba >= threshold).astype(int)
            accuracies.append(accuracy_score(y_test, y_pred_threshold))
        
        fig.add_trace(
            go.Scatter(x=thresholds, y=accuracies, mode='lines+markers',
                      name='Akurasi', line=dict(color='#3B82F6', width=3)),
            row=1, col=2
        )
        
        fig.update_layout(height=500, showlegend=True, barmode='overlay')
        fig.update_xaxes(title_text="Probabilitas", row=1, col=1)
        fig.update_yaxes(title_text="Frekuensi", row=1, col=1)
        fig.update_xaxes(title_text="Threshold", row=1, col=2)
        fig.update_yaxes(title_text="Akurasi", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Threshold slider interaktif
        threshold = st.slider("Ubah threshold klasifikasi", 0.0, 1.0, 0.5, 0.05)
        y_pred_custom = (y_pred_proba >= threshold).astype(int)
        custom_accuracy = accuracy_score(y_test, y_pred_custom)
        
        st.info(f"**Akurasi dengan threshold {threshold:.2f}: {custom_accuracy:.2%}**")

def predict_new(df):
    """Halaman untuk prediksi data baru"""
    st.markdown('<h2 class="sub-header">🎯 Prediksi Penerimaan Beasiswa Baru</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <h3>📋 Cara Menggunakan</h3>
    <p>Isi form di bawah ini dengan data calon penerima beasiswa. Sistem akan memprediksi 
    probabilitas penerimaan berdasarkan model yang telah dilatih.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Form input
    with st.form("prediction_form"):
        st.markdown("### 📝 Data Calon Penerima")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, value=3.5, step=0.1)
            pendapatan = st.number_input("Pendapatan Orang Tua (juta)", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
            organisasi = st.number_input("Jumlah Keikutsertaan Organisasi", min_value=0, max_value=10, value=2)
            pengalaman_sosial = st.number_input("Pengalaman Sosial (jam)", min_value=0, max_value=500, value=100)
        
        with col2:
            prestasi_akademik = st.number_input("Jumlah Prestasi Akademik", min_value=0, max_value=10, value=2)
            prestasi_non_akademik = st.number_input("Jumlah Prestasi Non-Akademik", min_value=0, max_value=10, value=1)
            asal_sekolah = st.selectbox("Asal Sekolah", ["Negeri_Kota", "Swasta_Kota", "Negeri_Desa", "Swasta_Desa"])
            lokasi_domisili = st.selectbox("Lokasi Domisili", ["Kota", "Kabupaten"])
            gender = st.selectbox("Gender", ["L", "P"])
            status_disabilitas = st.selectbox("Status Disabilitas", ["Tidak", "Ya"])
        
        submit_button = st.form_submit_button("🚀 Prediksi Sekarang", type="primary")
    
    if submit_button:
        # Preprocess input
        X, y, label_encoders = preprocess_data(df)
        
        # Encode input kategorikal
        input_data = {
            'IPK': ipk,
            'Pendapatan_Orang_Tua': pendapatan,
            'Keikutsertaan_Organisasi': organisasi,
            'Pengalaman_Sosial': pengalaman_sosial,
            'Prestasi_Akademik': prestasi_akademik,
            'Prestasi_Non_Akademik': prestasi_non_akademik,
            'Asal_Sekolah': label_encoders['Asal_Sekolah'].transform([asal_sekolah])[0],
            'Lokasi_Domisili': label_encoders['Lokasi_Domisili'].transform([lokasi_domisili])[0],
            'Gender': label_encoders['Gender'].transform([gender])[0],
            'Status_Disabilitas': label_encoders['Status_Disabilitas'].transform([status_disabilitas])[0]
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure correct column order
        input_df = input_df[X.columns]
        
        try:
            # Load model dan scaler
            if os.path.exists('model_beasiswa.pkl') and os.path.exists('scaler_beasiswa.pkl'):
                model = joblib.load('model_beasiswa.pkl')
                scaler = joblib.load('scaler_beasiswa.pkl')
            else:
                # Train new model if not exists
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=random_state, stratify=y
                )
                model, scaler, _, _ = train_model(X_train, X_test, y_train, y_test)
            
            # Scale input
            input_scaled = scaler.transform(input_df)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Tampilkan hasil
            st.markdown("---")
            st.markdown("## 📊 Hasil Prediksi")
            
            # Progress bar untuk probabilitas
            prob_diterima = probability[1] * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### Probabilitas Diterima: **{prob_diterima:.1f}%**")
                st.progress(prob_diterima / 100)
                
                if prob_diterima >= 50:
                    st.markdown('<p class="success-text">✅ REKOMENDASI: DIREKOMENDASIKAN</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="warning-text">⚠️ REKOMENDASI: TIDAK DIREKOMENDASIKAN</p>', unsafe_allow_html=True)
            
            with col2:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_diterima,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Probabilitas Diterima"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "red"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Detail probabilitas
            st.markdown("### 📈 Detail Probabilitas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Probabilitas Diterima", f"{probability[1]:.2%}")
                st.metric("Probabilitas Ditolak", f"{probability[0]:.2%}")
            
            with col2:
                # Donut chart
                fig = go.Figure(data=[go.Pie(
                    labels=['Ditolak', 'Diterima'],
                    values=[probability[0], probability[1]],
                    hole=.3,
                    marker_colors=['#EF4444', '#10B981']
                )])
                
                fig.update_layout(title_text="Distribusi Probabilitas", height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Analisis faktor
            st.markdown("### 🔍 Analisis Faktor Penentu")
            
            # Hitung skor untuk setiap faktor
            faktor_analisis = {
                'IPK': ipk / 4.0 * 100,
                'Pendapatan': (1 - min(pendapatan / 50.0, 1)) * 100,  # Semakin rendah semakin baik
                'Organisasi': (organisasi / 10) * 100,
                'Pengalaman Sosial': (pengalaman_sosial / 500) * 100,
                'Prestasi Akademik': (prestasi_akademik / 10) * 100,
                'Prestasi Non-Akademik': (prestasi_non_akademik / 10) * 100
            }
            
            # Tampilkan sebagai bar chart
            fig = px.bar(
                x=list(faktor_analisis.keys()),
                y=list(faktor_analisis.values()),
                title="Skor Relatif Faktor Penentu (Semakin tinggi semakin baik)",
                labels={'x': 'Faktor', 'y': 'Skor (%)'},
                color=list(faktor_analisis.values()),
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
            
            # Rekomendasi perbaikan
            if prob_diterima < 50:
                st.markdown("### 💡 Rekomendasi Perbaikan")
                rekomendasi = []
                
                if ipk < 3.0:
                    rekomendasi.append("Tingkatkan IPK di atas 3.0")
                if pendapatan > 15:
                    rekomendasi.append("Sertakan bukti kebutuhan finansial")
                if organisasi < 2:
                    rekomendasi.append("Ikuti lebih banyak kegiatan organisasi")
                if prestasi_akademik < 2:
                    rekomendasi.append("Tingkatkan prestasi akademik")
                
                if rekomendasi:
                    for rec in rekomendasi:
                        st.write(f"• {rec}")
                else:
                    st.write("Profil sudah cukup baik, fokus pada penguatan esai dan rekomendasi.")
        
        except Exception as e:
            st.error(f"Terjadi error dalam prediksi: {str(e)}")

def show_about():
    """Halaman tentang aplikasi"""
    st.markdown('<h2 class="sub-header">📋 Tentang Aplikasi</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <h3>🎯 Tujuan Aplikasi</h3>
    <p>Aplikasi ini dikembangkan untuk membantu dalam proses seleksi penerimaan beasiswa 
    dengan menggunakan pendekatan data science dan machine learning.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏗️ Arsitektur Sistem")
        st.markdown("""
        1. **Data Collection**: Dataset berisi 10 fitur dan 1 target
        2. **Preprocessing**: Cleaning, encoding, dan normalisasi
        3. **Modeling**: Logistic Regression dengan regularisasi
        4. **Evaluation**: Cross-validation dan berbagai metrik
        5. **Deployment**: Aplikasi Streamlit interaktif
        """)
    
    with col2:
        st.markdown("### 🔧 Teknologi yang Digunakan")
        st.markdown("""
        - **Python 3.9+**
        - **Streamlit** untuk antarmuka web
        - **Scikit-learn** untuk machine learning
        - **Pandas & NumPy** untuk data processing
        - **Plotly & Matplotlib** untuk visualisasi
        - **Joblib** untuk model persistence
        """)
    
    st.markdown("### 📊 Dataset")
    st.markdown("""
    Dataset berisi **500+ sampel** dengan **10 fitur** berikut:
    
    1. **IPK** (Numerik): Indeks Prestasi Kumulatif
    2. **Pendapatan_Orang_Tua** (Numerik): Dalam juta rupiah
    3. **Asal_Sekolah** (Kategorik): Negeri/Swasta, Kota/Desa
    4. **Lokasi_Domisili** (Kategorik): Kota/Kabupaten
    5. **Keikutsertaan_Organisasi** (Numerik): Jumlah organisasi
    6. **Pengalaman_Sosial** (Numerik): Jam pengalaman sosial
    7. **Gender** (Kategorik): L/P
    8. **Status_Disabilitas** (Kategorik): Ya/Tidak
    9. **Prestasi_Akademik** (Numerik): Jumlah prestasi akademik
    10. **Prestasi_Non_Akademik** (Numerik): Jumlah prestasi non-akademik
    
    **Target**: Diterima_Beasiswa (0=Tidak, 1=Ya)
    """)
    
    st.markdown("### 👥 Tim Pengembang")
    st.markdown("""
    - **Data Scientist**: Analisis dan modeling
    - **ML Engineer**: Deployment dan optimization
    - **UI/UX Designer**: Antarmuka pengguna
    - **Domain Expert**: Ahli bidang beasiswa
    """)
    
    st.markdown("### 📞 Kontak & Dukungan")
    st.markdown("""
    Untuk pertanyaan, masukan, atau laporan bug:
    - Email: support@beasiswa-prediction.com
    - GitHub: github.com/beasiswa-prediction
    - Dokumentasi: docs.beasiswa-prediction.com
    """)
    
    st.markdown("---")
    st.markdown("**© 2024 Sistem Prediksi Beasiswa. Hak cipta dilindungi.**")

# Jalankan aplikasi
if __name__ == "__main__":
    main()