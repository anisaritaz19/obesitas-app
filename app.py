"""
app.py - Aplikasi Flask untuk klasifikasi tingkat obesitas (Random Forest)
"""
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)
app.secret_key = 'obesity_rf_2024'

# Load model
with open('model.pkl', 'rb') as f:
    arts = pickle.load(f)

model          = arts['model']
le_dict        = arts['label_encoders']
le_target      = arts['label_encoder_target']
scaler         = arts['scaler']
feature_names  = arts['feature_names']
target_classes = arts['target_classes']
num_cols       = arts['num_cols']
metrics        = arts['metrics']

CLASS_INFO = {
    'Insufficient_Weight': {
        'label': 'Kekurangan Berat Badan',
        'label_en': 'Insufficient Weight',
        'color': '#3498db',
        'badge': 'info',
        'icon': 'fa-arrow-down',
        'bmi_range': '< 18.5',
        'risk': 'Sedang',
        'risk_color': 'warning',
        'image': 'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=500&h=300&fit=crop',
        'desc': 'Berat badan Anda berada di bawah normal. Kondisi ini dapat meningkatkan risiko malnutrisi dan melemahkan sistem imun.',
        'tips': [
            'Tingkatkan asupan kalori dengan makanan bergizi tinggi',
            'Konsumsi protein, karbohidrat kompleks, dan lemak sehat',
            'Lakukan olahraga kekuatan untuk membangun massa otot',
            'Makan 5–6 kali sehari dalam porsi lebih kecil',
            'Konsultasi dengan ahli gizi untuk program kenaikan berat badan',
        ]
    },
    'Normal_Weight': {
        'label': 'Berat Badan Normal',
        'label_en': 'Normal Weight',
        'color': '#27ae60',
        'badge': 'success',
        'icon': 'fa-check-circle',
        'bmi_range': '18.5 – 24.9',
        'risk': 'Rendah',
        'risk_color': 'success',
        'image': 'https://images.unsplash.com/photo-1476480862126-209bfaa8edc8?w=500&h=300&fit=crop',
        'desc': 'Selamat! Berat badan Anda berada dalam kisaran normal. Pertahankan gaya hidup sehat Anda.',
        'tips': [
            'Pertahankan pola makan seimbang dengan gizi lengkap',
            'Tetap aktif secara fisik minimal 30 menit per hari',
            'Cukupi kebutuhan cairan 2 liter per hari',
            'Istirahat cukup 7–8 jam per malam',
            'Lakukan pemeriksaan kesehatan rutin setiap tahun',
        ]
    },
    'Overweight_Level_I': {
        'label': 'Kelebihan Berat Badan Tingkat I',
        'label_en': 'Overweight Level I',
        'color': '#f39c12',
        'badge': 'warning',
        'icon': 'fa-exclamation-circle',
        'bmi_range': '25.0 – 27.4',
        'risk': 'Sedang',
        'risk_color': 'warning',
        'image': 'https://images.unsplash.com/photo-1490645935967-10de6ba17061?w=500&h=300&fit=crop',
        'desc': 'Berat badan Anda sedikit di atas normal. Mulai terapkan perubahan gaya hidup sebelum kondisi semakin berat.',
        'tips': [
            'Kurangi asupan kalori 300–500 kkal per hari',
            'Perbanyak konsumsi sayur, buah, dan serat',
            'Tingkatkan aktivitas fisik menjadi 45–60 menit/hari',
            'Hindari makanan tinggi gula dan lemak jenuh',
            'Target penurunan 0.5 kg per minggu secara bertahap',
        ]
    },
    'Overweight_Level_II': {
        'label': 'Kelebihan Berat Badan Tingkat II',
        'label_en': 'Overweight Level II',
        'color': '#e67e22',
        'badge': 'warning',
        'icon': 'fa-exclamation-triangle',
        'bmi_range': '27.5 – 29.9',
        'risk': 'Tinggi',
        'risk_color': 'danger',
        'image': 'https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=500&h=300&fit=crop',
        'desc': 'Risiko signifikan untuk diabetes tipe 2 dan hipertensi. Perubahan gaya hidup segera diperlukan.',
        'tips': [
            'Konsultasikan dengan dokter untuk program penurunan berat badan',
            'Targetkan penurunan 5–10% berat badan dalam 6 bulan',
            'Kombinasikan diet rendah kalori dengan olahraga teratur',
            'Pantau tekanan darah dan gula darah secara rutin',
            'Hindari alkohol dan rokok yang memperburuk kondisi',
        ]
    },
    'Obesity_Type_I': {
        'label': 'Obesitas Tipe I',
        'label_en': 'Obesity Type I',
        'color': '#e74c3c',
        'badge': 'danger',
        'icon': 'fa-times-circle',
        'bmi_range': '30.0 – 34.9',
        'risk': 'Sangat Tinggi',
        'risk_color': 'danger',
        'image': 'https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?w=500&h=300&fit=crop',
        'desc': 'Risiko tinggi penyakit kardiovaskular, diabetes, dan gangguan sendi. Intervensi profesional sangat disarankan.',
        'tips': [
            'Segera cari bantuan profesional (dokter, ahli gizi, psikolog)',
            'Ikuti program penurunan berat badan yang terstruktur',
            'Olahraga minimal 60 menit per hari, 5 hari/minggu',
            'Hindari diet ekstrem – lakukan perubahan bertahap',
            'Bergabung dengan komunitas pendukung penurunan berat badan',
        ]
    },
    'Obesity_Type_II': {
        'label': 'Obesitas Tipe II',
        'label_en': 'Obesity Type II',
        'color': '#c0392b',
        'badge': 'danger',
        'icon': 'fa-skull-crossbones',
        'bmi_range': '35.0 – 39.9',
        'risk': 'Sangat Serius',
        'risk_color': 'danger',
        'image': 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=500&h=300&fit=crop',
        'desc': 'Kondisi serius yang memerlukan intervensi medis segera. Risiko tinggi komplikasi kesehatan yang mengancam jiwa.',
        'tips': [
            'SEGERA konsultasi dengan tim medis spesialis',
            'Lakukan evaluasi kesehatan menyeluruh',
            'Intervensi medis atau obat-obatan mungkin diperlukan',
            'Dukungan psikologis untuk perubahan perilaku jangka panjang',
            'Pantau kondisi kesehatan secara intensif',
        ]
    },
    'Obesity_Type_III': {
        'label': 'Obesitas Tipe III (Morbid)',
        'label_en': 'Obesity Type III',
        'color': '#922b21',
        'badge': 'danger',
        'icon': 'fa-hospital',
        'bmi_range': '≥ 40.0',
        'risk': 'Ekstrim',
        'risk_color': 'danger',
        'image': 'https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?w=500&h=300&fit=crop',
        'desc': 'Kondisi kritis yang mengancam jiwa. Intervensi medis segera sangat diperlukan termasuk kemungkinan operasi.',
        'tips': [
            '⚠️ DARURAT KESEHATAN – Segera hubungi dokter spesialis',
            'Evaluasi kelayakan operasi bariatrik bersama dokter',
            'Pendekatan multidisiplin: dokter, ahli gizi, psikolog, fisioterapis',
            'Dukungan keluarga dan lingkungan sangat krusial',
            'Tidak ada solusi cepat – butuh komitmen jangka panjang',
        ]
    },
}

BMI_THRESHOLDS = [
    (18.5, 'Insufficient_Weight'),
    (25.0, 'Normal_Weight'),
    (27.5, 'Overweight_Level_I'),
    (30.0, 'Overweight_Level_II'),
    (35.0, 'Obesity_Type_I'),
    (40.0, 'Obesity_Type_II'),
    (999,  'Obesity_Type_III'),
]

def bmi_category(bmi):
    for threshold, cat in BMI_THRESHOLDS:
        if bmi < threshold:
            return cat
    return 'Obesity_Type_III'

@app.route('/')
def index():
    return render_template('index.html', metrics=metrics)

@app.route('/dataset')
def dataset():
    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
    dist = df['NObeyesdad'].value_counts().to_dict()
    return render_template('dataset.html', total=len(df), dist=dist)

@app.route('/about')
def about():
    return render_template('about.html', metrics=metrics)

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender           = request.form.get('gender', 'Male')
        age              = float(request.form.get('age', 25))
        height           = float(request.form.get('height', 1.70))
        weight           = float(request.form.get('weight', 70))
        family_history   = request.form.get('family_history', 'no')
        favc             = request.form.get('high_caloric_food', 'no')
        physical_activity= float(request.form.get('physical_activity', 1))
        water            = float(request.form.get('water_consumption', 2))
        tech             = float(request.form.get('tech_usage', 1))
        transportation   = request.form.get('transportation', 'Public_Transportation')

        bmi = weight / (height ** 2)

        # Build raw input matching dataset encoding
        transport_map = {
            'Automobile': 'Automobile', 'Motorbike': 'Motorbike',
            'Bike': 'Bike', 'Public_Transportation': 'Public_Transportation',
            'Walking': 'Walking'
        }

        raw = {
            'Gender':                         gender,
            'Age':                            age,
            'Height':                         height,
            'Weight':                         weight,
            'family_history_with_overweight': family_history,
            'FAVC':                           favc,
            'FCVC':                           2.0,
            'NCP':                            3.0,
            'CAEC':                           'Sometimes',
            'SMOKE':                          'no',
            'CH2O':                           water,
            'SCC':                            'no',
            'FAF':                            physical_activity,
            'TUE':                            tech,
            'CALC':                           'no',
            'MTRANS':                         transport_map.get(transportation, 'Public_Transportation'),
        }

        input_df = pd.DataFrame([raw])

        # Label-encode categoricals using fitted encoders
        cat_cols = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']
        for col in cat_cols:
            le = le_dict[col]
            val = input_df[col].iloc[0]
            if val in le.classes_:
                input_df[col] = le.transform([val])
            else:
                input_df[col] = 0

        # Reorder columns
        input_df = input_df[feature_names].astype(float)

        # Scale numerical
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        pred_idx = model.predict(input_df)[0]
        proba    = model.predict_proba(input_df)[0]

        pred_class  = target_classes[pred_idx]
        confidence  = round(proba[pred_idx] * 100, 1)
        info        = CLASS_INFO.get(pred_class, CLASS_INFO['Normal_Weight'])

        prob_dist = {
            target_classes[i]: round(p * 100, 1)
            for i, p in enumerate(proba)
        }

        result = {
            'class':        pred_class,
            'label':        info['label'],
            'label_en':     info['label_en'],
            'confidence':   confidence,
            'bmi':          round(bmi, 1),
            'bmi_range':    info['bmi_range'],
            'color':        info['color'],
            'badge':        info['badge'],
            'icon':         info['icon'],
            'risk':         info['risk'],
            'risk_color':   info['risk_color'],
            'image':        info['image'],
            'desc':         info['desc'],
            'tips':         info['tips'],
            'prob_dist':    prob_dist,
            # user inputs for summary
            'gender':       gender,
            'age':          int(age),
            'height':       height,
            'weight':       weight,
            'class_info':   CLASS_INFO,
        }

        return render_template('result.html', result=result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('prediction.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
