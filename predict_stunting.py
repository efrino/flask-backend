import os
import numpy as np
import pandas as pd
import tensorflow as tf

def get_normal_values(age, gender, normal_values_df):
    """
    Mendapatkan nilai tinggi badan dan berat badan normal
    berdasarkan umur (bulan) dan jenis kelamin ('Laki-laki', 'Perempuan').
    Jika usia tidak ada, cari usia terdekat.
    """
    try:
        # Encode gender: 0 = Laki-laki, 1 = Perempuan
        gender_encoded = 0 if gender == 'Laki-laki' else 1
        age = int(age)

        # Cari baris yang persis sama
        normal_row = normal_values_df[
            (normal_values_df['Umur (bulan)'] == age) &
            (normal_values_df['Jenis Kelamin'] == gender_encoded)
        ]
        if not normal_row.empty:
            return (
                normal_row['Tinggi Badan (cm)'].iloc[0],
                normal_row['Berat Badan (kg)'].iloc[0]
            )

        # Jika tidak ada, cari usia terdekat
        available_ages = normal_values_df[
            normal_values_df['Jenis Kelamin'] == gender_encoded
        ]['Umur (bulan)'].unique()

        if len(available_ages) == 0:
            print(f"Tidak ada data untuk jenis kelamin {gender}.")
            return None, None

        closest_age = available_ages[np.argmin(np.abs(available_ages - age))]
        normal_row = normal_values_df[
            (normal_values_df['Umur (bulan)'] == closest_age) &
            (normal_values_df['Jenis Kelamin'] == gender_encoded)
        ]
        if not normal_row.empty:
            print(f"Menggunakan data usia terdekat: {closest_age} bulan.")
            return (
                normal_row['Tinggi Badan (cm)'].iloc[0],
                normal_row['Berat Badan (kg)'].iloc[0]
            )
        return None, None
    except Exception as e:
        print(f"Error pada get_normal_values: {e}")
        return None, None

def get_nutrition_recommendation(stunting_status, age):
    """
    Mengembalikan rekomendasi gizi berbasis status stunting dan umur.
    """
    recommendations = {
        'Severely Stunted': {
            'description': 'Anak mengalami stunting berat. Perlu intervensi gizi intensif dan konsultasi medis.',
            'food': ['Susu tinggi protein (sesuai usia)', 'Bubur kacang hijau', 'Telur rebus', 'Ikan salmon/teri', 'Sayur bayam'],
            'frequency': '6-8 kali/hari (porsi kecil, 50-100g per porsi)',
            'supplements': 'Konsultasi dokter untuk suplemen zat besi, zinc, dan vitamin A.',
            'notes': 'Pantau berat dan tinggi setiap 2 minggu. Libatkan puskesmas.'
        },
        'Stunted': {
            'description': 'Anak mengalami stunting ringan. Perbaikan gizi diperlukan untuk pertumbuhan optimal.',
            'food': ['Susu full cream', 'Nasi tim dengan ayam', 'Tempe goreng', 'Brokoli kukus', 'Pisang'],
            'frequency': '5-6 kali/hari (porsi sedang, 100-150g per porsi)',
            'supplements': 'Multivitamin anak (dosis sesuai usia).',
            'notes': 'Tingkatkan asupan protein dan kalori. Konsultasi gizi jika tidak membaik dalam 1 bulan.'
        },
        'Normal': {
            'description': 'Pertumbuhan anak normal. Pertahankan pola makan seimbang.',
            'food': ['Nasi/ubi', 'Ayam/daging sapi', 'Sayur kolplay', 'Tahu', 'Buah apel'],
            'frequency': '4-5 kali/hari (porsi sesuai usia, 150-200g per porsi)',
            'supplements': 'Tidak diperlukan kecuali defisiensi spesifik.',
            'notes': 'Pastikan variasi makanan dan aktivitas fisik cukup.'
        },
        'Tall': {
            'description': 'Anak memiliki pertumbuhan di atas rata-rata. Pastikan gizi seimbang untuk mendukung perkembangan.',
            'food': ['Roti gandum', 'Ikan tuna', 'Kacang almond', 'Wortel', 'Mangga'],
            'frequency': '4-5 kali/hari (porsi sesuai usia, 150-200g per porsi)',
            'supplements': 'Tidak diperlukan kecuali aktivitas fisik tinggi.',
            'notes': 'Monitor BMI untuk mencegah obesitas.'
        }
    }

    # Modifikasi untuk bayi < 6 bulan (ASI eksklusif)
    if age < 6:
        for key in recommendations.keys():
            recommendations[key]['food'] = ['ASI eksklusif atau susu formula khusus']
            recommendations[key]['frequency'] = '7-12 kali/hari (sesuai kebutuhan)'

    return recommendations.get(stunting_status, {
        'description': 'Status tidak dikenali.',
        'food': [],
        'frequency': '',
        'supplements': '',
        'notes': ''
    })

def predict_stunting(age, gender, height, weight,
                     model_path='model/stunting_model.h5',
                     mean_path='model/scaler_mean.npy',
                     std_path='model/scaler_std.npy',
                     normal_values_path='model/normal_values.csv'):
    """
    Fungsi utama prediksi stunting.
    Mengembalikan dictionary hasil prediksi, confidence, rekomendasi, dan analisis tambahan.
    """
    try:
        # Validasi input dasar
        if not isinstance(age, (int, float)) or age < 0:
            raise ValueError("Umur harus berupa angka positif.")
        if gender not in ['Laki-laki', 'Perempuan']:
            raise ValueError("Jenis kelamin harus 'Laki-laki' atau 'Perempuan'.")
        if not isinstance(height, (int, float)) or height <= 0:
            raise ValueError("Tinggi badan harus berupa angka positif.")
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError("Berat badan harus berupa angka positif.")

        # Cek keberadaan file
        for file_path in [model_path, mean_path, std_path, normal_values_path]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

        # Load model
        model = tf.keras.models.load_model(model_path)

        # Load mean & std untuk normalisasi
        mean_values = np.load(mean_path)
        std_values = np.load(std_path)

        # Load dataset nilai normal (Umur, Gender encoded, TB, BB)
        normal_values_df = pd.read_csv(normal_values_path)
        if normal_values_df.empty:
            raise ValueError("File normal_values.csv kosong atau tidak valid.")

        # Encode gender
        gender_encoded = 0 if gender == 'Laki-laki' else 1

        # Siapkan input data
        input_data = np.array([[age, gender_encoded, height, weight]], dtype=np.float32)

        # Normalisasi
        input_data = (input_data - mean_values) / std_values

        # Prediksi dengan model
        prediction = model.predict(input_data)
        predicted_class_index = np.argmax(prediction, axis=1)[0]

        raw_confidence = prediction[0][predicted_class_index] * 100
        # Skala confidence ke maksimum 90%
        confidence = min(raw_confidence * 0.9, 90.0)
        confidence = round(confidence)

        # Label prediksi
        labels = {
            0: 'Severely Stunted',
            1: 'Stunted',
            2: 'Normal',
            3: 'Tall'
        }
        stunting_status = labels.get(predicted_class_index, 'Unknown')

        # Rekomendasi gizi
        nutrition_recommendation = get_nutrition_recommendation(stunting_status, age)

        # Dapatkan nilai normal TB & BB dari dataset
        tb_normal, bb_normal = get_normal_values(age, gender, normal_values_df)

        additional_info = {}

        if tb_normal is not None and bb_normal is not None:
            # Hitung persentase perbedaan (lebih pendek/tinggi, lebih ringan/berat)
            persentase_tb = round(((tb_normal - height) / tb_normal) * 100, 2)
            persentase_bb = round(((bb_normal - weight) / bb_normal) * 100, 2)

            threshold = 0.1  # ambang batas perbedaan dianggap kecil

            # Penjelasan TB
            if abs(persentase_tb) < threshold:
                tb_explanation = "Tinggi badan anak sesuai dengan rata-rata untuk usianya."
            elif persentase_tb > 0:
                tb_explanation = f"Tinggi badan anak {persentase_tb}% lebih rendah dari rata-rata normal."
            else:
                tb_explanation = f"Tinggi badan anak {abs(persentase_tb)}% lebih tinggi dari rata-rata normal."

            # Penjelasan BB
            if abs(persentase_bb) < threshold:
                bb_explanation = "Berat badan anak sesuai dengan rata-rata untuk usianya."
            elif persentase_bb > 0:
                bb_explanation = f"Berat badan anak {persentase_bb}% lebih rendah dari rata-rata normal."
            else:
                bb_explanation = f"Berat badan anak {abs(persentase_bb)}% lebih tinggi dari rata-rata normal."

            additional_info['normal_height'] = tb_normal
            additional_info['normal_weight'] = bb_normal
            additional_info['height_diff_percentage'] = persentase_tb
            additional_info['weight_diff_percentage'] = persentase_bb
            additional_info['height_explanation'] = tb_explanation
            additional_info['weight_explanation'] = bb_explanation
        else:
            additional_info['normal_height'] = None
            additional_info['normal_weight'] = None
            additional_info['height_diff_percentage'] = None
            additional_info['weight_diff_percentage'] = None
            additional_info['height_explanation'] = "Data normal tidak ditemukan."
            additional_info['weight_explanation'] = "Data normal tidak ditemukan."

        # Hasil akhir
        result = {
            'status': stunting_status,
            'confidence': confidence,
            'nutrition_recommendation': nutrition_recommendation,
            'additional_info': additional_info
        }

        return result

    except Exception as e:
        return {
            'error': str(e)
        }
