# AI Phishing Detector

## وصف المشروع
مشروع لاكتشاف الروابط التي قد تحتوي على مؤشرات تصيد احتيالي باستخدام
Machine Learning وتحليل خصائص الرابط.

## فكرة المشروع
يقوم المستخدم بإدخال رابط URL، ثم يتم إرسال الرابط إلى Flask Backend.
بعد ذلك يقوم نموذج Machine Learning بتحليل الرابط وإعطاء درجة خطر ونتيجة.

## مكونات المشروع

- Frontend: HTML + CSS + JavaScript
- Backend: Python + Flask
- Machine Learning: Scikit-learn
- Dataset: بيانات روابط شرعية وروابط تصيد
- Model: Logistic-style text classification باستخدام خصائص نص الرابط

## هيكل المشروع

AI-Phishing-Detector/
│
├── backend/
│   └── app.py
│
├── frontend/
│   └── index.html
│
├── model/
│   ├── dataset_real.csv
│   ├── dataset.csv
│   ├── phishing_model.pkl
│   ├── train_model.py
│   └── vectorizer.pkl
│
├── requirements.txt
└── README.md

## طريقة التشغيل

### 1. تثبيت المكتبات

```bash
pip install -r requirements.txt