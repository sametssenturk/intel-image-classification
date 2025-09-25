# Intel Görüntü Sınıflandırması (CNN) – Akbank Derin Öğrenme Bootcamp Projesi

## 🔍 Projenin Amacı
Bu proje, Intel Image Classification veri setindeki 6 farklı doğal/şehirsel ortam sınıfını (buildings, forest, glacier, mountain, sea, street) evrişimsel sinir ağları (CNN) ve transfer learning (VGG16) kullanarak yüksek doğrulukta otomatik olarak sınıflandırmayı amaçlar. Hedef; veri keşfi → önişleme → veri artırma → model eğitimi → değerlendirme → açıklanabilirlik (Grad-CAM) → tahmin aşamalarını uçtan uca yeniden üretilebilir bir derin öğrenme pipeline'ı halinde sunmaktır.

## 🗂 Veri Seti Hakkında
- **Kaynak:** Intel Image Classification (Kaggle)
- **Sınıflar:** `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street` (6 sınıf)
- **Toplam Görüntü:** ≈ 25.000
  - Eğitim (seg_train): ~14K
  - Test (seg_test): ~3K
  - Bağımsız tahmin (seg_pred): ~7K
- **Görüntü Boyutu:** 150×150 piksel (yeniden boyutlandırılmış RGB)
- **Format:** JPEG / PNG
- **Dağılım:** Sınıflar arasında görece dengeli; yine de "class_weight" ile olası küçük dengesizlikler hesaba katılmıştır.

## 🧠 Kullanılan Yöntemler ve Pipeline
### 1. Kütüphaneler
- Python, NumPy, Pandas, OpenCV, Matplotlib, Seaborn
- TensorFlow / Keras (CNN katmanları, eğitim callback'leri)
- Scikit-learn (train/validation ayrımı, metrikler)

### 2. Veri Keşfi (EDA)
- Klasör yapısı ve sınıf başına görüntü sayıları otomatik olarak listelendi.
- Her sınıftan rastgele örnek görseller ve sınıf dağılım grafikleri (bar + pie chart) üretildi.

### 3. Veri Önişleme
- Görüntüler okunup RGB'ye çevrildi, 150×150 boyutuna getirildi.
- Etiketler sayısal forma (0–5) ve ardından one-hot formata dönüştürüldü.
- Eğitim/validation ayrımı: stratified (örnekleme dengesini koruyarak) %80 / %20.

### 4. Veri Artırma (Data Augmentation)
Aşırı öğrenmeyi (overfitting) azaltmak ve genelleme gücünü artırmak için dinamik dönüşümler uygulandı:
- rotation_range=25
- width/height_shift_range=0.2
- zoom_range=0.2
- horizontal_flip=True
- brightness_range=[0.8, 1.2]
- shear_range=0.1
- fill_mode='reflect'

VGG16 için ayrı bir pipeline: Görseller 0–255 aralığında bırakılıp Lambda katmanında `tf.keras.applications.vgg16.preprocess_input` çağrıldı.

### 5. Transfer Learning (VGG16)
- Pre-trained `VGG16 (ImageNet)` tabanı `include_top=False` ile yüklendi ve ilk aşamada donduruldu (feature extractor).
- Üst katmanlar (head):
  - GlobalAveragePooling2D
  - Dense(512, relu) + BatchNormalization + Dropout(0.5)
  - Dense(256, relu) + Dropout(0.3)
  - Dense(6, softmax)
- Optimizasyon: Adam (lr=1e-4)
- Kayıp Fonksiyonu: categorical_crossentropy
- Callback'ler:
  - EarlyStopping(patience=8, restore_best_weights=True)
  - ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-7)
  - ModelCheckpoint (en iyi val_accuracy)
- Class weights: Küçük sınıf dengesizliklerini dengelemek için `compute_class_weight` ile hesaplandı.

### 6. Eğitim Takibi ve Değerlendirme
- Eğitim/validation accuracy & loss grafikleri izlendi.
- Test seti üzerinde:
  - Accuracy
  - Confusion Matrix (ham + normalize)
  - Classification Report (precision, recall, F1)
  - Sınıf bazlı F1-score bar grafiği

### 7. Açıklanabilirlik: Grad-CAM
- VGG16'in son konvolüsyon bloğundaki (`block5_conv3`) aktivasyonlar üzerinden ısı haritaları üretildi.
- Her sınıftan temsilî test görüntüsü seçilip modelin karar verdiği kritik bölgeler görselleştirildi.

### 8. Bağımsız Prediction Seti (seg_pred)
- Klasör adlarından bağımsız rastgele görüntüler modele verildi.
- Dosya adından olası gerçek sınıf çıkarımı (string match) ile tahmin karşılaştırmalı gösterildi.


## 🚀 Geliştirme Fikirleri (İleri Çalışma)
- VGG16 katmanlarının bir kısmını yeniden eğiterek (fine-tuning) ek performans artışı.
- Alternatif modeller: EfficientNet, ResNet50, MobileNetV2.
- Veri artırma stratejisinin AutoAugment / RandAugment ile optimize edilmesi.
- Model pruningi ve quantization (mobil dağıtım için).
- MLOps: Eğitim ve izleme pipeline'ının (MLflow, Weights & Biases) eklenmesi.


## 📝 Lisans ve Kullanım
Veri seti Kaggle üzerinden sağlanmaktadır; ilgili lisans koşullarını Kaggle sayfasından kontrol ediniz. Eğitim amaçlı olarak hazırlanmıştır.

## 🙌 Teşekkür
Akbank Derin Öğrenme Bootcamp eğitmenlerine ve açık kaynak topluluğuna teşekkürler.

## Kaggle Notebook Linki : https://www.kaggle.com/code/sametsenturk/intel-image-classification-cnn
