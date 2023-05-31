# Emotion Detection using CNN
### Proje Hakkında

- Bu proje, Evrişimli Sinir Ağı (CNN) ve OpenCV kullanılarak insan yüz ifadesinin algılanması hakkındadır.
- Bu proje Python 3.6.8 ve Keras 2.2.4'te uygulanmıştır.
- Bu proje, 16 GB RAM ve i7 işlemci ile Windows 11'de test edilmiştir.

### Proje Yapısı

- `data` klasörü, modelin eğitimi için kullanılan veri kümesini içerir.
- `model` klasörü, eğitilmiş model yapısı ve ağırlıklarını içerir.
- `TestDlib.py`, modeli test etmek için çalıştırılacak ana dosyadır.
- `TrainModel.py`, modeli eğitmek için çalıştırılacak ana dosyadır.
- `TestCascade.py`, modeli test etmek için çalıştırılabilir fakat doğruluk oranı daha az bir yüz tanıma algoritması kullanır.

### Projeyi nasıl çalıştırabilirim

- Tüm kütüphaneleri yükleyin.
- Bir `data` klasörü oluşturup indirdiğiniz FER2013 veri setini `test` ve `train` olarak iki ayrı klasörde ekleyin.
- Modeli eğitmek için `python TrainModel.py` çalıştırın.
- Eğitilmiş modeli kullanmak için çıktıları `model` klasörüne ekleyin.
- Modeli test etmek için `python TestDlib.py` çalıştırın.

### FER2013 data seti
- https://www.kaggle.com/msambare/fer2013

### Evrim Arda Kalafat - 2023
