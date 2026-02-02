# Meme Kanseri Tespit ve BI-RADS Sınıflandırma Projesi

Mamografi görüntülerinden meme kanseri tespiti ve BI-RADS sınıflandırma için geliştirme aşamasındaki bir proje.

## Şu Ana Kadar Yapılanlar

1. Veri setlerini topladık (RSNA: PNG, INbreast: DICOM)
2. DICOM görüntüleri PNG formatına dönüştürdük
3. YOLO ile meme bölgelerini tespit edip kırptık (ROI extraction, %82 başarı)
4. Basit bir binary classification modeli kurduk

## Dosyalar

**dcm_to_png_donusturme.py**
DICOM formatındaki INbreast görüntülerini PNG'ye çevirir.

**YOLO.py**
YOLOv8 ile mamografi görüntülerinden meme bölgelerini tespit eder ve kırpar.

**modelleme.py**
Binary classification yapan basit CNN modeli. Kırpılmış görüntüleri sınıflandırır.

**Recognition_Manuel.py**
Deneysel alternatif model. Şu an kullanılmıyor.

**NBCL_Binary_Egitim.ipynb**
Binary classification eğitim dosyası.

## Kurulum

```bash
pip install -r requirements.txt
```

## Sıradaki Adımlar

- BI-RADS sınıflandırma sistemi (0-6 kategorileri)
- Model iyileştirmeleri
- Veri artırma

