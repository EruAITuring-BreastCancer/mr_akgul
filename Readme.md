# Meme Kanseri Tespit ve BI-RADS Sınıflandırma Projesi

Mamografi görüntülerinden meme kanseri tespiti ve BI-RADS sınıflandırma için geliştirme aşamasındaki bir proje.

## Şu Ana Kadar Yapılanlar

- Veri setlerini topladık (RSNA: PNG, INbreast: DICOM)
- DICOM görüntüleri PNG formatına dönüştürdük
- YOLO ile meme bölgelerini tespit edip kırptık (ROI extraction)
- Basit bir ensemble learning modeli kurduk(BIRADS4 için)

## Dosyalar

**dcm_to_png_converter.py** - DICOM formatındaki INbreast görüntülerini PNG'ye çevirir.

**YOLO.py** - YOLOv8 ile mamografi görüntülerinden meme bölgelerini tespit eder ve kırpar.

**çoklu-sınıflandırma-model** - Başlangıç için BIRADS4 ensemble learning modeli kurduk

## Sıradaki Adımlar

- BI-RADS sınıflandırma sistemi (0-6 kategorileri)
- Model iyileştirmeleri
- Veri artırma
