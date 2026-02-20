import os
import sys

try:
    from ultralytics import YOLO
    import cv2
except RuntimeError as e:
    if "torchvision::nms does not exist" in str(e):
        print("=" * 80)
        print("HATA: PyTorch/Torchvision versiyon uyumsuzluğu!")
        print("=" * 80)
        print("\nÇÖZÜM: Terminalden şu komutu çalıştır:")
        print("\n  /usr/local/bin/python3.10 -m pip install --upgrade torch torchvision ultralytics")
        print("\nVeya:")
        print("\n  python3 -m pip install --upgrade torch torchvision ultralytics")
        print("\n" + "=" * 80)
        sys.exit(1)
    else:
        raise

if __name__ == '__main__':
    # modelin yolunu buraya girdim
    model = YOLO("/weights/yolov8n.pt")

    # Test görüntülerini buraya girdim (PNG'ye çevrilmiş DICOM dosyaları)
    test_image_folder = "/Users/ergulakgul/Desktop/yeni-veri-seti-cıktı"
    output_cropped_folder = "/Users/ergulakgul/Desktop/yeni-veri-seti-roi-kirpilmis"

    os.makedirs(output_cropped_folder, exist_ok=True)

    # PNG dosyalarını al
    image_files = [f for f in os.listdir(test_image_folder) if f.endswith('.png')]
    total_images = len(image_files)
    processed = 0
    cropped_total = 0
    no_detection = 0

    print(f"Toplam {total_images} PNG dosyası bulundu.")
    print(f"ROI çıkarma işlemi başlıyor...")
    print("=" * 80)

    # test görüntüleri üzerinde ilgi alanı kırp
    for image_name in image_files:
        image_path = os.path.join(test_image_folder, image_name)
        results = model.predict(image_path, save=False, conf=0.8, verbose=False)  # 0.8 güven eşiği

        # Tahmin edilen bounding box'lardan ilgi alanını kırp
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠ Görüntü okunamadı: {image_name}")
            continue

        height, width, _ = image.shape
        boxes = results[0].boxes.xyxy

        if len(boxes) == 0:
            no_detection += 1
            print(f"⚠ Tespit yok: {image_name}")
            continue

        for i, box in enumerate(boxes):  # x_min, y_min, x_max, y_max formatı
            x_min, y_min, x_max, y_max = map(int, box.tolist())
            cropped_image = image[y_min:y_max, x_min:x_max]  # İlgi alanını kırp

            # Kırpılan görüntüyü kaydet
            output_path = os.path.join(output_cropped_folder, f"{os.path.splitext(image_name)[0]}_crop{i}.png")
            cv2.imwrite(output_path, cropped_image)
            cropped_total += 1

        processed += 1
        if processed % 10 == 0:
            print(f"İşlenen: {processed}/{total_images} - Kırpılan: {cropped_total}")

    print("\n" + "=" * 80)
    print("ROI ÇIKARMA TAMAMLANDI!")
    print("=" * 80)
    print(f"Toplam işlenen: {processed}/{total_images}")
    print(f"Toplam kırpılan ROI: {cropped_total}")
    print(f"Tespit edilemeyen: {no_detection}")
    print(f"Çıktı klasörü: {output_cropped_folder}")
    print("=" * 80)

    """

    model = YOLO("yolov8n.pt")  # YOLOv5 Nano modelini kullan
    model.train(
        data = "data/data.yaml",
        epochs=100,  # bunu arttırmam lazım
        workers=4,
        imgsz=640,
        batch=8,
        device='0'  # GPU (0) veya CPU (cpu) kullanımı
        # name="meme_modeli1

    )
"""
