# maldi-shift

MALDI-TOF kütle spektrumlarından antimikrobiyal direnç öngörüsü:
merkezler arası ve zamansal dağılım kayması altında güvenilirlik.

## Veri

DRIAMS (Weis ve ark., Dryad, doi:10.5061/dryad.bzkh1899q), 19 Ağustos 2025 sürümü.
Dört İsviçre merkezi: A = Basel Üniversite Hastanesi, B = Basel-Land Kanton Hastanesi,
C = Aarau Kanton Hastanesi, D = Viollier AG laboratuvarı.
Lisans: CC0. Ham veri bu depoda yer almaz.

## Envanter bulguları

| Merkez | İzolat | Etiket | Yıl |
|---|---|---|---|
| DRIAMS-A | 111.257 | 563.826 | 2015-2018 |
| DRIAMS-B | 2.386 | 35.920 | 2018 |
| DRIAMS-C | 4.696 | 54.684 | 2018 |
| DRIAMS-D | 10.369 | 112.545 | 2018 |

## Veri hazırlığında tespit edilen üç tuzak

1. **Türetilmiş fenotipler.** EUCAST uzman kuralları nedeniyle bazı ilaç sütunları
   birbirinin birebir kopyası (korelasyon = 1.000). Grup büyüklüğü merkeze göre
   değişiyor: DRIAMS-B'de üçlü, DRIAMS-A'da sekizli. Bildirilen görev sayısı şişkin.

2. **Hasta bazlı sızıntı.** DRIAMS-A 2018'de 30.069 izolat / 11.844 hasta;
   hastaların %39,3'ünde birden fazla izolat var. İzolat bazlı bölme dahili
   performansı şişirir. Gruplama anahtarı: gerçek patient_no, kimliksizlerde
   izolat kodu (kayıtların %26,5'i `nan_yıl_tür` kalıbıyla yapay olarak üretilmiş).

3. **Yıl klasörleri takvim yılı değil.** `2018` dosyası 2017-02-01 ile 2018-08-31
   arasını kapsıyor. Zamansal bölme `acquisition_date` üzerinden kurulmalı.

## Dosyalar

- `inventory.py` — envanter çıkarıcı
- `outputs/ozet_tur_ilac.csv` — merkez × tür × ilaç kırılımı
- `outputs/uygun_ciftler.csv` — n ≥ 100 ve direnç oranı %5-95 olan çiftler
- `outputs/ortak_ciftler.csv` — merkezler arası ortak çift matrisi

## Ortam

Python 3.11, PyTorch 2.11+cu128, LightGBM 4.7, PyWavelets 1.8, SHAP 0.51
