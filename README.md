# Defining Self-Attention from Scratch

Bu proje, modern yapay zeka modellerinin temel taşı olan **Scaled Dot-Product Attention** mekanizmasının, herhangi bir derin öğrenme kütüphanesi (PyTorch, TensorFlow vb.) kullanılmadan, sadece Python'un standart kütüphaneleri (`math`, `random`) ile sıfırdan geliştirilmiş halidir.

## Projenin Amacı
Derin öğrenme modelleri genellikle birer "kara kutu" (black-box) gibi çalışır. Bu çalışma, Transformer mimarisinin içindeki matematiksel işlemleri —matris çarpımlarından aktivasyon fonksiyonlarına kadar— şeffaf bir şekilde görmenizi sağlamak amacıyla hazırlanmıştır.

## Temel Özellikler
* **Saf Python Implementasyonu:** Harici bağımlılık gerektirmez.
* **Şeffaf Matematiksel İşlemler:**
    * `matmul`: Lineer cebir kurallarına uygun matris çarpımı.
    * `transpose`: Matris transpoz hesaplama.
    * `softmax`: Olasılık dağılımı hesaplama.
    * `relu`: Doğrusal olmayan aktivasyon katmanı.
* **Encoder Bloğu Simülasyonu:** Girdiden çıktıya kadar bir Transformer Encoder bloğunun veri akışını (`Embedding` -> `Linear` -> `Activation` -> `Attention` -> `Linear`) simüle eden `NeuralModel` yapısı.

## Matematiksel Arka Plan
Model, standart Transformer makalesinde (*Attention Is All You Need*) tanımlanan şu formülü temel alır:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

## Dosya Yapısı
* `self_attention.py`: Temel matematiksel operasyonlar ve Attention mekanizmasının çekirdek kodları.
* `neural_model.py`: Modelin mimarisi, ağırlıkların rastgele başlatılması ve *forward pass* mantığı.

## Nasıl Çalıştırılır?
Projenin çalışması için Python yüklü olması yeterlidir. Herhangi bir `pip install` işlemine gerek yoktur.

```bash
# Modelin çıktısını görmek için
python neural_model.py
