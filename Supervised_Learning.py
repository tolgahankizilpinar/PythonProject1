
"""
ML(Machine Learning)

Supervised Learning (Denetimli Öğrenme= Feature + Label)
Modelin hem giriş verileri(x)
hem de bu verilere ait doğruları cevapları (Etiketleri[Label]: Y)

Temel Yapı:
    X(Features/Özellikler) --> MODEL --> y (Label/Etiketler)

Örneğin:
- E-Posta spam mı değil mi ?
- Ev fiyatları ne kadar olur ?
- Bu müşteriye borç beyaz eşya verilir mi ?

Öğrenme türü olan LABEL vardır


Supervised Learning de Label vardır.
Model geçmişteki doğru cevapları öğrenir. Bu örnekte Label 0=Kaldı, 1=Geçti



"""


"""
Bir öğrencinin:
1-) Günlük çalışma saati
2-) Derse katılım yüzdesi
bu bilgilere bakarak sınavı geçip geçmeyeceğini tahmin edelim. 

Label:
    0 : Kaldı
    1 : Geçti
    
Kullanılan algoritma:
    Logistic Regression 
    

Kurulum:
    pip install numpy scikit-learn
    python -c "import numpy; import sklearn; print('Kurulum başarılı')"
    
    python -m pip install -r requirements.txt
    
    -m → module
    -c → command
"""


import numpy as np
from sklearn.linear_model import LogisticRegression

def main():
    # x (Features)
    x = np.array([
        [1,30],
        [2,40],
        [2,50],
        [3,55],
        [4,60],
        [5,65],
        [6,75],
        [7,85],
        [8,90],
        [9,95]
    ])


    # y(Label)
    # 0 = Kaldı, 1 = Geçti

    # NOT: Supervised Learning'in en önemli özelliği; x verileriyle birlikte y labellarının bulunmasıdır.

    y = np.array([
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1
    ])

    print("*******  SUOERVISED LEARNING  *******")
    print("\nx - Öğrenci Özellikleri(Features)")
    print(x)


    print("\ny - Label(Etiketler)")
    print(y)

    """
    Model Oluşturma
    LogisticRegression bir sınıflandırma algoritmasıdır.
    Burada 2 tane sınıf vardır, bunlar 0 --> Kaldı , 1 --> Geçti
    LogisticRegression, iki veya daha fazla sınıfın hangisine ait olduğunu tahmin etmek için kullanılan sınıfın algortmasıdır.
    """

    model = LogisticRegression()

    # Modeli Eğitim
    # Model hem özellikleri hem de doğru cevapları görsün
    # Bu ilişkide çalışma saati + Katılım oranı --> Geçti/Kaldı
    model.fit(x, y)


    # Instance
    # Örnek: Öğrenci 6 Saat çalışıyor, Derse Katılım %80 olsun

    newStudent = np.array([[6, 80]])


    # Tahmin
    prediction = model.predict(newStudent)[0]


    # Tahnim Olasılıkları
    probabilities = model.predict_proba(newStudent)[0]

    print("\nModel Tahmini: ", prediction)


    # Conditional
    if prediction == 1:
        print("Sonuç: Öğrencinin GEÇMESİ bekleniyor")
    else:
        print("Sonuç: Öğrencinin KALMASI bekleniyor")


    print("\nOlasılıklar")
    print(f"Kalma Olasılığı: , %{probabilities[0] * 100:.2f}")
    print(f"Geçme Olasılığı: , %{probabilities[1] * 100:.2f}")


if __name__ == '__main__':
    main()



