# Laporan Proyek Machine Learning - Jelang Ramadhan

## Domain Proyek

Penelitian sebelumnya yang dilakukan oleh Suripto, Rahmanita, dan Kirana (2022) dalam artikel _Heart Disease Classification Using Data Science Tools – A Review and Hands-on_ menganalisis klasifikasi penyakit jantung menggunakan berbagai algoritma _machine learning_, seperti _Decision Tree Model_, _Support Vector Machine_, _Naïve Bayes Classifier_, _Random Forest_, _k-Nearest Neighbor_, dan _Logistic Regression_. Penelitian tersebut menggunakan dataset _heart-failure-prediction_ dari Kaggle dan menerapkan metode _test and score_ dengan pengulangan sebanyak 20 kali serta proporsi data latih sebesar 95%. Hasil pengujian menunjukkan bahwa metode _Logistic Regression_ memiliki tingkat _Classification Accuracy_ (CA) tertinggi sebesar 87,9%, sedangkan algoritma _k-Nearest Neighbor_ memiliki CA terendah dengan persentase 70,6%.

Pada penelitian ini, klasifikasi penyakit jantung dilakukan menggunakan dataset yang sama, tetapi dengan pendekatan _Deep Learning_ menggunakan _Neural Network_. Tujuan penelitian ini adalah meningkatkan akurasi klasifikasi untuk menemukan model terbaik dalam mendeteksi penyakit jantung. Implementasi model ini diharapkan dapat membantu tenaga medis dalam melakukan deteksi dini penyakit jantung berdasarkan berbagai fitur, seperti usia, jenis kelamin, tipe nyeri dada, kadar kolesterol, dan parameter lainnya.

### **Referensi**

Suripto, R., Rahmanita, R. N. dan Kirana, A. S. (2022) ‘Heart disease classification using data science tools – a review and hands-on’, _Management and Industrial Engineering Binus University_. Tersedia di: [https://mie.binus.ac.id/2022/08/25/heart-disease-classification-using-data-science-tools-a-review-and-hands-on/](https://mie.binus.ac.id/2022/08/25/heart-disease-classification-using-data-science-tools-a-review-and-hands-on/) (Diakses: 15 Maret 2025).

## Business Understanding

### Problem Statements

- Bagaimana memprediksi risiko serangan jantung pada pasien untuk mengurangi waktu tunggu diagnosis?
- Bagaimana memilihi model yang dapat mengklasifikasi pasien dengan penyakit jantung tersebut dengan lebih baik

### Goals

- Membuat model yang dapat mengklasifikasikan seseoarag mengalami penyakit jantung seakurat mungkin berdasarkan fitur-fitur yang ada.
- Membuat beberapa model dan memilih model dengan metrik evaluasi terbaik diantaranya.

### Solution statements

- Dalam merangkai model menggunakn ANN akan menggunakan beberapa jenis layes seperti Dense, BatchNormalization, Dropout. serta menggukan activation relu dan sigmoid (karena klasifikasi biner). Tidak lupa untuk melakuakn hyperparameter tuning untuk memaksimalkan model.
- Merangkai model machine learning Logistic Regression sesuai dengan referensi artikel yang menemukan model terbaik untuk mengklasifikasikan pada kasus ini.
- Dalam Data Preparation, kami tidak menghapus data outlier karena data tersebut penting, karena bisa jadi outlier tersebut adalah salah satu penyebab seseorang terkena penyakit jantung.

## Data Understanding

Data yang kami gunakan berjudul "heart-failure-prediction" yang berisikan data kesehatan seseorang yang mengalami penyakit jantung dan tidak. ada 11 fitur dalam dataset tersebut, seperti data usia, kelamin, kadar kolesterol, gula darah dan yang lain - lain. Dataset mempunyai jumlah baris 918, tidak mempunyai data duplicated, missing value, dan inconsistent value. Memiliki 5 data kategorikal dan 6 data numerik.

Dalam proses EDA, didalam dataset mengandung outlier untuk beberapa kolom numerik, seperti kolom RestingBP, Cholesterol, MaxHR, dan Oldpeak. Data outlier tersebut tidak dihapus karena dianggap penting pada kasus ini.

Untuk mendowload dataset tesbut: [Kaggle.com](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:

- Age : Usia (Tahun)
- Sex : Jenis kelamin (M: Laki-laki/ F: Perempuan)
- ChestPainType : Tipe nyeri dada (TA: Angina Khas, ATA: Angina Atipikal, NAP: Nyeri Non-Anginal, ASY: Asimptomatik)
- RestingBP : Tekanan darah (mmHg)
- Cholesterol : Kadar kolesterol (mm/dl)
- FastingBS : Gula darah saat puasa (1: jika FastingBS > 120 mg/dl, 0: selainya)
- RestingECG : Hasil elektrokardiogram istirahat (Normal: Normal, ST: memiliki kelainan gelombang ST-T (inversi gelombang T dan / atau elevasi ST atau depresi > 0,05 mV), LVH: menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri dengan kriteria Estes)
- MaxHR : Detak jantung maksimum tercapai (antara 60 dan 202)
- ExerciseAngina : Angina yang diinduksi latihan (Y: Ya, N: Tidak)
- Oldpeak : Oldpeak = ST (Nilai numerik diukur dalam depresi)
- ST_Slope : Kemiringan segmen ST latihan puncak (Atas: upsloping, Datar: datar, Bawah: downsloping)
- HeartDisease : Hasil (1: penyakit jantung, 0: Normal)

## Data Preparation

Data preparation merupakan tahapan penting dalam proses pengembangan model machine learning. Ini adalah tahap di mana kita melakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan. Ada beberapa tahapan yang umum dilakukan pada data preparation, antara lain, seleksi fitur, transformasi data, feature engineering, dan dimensionality reduction. Pada kasus kali ini tahapan Data Preparation sebagai berikut:

1. Menganalisi jumlah data dan tipenya menggunakan .info() untuk mengetahui kemungkinan data duplicated, missing value, dan inconsistent value.
2. Encoding fitur **(One Hot Encoding)** data kategori agar fitur dapat dibaca oleh model.
3. Cek hasil encoding untuk memastikan jenis data sudah sesuai.
4. Standarisasi kolom numerik dilakukan menggunakan modul StandardScaler dari library sklearn.preprocessing. Teknik ini dikenal dengan **Z-score Normalization**, tujuannya adalah mengubah data numerik sehingga memiliki rata-rata (mean) 0 dan standar deviasi 1. Rumus matematikanya sebagi berikut: \
   $Z_{score}(X')=\frac{X-\mu}{\sigma}$ \
   Dimana,

   - Z Score ($X'$) : nilai setelah standarisasi
   - $X$ : Nilai asli suatu fitur
   - $\mu$ : Rata
   - rata (mean) dari fitur tersebut - $\sigma$ : Standar deviasi dari fitur tersebut

5. Memecah ke dalam Train dan Test Set untuk memisahkan data yang digunakan untuk proses training dan testing data.

## Modeling

### Model 1 - ANN (Artificial Neural Network)

Artificial Neural Network (ANN) meniru cara kerja otak manusia dalam memproses informasi. ANN terdiri dari sejumlah neuron buatan yang diorganisir dalam beberapa lapisan (Input Layer, Dense Layer(s), dan Output Layer ). Berikut adalah cara kerjanya secara umum:

1. Input Layer

   - Data yang diberikan berjenis numerik
   - Setiap fitur direpresentasikan sebagai neuron dalam Input Layer, lalu diteruskan ke Dense Layer

2. Forward Propagation

   - Informasi dari Input Layer diteruskan ke setiap neuron di Dense Layer.
   - Setiap neuron menghitung nilai output dengan rumus sebagai berikut: \
     $y=f(WX+b)$\
     dimana,
     - f : fungsi aktivasi
     - W : Bobot
     - X : Nilai input
     - b : Bias
   - Output dari satu lapisan menjadi input bagi lapisan berikutnya.

3. Loss Function

   - Hasil dari Output layer dibandingkan dengan target sebenarnya dan menghitung niali errornya dengan loss function.

4. Backpropagation

   - ANN menyesuaikan bobot dan bias untuk mengurangi error.
   - Gradien error dihitung menggunakan turunan fungsi loss terhadap bobot.
   - Algoritma optimasi (misalnya, Gradient Descent) digunakan untuk memperbarui bobot.

5. Training

   - Langkah forward propagation dan backpropagation diulang selama beberapa epoch.
   - Model terus belajar dari data hingga mencapai konvergensi atau error minimal.

6. Validasi dan Evaluasi Model
   - Setelah pelatihan, model diuji dengan data yang belum pernah dilihat (data validasi atau testing).
   - Metrik evaluasi seperti Akurasi, Precision, Recall, dan F1-score digunakan untuk mengukur performa model.

Pada proyek ini, parameter yang digunakan sebagai berikut:

1. Sequential Layer : sesuai untuk tumpukan lapisan di mana tiap lapisan memiliki tepat satu tensor masukan dan satu tensor keluaran.

2. Dense(64, activation='relu', input_shape=(x_train.shape[1],))

   - Memiliki jumlah parameter neuron sebanyak 64 unit
   - ReLu (Rectified Linear Unit) sebagai parameter fungsi aktivasinya untuk meningkatkan kemampuan non-linearitas jaringan.
   - Mengambil input sesuai dengan fitur pada training set (sebagai input layer)

3. BatchNormalization()

   - Parameternya defult
   - lapisan ini digunakan untuk menormalkan input sebelum masuk ke lapisan berikutnya

4. Dropout(0.3)

   - Jumlah parameter untuk menonaktifkan neuron sebanyak 30%
   - Lapisan ini digunakan untuk membantu model menjadi lebih robust dan mencegah overfitting.

5. Dense(64, activation='relu')

   - Memiliki jumlah parameter neuron sebanyak 64 unit
   - ReLu (Rectified Linear Unit) sebagai parameter fungsi aktivasinya untuk meningkatkan kemampuan non-linearitas jaringan.

6. Dense(1, activation='sigmoid')
   - Memiliki jumlah parameter neuron sebanyak 64 unit (sebagai output layer)
   - Sigmoid sebagai parameter fungsi aktivasi untuk menghasilkan output probabilitas (antara 0 dan 1)

Kelebihan dan Kekurangan ANN

- Kelebihan

  - Kemampuan dalam mempelajari pola kompleks
  - Generalisasi yang baik
  - Kemampuan adaptasi yang baik
  - Pemrosesan paralel
  - Fleksibel dalam berbagai aplikasi

- Kekurangan
  - Membutuhkan banyak data
  - Proses training lama
  - Sensitif terhadap hyperparameter
  - Rentan terhadap overfitting

### Model 2 - Logistic Regression

Logistic Regression didefinisikan sebagai algoritme pembelajaran mesin yang diawasi (supervised learning) yang menyelesaikan tugas klasifikasi biner dengan memprediksi probabilitas hasil, peristiwa, atau pengamatan. Logistic Regression bekerja dengan menggunakan persamaan linear, tetapi hasilnya kemudian dikonversi ke rentang 0 hingga 1 menggunakan fungsi sigmoid. Cara kerjanya sebagai berikut:

1. Persamaan dasar Logistic Regression

   - Logistic Regression menggunakan persamaan linear dalam memproses data, tetapi menghasilkan output 0 dan 1 (sigmoid function)

     - Persamaan linear \
       $z=WX+b$\
       dimana,

       - $z$ : Nilai output
       - $W$ : Bobot
       - $X$ : Nilai input
       - $b$ : Bias

     - Sigmoid Function \
       $P(Y=1)=\sigma(z)=\frac{1}{1+e^{-z}}$\
       dimana,
       - $P(Y=1)$ : Probabilitas output bernilai
       - $\sigma(z)$ : Fungsi sigmoid dengan input $z$ (persamaan linear)
       - $e$ : Bilangan Euler (~2.71828)

2. Training Data

   - Inisiasi bobot (bisa bernilai nol atau nilai acaka)
   - Forward propagation\
     Data Input dimasukkan kedalam persamaan linear dan ouputnya dimasukkan ke Sigmoid Function untuk mendapatkan probabilitasnya
   - Menghitung Loss (Binary Cross Entropi)\
     Loss = $-\frac{1}{n}\sum[Y log(\hat{Y})+(1-Y)log(1-\hat{Y})]$
   - Backpropagation dan optimasi\
     Model memperbarui nilai bobot ($W$) menggunakan Gradient Descent\
     $W=W-\alpha\frac{\partial Loss}{\partial W}$\
     dimana, $\alpha$ : Learning rate

3. Prediksi\
   Setelah model selesai dilatih, model akan diuji dengan data baru dan akan menentukan probabilitasnya.

Pada proyek ini, parameter yang digunakan adalah nilai **defult** untuk semua parameter

Kelebihan dan Kekurangan Logictic Regression

- Kelebihan

  - Komputasi ringan
  - Mudah diinterpretasikan
  - Tidak memerlukan banyak data

- Kekurangan
  - Kurang cocok menangani data non-linear
  - Tergantung pada fitur
  - Rentan terhadap outlier

## Evaluation

Metrik evaluasi yang digunakan untuk proyek kali ini adalah **Accuracy, Precission, Recall, dan F1 Score**

- Accuracy = $(\frac{TP+TN}{TP+TN+FP+FN})$ \
  Proporsi banyaknya data yang diprediksi dengan benar terhadap total data yang diprediksi
- Precission = $\frac{TP}{TP+FP}$ \
  Mengukur seberapa banyak data yang prediksi ke suatu kelas yang memang benar masuk kelas tersebut
- Recall = $\frac{TP}{TP+FN}$ \
  Mengukur kemampuan model untuk mendeteksi semua data suatu kelas di antara semua data yang benar-benar masuk kelas tersebut
- F1-Score = $2*\frac{Recall*Precission}{Recall + Precission}$ \
  Metrik ini diukur dari rata-rata harmonis precision dan recall, sehingga memberikan gambaran yang lebih seimbang tentang kinerja model terutama ketika terdapat ketidakseimbangan antara jumlah kelas positif dan negatif

**Keterangan**

- **True Positive (TP)**: Jumlah data dari kelas positif yang diprediksi benar (masuk kelas positif)
- **True Negative (TN)**: Jumlah data dari kelas negatif yang diprediksi benar (masuk kelas negatif)
- **False Positive (FP)**: Jumlah data dari kelas negatif yang salah prediksi (seharusnya negatif namun diprediksi positif)
- **False negative (FN)**: Jumlah data dari kelas positif yang salah prediksi (seharusnya positif namun diprediksi negatif)

### Logistic Regression

Setelah melakukan proses training data, ditemukan nilai dari matriks evaluasi sebagai berikut,

- **Accuracy** : $88\%$
- **Precission** : $89\%$
- **Recall** : $89\%$
- **F1 Score** : $89\%$

### ANN

Setelah melakukan proses training data, ditemukan nilai dari matriks evaluasi sebagai berikut,

- **Accuracy** : $90\%$
- **Precission** : $90\%$
- **Recall** : $90\%$
- **F1 Score** : $90\%$

Dengan demikian, model terbaik pada proyek kali ini adalah menggunakan **ANN**

### Conclusion

Setelah melakukan berbagai tahapan pada proyek ini, mulai dari tahap Business Understanding, Data Understanding, Data Preparation, Modeling, dan Evaluation untuk melakukan klasifikasi penyakit jantung dengan menggunakan dataset _heart-failure-prediction_ dari Kaggle. Ditemukan bahwa pendekatan dengan ANN memberikan hasil lebih baik dibandingkan dengan Logistic Regression.
Pertanyaan yang dijabarkan pada bagian **Problem Stantments** dapat dijawab semua dengan baik pada akhirnya, dengan menghasilkan **Goals** yang memuaskan, berikut detailnya:

- Goal 1 : Dalam proyek ini telah membuat model yang dapat mengklasifikasikan seseorang mengalami penyakit jantung seakurat mungkin untuk membantu mempercepat waktu tunggu diagnosis

- Goal 2 : Dalam proyek ini telah membuat 2 macam model dengan pendekatan yang berbeda. Pertama, pendekatan Deep Learning menggunakan ANN (Artificial Neural Network) dan yang kedua dengan Machine Learning menggunakan Logistic Regression. Hasil yang didapatkan sebagai berikut:

  - Model 1 - ANN (Artificial Neural Network):\
    **Accuracy** : $90\%$\
    **Precission** : $90\%$\
    **Recall** : $90\%$\
    **F1 Score** : $90\%$

  - Model 2 - Logistic Regression\
    **Accuracy** : $88\%$\
    **Precission** : $89\%$\
    **Recall** : $89\%$\
    **F1 Score** : $89\%$

- Dengan demikian dapat disimpulkan bahwa model dengan pendekatan Deep Learning menggunakan ANN, karena model tersebut menghasilkan nilai matriks evaluasi terbaik diantara kedua model yang telah dibuat.

Pada setiap **Solution statements** telah memberikan dampak yang baik dalam memaksimalkan pencapaian **Goals**. Berikut dampak yang diberikan:

- Menggunakan beberapa jenis layes seperti Dense, BatchNormalization, Dropout. serta menggukan activation relu dan sigmoid (karena klasifikasi biner). Tidak lupa untuk melakuakn hyperparameter tuning untuk memaksimalkan model. Melakukan hal tersebut dapat meningkatkan kualtias klasifikasi dengan **memimimalkan overfitting** proses training daripada hanya menggunakan satu jenis layer
- Merangkai model machine learning Logistic Regression sesuai dengan referensi artikel yang menemukan model terbaik untuk mengklasifikasikan pada kasus ini. Dengan menggunakan yang telah dilatih sebelumnya dengan dataset yang sama pada referensi artikel, model terbaru (ANN) dapat dibandingkan dengan **model yang credible**.
- Dengan tidak menghapus outlier model dapat **mendeteksi fitur fitur** yang memepengaruhi seseorang mengalami penyakit jantung dengan baik
