# **Laporan Proyek Machine Learning - Ivan Sholana**

## **Project Overview**

![image](https://github.com/user-attachments/assets/0148fe7e-73eb-4ffa-89c3-03ea682bd482)

Seiring dengan pesatnya pertumbuhan industri hiburan digital, khususnya platform streaming film, kebutuhan akan sistem rekomendasi yang efektif menjadi semakin penting. Jumlah film yang tersedia secara daring terus meningkat, sehingga pengguna seringkali mengalami kesulitan dalam menemukan film yang sesuai dengan preferensi mereka. Oleh karena itu, sistem rekomendasi hadir sebagai solusi untuk membantu pengguna menemukan konten yang relevan dan menarik secara lebih efisien. Salah satu pendekatan yang banyak digunakan dalam pengembangan sistem rekomendasi adalah content-based filtering.
Content-based filtering bekerja dengan menganalisis kesamaan fitur konten dari item yang direkomendasikan. Dalam konteks project ini yang menggunakan rekomendasi film, content-based filtering memanfaatkan atribut-atribut seperti
- judul
- genre
- sinopsis
- kata kunci (keywords)
- pemeran (cast)
- sutradara (director)

Untuk menambah kualitas rekomendasi yang diberikan digunakan metrik popularitas seperti rating dan jumlah voting. Dengan menganalisis fitur-fitur tersebut, sistem dapat mengidentifikasi kemiripan antara film yang telah disukai pengguna sebelumnya dengan film lainnya dalam basis data, sehingga dapat memberikan rekomendasi yang lebih personal. Keunggulan sistem rekomendasi berbasis konten adalah kemampuannya dalam menghasilkan rekomendasi yang relevan tanpa memerlukan data preferensi dari banyak pengguna lain. Sistem ini berfokus pada profil preferensi masing-masing pengguna berdasarkan riwayat interaksi atau film favoritnya. Dengan demikian, pendekatan ini sangat sesuai untuk mengatasi permasalahan cold start yang umum terjadi pada pengguna baru dalam sistem rekomendasi berbasis kolaboratif.

## **RISET TERKAIT**
Salah satu pendekatan yang banyak digunakan dalam pengembangan sistem rekomendasi adalah content-based filtering. 
Pendekatan ini bekerja dengan menganalisis kesamaan fitur konten dari item yang direkomendasikan (Adomavicius & Tuzhilin, 2005). 
Dalam konteks rekomendasi film, content-based filtering memanfaatkan atribut-atribut seperti judul, genre, sinopsis, kata kunci (keywords), pemeran (cast), dan sutradara (director). 
Penelitian oleh Pratama dan Pratomo (2023) menunjukkan bahwa penggunaan sinopsis film dengan algoritma TF-IDF dan cosine similarity dapat menghasilkan rekomendasi yang relevan berdasarkan kesamaan tekstual. 
Untuk meningkatkan kualitas rekomendasi, metrik popularitas seperti rating dan jumlah voting sering diintegrasikan, seperti yang ditunjukkan dalam sistem hibrida yang menggabungkan fitur konten dan popularitas untuk meningkatkan akurasi (Siregar et al., 2023).
Dengan menganalisis fitur-fitur tersebut, sistem dapat mengidentifikasi kemiripan antara film yang telah disukai pengguna sebelumnya dengan film lain dalam basis data, sehingga memberikan rekomendasi yang lebih personal. Keunggulan utama sistem rekomendasi berbasis konten adalah kemampuannya menghasilkan rekomendasi yang relevan tanpa memerlukan data preferensi dari banyak pengguna lain (Lops et al., 2011). 
Sistem ini berfokus pada profil preferensi masing-masing pengguna berdasarkan riwayat interaksi atau film favoritnya. Pendekatan ini sangat efektif untuk mengatasi permasalahan cold start yang umum terjadi pada pengguna baru dalam sistem rekomendasi berbasis kolaboratif, karena hanya memerlukan metadata film yang tersedia sejak awal (Schein et al., 2002). Sebagai contoh, penelitian oleh Rakesh (2023) menunjukkan bahwa sistem berbasis konten meningkatkan presisi dan recall sebesar 20% dan 25% untuk film baru atau niche yang memiliki sedikit data interaksi.

- Adomavicius, G., & Tuzhilin, A. (2005). Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions. IEEE Transactions on Knowledge and Data Engineering, 17(6), 734-749. https://doi.org/10.1109/TKDE.2005.99
- Pratama, R. R., & Pratomo, A. W. (2023). Movie recommendation system using content-based filtering with TF-IDF and cosine similarity. International Journal on Information and Communication Technology, 9(1), 1-10. https://socjs.telkomuniversity.ac.id/ojs/index.php/ijoict/article/view/747
- Siregar, A. P., Nababan, A. A., & Gunawan, D. (2023). Hybrid movie recommender system based on content-based filtering and weighted average method. International Journal of Intelligent Systems and Applications, 15(4), 1-12. https://www.ijisae.org/index.php/IJISAE/article/view/3102
- Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based recommender systems: State of the art and trends. In Recommender Systems Handbook (pp. 73-105). Springer. https://doi.org/10.1007/978-0-387-85820-3_3
- Schein, A. I., Popescul, A., Ungar, L. H., & Pennock, D. M. (2002). Methods and metrics for cold-start recommendations. In Proceedings of the 25th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 253-260). https://doi.org/10.1145/564376.564421
- Rakesh, S. (2023). Movie recommendation system using content-based filtering technique. Baghdad Journal for Engineering and Physical Sciences, 4(1), 1-10. https://bjeps.alkafeel.edu.iq/journal/vol4/iss1/7/

## **PENTINGNYA PROYEK**
Proyek ini penting untuk dikembangkan karena beberapa alasan di bawah ini:
1. **Meningkatkan Pengalaman Pengguna dalam Konsumsi Konten** :
  - Melalui sistem rekomendasi film, dapat membantu pengguna menemukan film yang sesuai dengan minat mereka tanpa harus mencari secara manual di antara ribuan judul.
  - Melalui pemanfaatan fitur seperti sinopsis, genre, pemeran, dan sutradara, sistem rekomendasi ini dapat memberikan rekomendasi yang lebih relevan sehingga meningkatkan kepuasan pengguna.
2. **Relevansi dalam Industri Hiburan Digital** :  
  - Saat ini, platform seperti Netflix, Amazon Prime, dan lainnya sangat bergantung pada sistem rekomendasi untuk mempertahankan pengguna. Proyek ini relevan karena menawarkan pendekatan berbasis konten yang dapat diintegrasikan ke dalam platform tersebut, terutama untuk pengguna baru yang belum memiliki riwayat tontonan yang mana pada kondisi ini **pendekatan berbasis kolaborasi kurang efektif.**

# **BUSINESS UNDERSTANDING**

## **Problem Statements**
Untuk mencapai pengembangan sistem rekomendasi film berbasis *content-based filtering* yang efektif dan berkualitas, proyek ini dirancang untuk menjawab beberapa permasalahan utama sebagai berikut:

1. Bagaimana merancang sistem rekomendasi yang dapat menghasilkan rekomendasi film yang relevan dan dipersonalisasi berdasarkan fitur-fitur konten seperti judul, sinopsis, genre, kata kunci, pemeran, dan sutradara?
2. Bagaimana menangani permasalahan kualitas data, seperti nilai yang hilang (misalnya genre kosong) dan inkonsistensi format, agar data yang digunakan dalam sistem rekomendasi tetap valid dan dapat diandalkan?
3. Bagaimana mengintegrasikan metrik popularitas seperti *weighted rating* dan *normalized popularity* ke dalam sistem rekomendasi untuk memastikan film yang direkomendasikan tidak hanya relevan tetapi juga berkualitas dan diminati banyak pengguna?

## **Goals**
Untuk mencapai tujuan di atas, proyek ini mengusulkan pendekatan content-based filtering dengan langkah-langkah berikut:

1. **Merancang Sistem Rekomendasi:**
    - Menggunakan dataset credits, keywords, dan movies_metadata untuk mengekstrak fitur seperti pemeran, sutradara, genre, sinopsis, dan kata kunci.
    - Mengatasi nilai genre yang hilang dengan klasifikasi NLP seperti pada goals nomor dua.
    - Mengintegrasikan dataset menggunakan merge berbasis kolom id untuk menciptakan dataset gabungan yang konsisten.
    - Menggunakan cosine similarity untuk menghitung kesamaan antara vektor TF-IDF (judul) dan BERT embeddings (fitur gabungan) dari film input dengan semua film dalam dataset.
    - Menggabungkan bobot fitur popularitas
    - Mengambil 5 film dengan skor terbaik.

2. **Penyelesaian Dataset Null:**
    - Menerapkan TF-IDF Vectorizer untuk judul film guna merepresentasikan feature kata-kata fitur sinopsis ke dalam tipe data angka.
    - Melakukan klasifikasi menggunakan multiclass classification
    - Melakukan fine-tunning model Pre-Trained Transofrmer
    - Melakukan perbandingan hasil antara metode TF-IDF dan hasil fine-tunning.
    - Mengisi nan value genre dengan metode terbaik.

3. **Integrasi Metriks Popularitas** :
    - Menghitung weighted rating menggunakan formula yang mempertimbangkan jumlah voting (vote_count) dan rata-rata rating (vote_average), serta nilai rata-rata global (C) dan ambang batas voting (m).
    - Menormalkan popularity untuk memastikan skala yang konsisten dalam perhitungan skor akhir.

## **Solution Statements**
  1. Menggunakan cosine similarity untuk menghitung kesamaan antara vektor TF-IDF (judul) dan BERT embeddings (fitur gabungan) dari film input dengan semua film dalam dataset.
  2. Mengembangkan fungsi rekomendasi yang menerima judul film sebagai input dan mengembalikan top-N film (misalnya, 5 film) dengan skor tertinggi berdasarkan kombinasi kesamaan konten, weighted rating, dan popularity.
  3. Menggabungkan skor kesamaan (TF-IDF dan BERT-based) dengan weighted rating dan normalized popularity menggunakan bobot tertentu (misalnya, 0.2 untuk masing-masing metrik) untuk menghasilkan skor akhir yang seimbang.


# **DATA UNDERSTANDING**
Data Source: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

![image](https://github.com/user-attachments/assets/5ebe2420-4534-4944-9ea4-862423877434)

| **Nama Fitur** | **Tipe Data** | **Deskripsi Singkat**                                                                 |
| -------------- | ------------- | ------------------------------------------------------------------------------------- |
| `cast`         | object        | Daftar pemeran (aktor/aktris) dalam film, biasanya disimpan dalam format JSON-like.   |
| `crew`         | object        | Daftar kru film, termasuk sutradara, penulis naskah, produser, dll. Format JSON-like. |
| `id`           | int / object  | ID unik film untuk menghubungkan data ini dengan dataset utama (`Movies Metadata`).   |

**DATA PREVIEW**

![image](https://github.com/user-attachments/assets/36a730e0-c365-44dc-af73-08ca67aa7c9b)



