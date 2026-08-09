# VBlur

Kamera web publik yang otomatis memburamkan feed saat mendeteksi gesture dua
jari berbentuk V. Deteksi tangan berjalan langsung di browser dengan MediaPipe;
video tidak dikirim ke server.

## Menjalankan lokal

Persyaratan: Node.js 22.13 atau lebih baru.

```bash
npm install
npm run dev
```

Buka `http://localhost:3000`, izinkan akses kamera, lalu tunjukkan gesture V.
Feed live akan blur selama gesture terlihat dan kembali jernih saat tangan
diturunkan. Tombol bulat di tengah bawah dapat digunakan untuk mematikan atau
menyalakan kamera. Saat tangan menutup hidung, tayangan akan membeku sementara,
memburamkan latar, menampilkan animasi kucing di keempat sudut, dan memainkan
suara. Satu animasi kucing berukuran lebih besar juga muncul di tengah. Kamera
kembali normal setelah audio selesai. Tampilan tidak merekam atau menyimpan
video maupun foto.

## Menjalankan dengan Docker

```bash
docker compose up --build
```

Buka `http://localhost:3000`. Hentikan dengan `docker compose down`.

## Pemeriksaan

```bash
npm run build
npm test
```

Proyek ini tidak memakai akun pengguna, autentikasi, atau database.
