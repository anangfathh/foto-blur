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

Buka `http://localhost:3000`, aktifkan kamera, lalu tunjukkan gesture V. Feed
live akan blur selama gesture terlihat dan kembali jernih saat tangan diturunkan.
Situs juga menyediakan blur manual dan tombol ganti kamera, tanpa merekam atau
menyimpan video maupun foto.

## Pemeriksaan

```bash
npm run build
npm test
```

Proyek ini tidak memakai akun pengguna, autentikasi, atau database.
