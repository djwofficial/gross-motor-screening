# Pembagian tugas coding: produk siap 27 Nov 2026

## Konteks
Rencana lama di Google Doc (tab *Coding plan to Dec 6*) targetnya: app jalan end-to-end tanggal 29 Nov, dry run 3 Des, dan semua siap 6 Des. Sekarang targetnya dimajukan: **produk harus siap 27 Nov**, jadi waktunya tinggal 20 hari (7–27 Nov). Fokusnya coding saja. Presentasi, laporan, video, dan persiapan hari testing nanti saja.

**Hari testing: Minggu 29 Nov** di 新竹市東區親子館, dua hari setelah code freeze (diputuskan 8 Nov). Tab lama *Coding plan to Dec 6* dan tab *MILESTONES* sudah dihapus dari Doc.

**Definisi "siap" 27 Nov:** di laptop NVIDIA untuk live, app bisa melakukan semua ini:
- satu jendela GUI: cek kesiapan, deteksi mulai/selesai otomatis, tebak task dan kaki, operator konfirmasi atau koreksi, lalu skor dan alasannya masuk `log.csv`, untuk 3 task;
- testing mode tanpa skor di layar;
- video tersimpan kalau V aktif;
- `rescore.py` memberi skor yang sama dengan skor live;
- dry run 3 orang × 3 task × 3 percobaan lolos (27 JSON + 27 baris log).

**Di luar target 27 Nov:** TCN (tidak dikerjakan), demo mode (Easy, hanya kalau ada waktu sisa), laporan, dan slide.

## Alur produk (diputuskan 8 Nov)
- **Satu jendela GUI** di laptop live (`python -m app.main`). Tanpa installer, web app, atau app HP.
- **Sebelum anak pertama:** cek kesiapan (badan masuk frame, fps ≥ 15, cahaya cukup, kaki di area start), dengan pesan perbaikan.
- **Setiap percobaan:** anak diam ±1 s di garis start (siap) → bergerak (mulai) → diam ±1,5 s atau keluar area (selesai) → app menebak task dan kaki → operator tekan Enter atau mengoreksi (1/2/3, L/R, Backspace = batal) → skor 0/1/2. Mode manual Start/Stop tetap ada; dry run menentukan mode untuk hari testing.
- **Satu anak per percobaan.** Orang lain di luar frame; `pick_child` mengabaikan yang masuk.
- **Tersimpan:** file titik (selalu) + baris `log.csv` (anak, umur, task, percobaan, kaki, tebakan task, dikoreksi atau tidak, skor, keyakinan, alasan). Kalau model tidak yakin: "perlu dicek guru". Video hanya kalau level 3 dan V aktif.
- **Layar:** tidak ada skor selama testing. Apa yang tampil di area besar (gambar kamera, stick figure, atau gambar netral) masih ditanyakan ke KEE.
- **Hasil:** `log.csv` per percobaan; setelah testing, tabel app vs KEE dan KEE vs KEE; offline, random forest vs baseline.
- **22ActionRecognitionTool tidak dipakai** (tanpa bobot model; recognition hanya menjawab task apa, bukan seberapa baik). Classifier task dan model scoring dilatih sendiri dari group B.

## Cara kerja
- **Satu repo dan satu format bersama.** Satu percobaan = array (frames × 25 × 3: x, y, score) dalam urutan BODY_25, ditambah timestamp dalam detik. Semua orang pakai data palsu dulu sampai bagian yang dibutuhkan siap, jadi tidak ada yang menunggu.
- **Hari pertama, Daryn membuat stub untuk setiap fungsi.** Signature-nya sudah final, isinya masih dummy, jadi semua orang bisa langsung import dan tes:
  - `live.pose.PoseStream.read() -> (t, kp26)`
  - `live.convert.halpe_to_body25(kp26) -> (25,3)`
  - `common.pick.pick_child(frames) -> track`
  - `features.clues.compute(task, points, t) -> dict`
  - `model.predict(task, clues) -> 0/1/2`
  - `app.session.save_try(...)`
- **Checkpoint tiap Rabu, Minggu, dan Kamis** (cek 15 menit). Kalau ada yang belum selesai di checkpoint, harus ada nama orang yang membantu dan tanggal barunya.
- **Level:** 🔴 Hard = di jalur kritis atau sulit · 🟡 Medium · 🟢 Easy = kecil, atau boleh mundur 1–2 hari.
- **Laptop:** Daryn dan Louise pakai Mac (tanpa CUDA). Yang punya NVIDIA: Marty, Revallo, Freya, Theresia. Akibatnya:
  - **Laptop live dipilih dari 4 laptop NVIDIA** saat CP1. Usulanku laptop Freya, karena dia yang pegang app dan memimpin dry run, asal fps-nya cukup.
  - Louise tetap pegang modul live. Kodenya tidak bergantung device, jadi dia develop di Mac pakai rtmlib `device='cpu'`. Seberapa cepat di Mac belum diukur, jadi itu masih tebakan. Semua angka fps resmi diukur di laptop live.
  - Daryn juga integrasi di Mac pakai file rekaman, lalu tes live di laptop live.
  - `requirements.txt` pakai platform marker supaya bisa diinstall di Mac maupun Windows/Linux: `onnxruntime-gpu; sys_platform != "darwin"` dan `onnxruntime; sys_platform == "darwin"`.

## Peta checkpoint
| CP | Tanggal | Yang harus sudah ada |
|---|---|---|
| CP0 | Min 8 Nov | Setup: repo, format, dan stub siap; semua bisa install dan jalankan Milestone 1 |
| CP1 | Rab 11 Nov | Setiap bagian punya versi pertama yang jalan sendiri |
| CP2 | Min 15 Nov | Setiap bagian menghasilkan output asli (bukan palsu) |
| CP3 | Kam 19 Nov | **Gate model:** RF mengalahkan baseline (dimajukan dari 22 Nov); semua komponen lengkap |
| CP4 | Min 22 Nov | **App end-to-end jalan untuk kita sendiri** (dimajukan dari 29 Nov) |
| CP5 | Rab 25 Nov | **Dry run** 27 percobaan lolos, rescore cocok (dimajukan dari 3 Des) |
| CP6 | Jum 27 Nov | Bug sudah diperbaiki, code freeze, tag `v1.0` → **produk siap** |

## Tugas per orang

### Daryn: repo, format, integrasi, rescore
| Due | Tugas | Level | Selesai kalau |
|---|---|---|---|
| **8 Nov** | Repo GitHub (private), folder layout, `requirements.txt` dengan versi pasti, README cara install, plus `scripts/summarize_group_B.py` untuk Milestone 1 | 🟢 | Satu teman clone, install, dan dapat baris pertama Milestone 1 tanpa tanya |
| 8 Nov | `common/format.py`: dataclass Try, save/load, validate, generator percobaan palsu, plus `examples/fake_try.npz` | 🟢 | `validate(load(fake))` lolos |
| 8 Nov | `AGENTS.md` dan `CLAUDE.md` di root repo, plus template kartu tugas (lihat bagian "Kerja dengan AI coding") | 🟢 | Teman coba tempel satu kartu ke AI-nya → AI bisa jelaskan aturan repo dengan benar |
| 9 Nov | Stub untuk semua fungsi (termasuk `live/segment.py`, `live/quality.py`, `model/task_classifier.py`) dan `app/main.py --fake` dalam mode otomatis dan manual; contract test, smoke test, CI GitHub Actions, CODEOWNERS | 🟡 | `python -m app.main --fake` menulis 1 JSON dan 1 baris log; test lulus dengan stub; CI hijau di `main` |
| 10 Nov | `rescore.py` versi awal: folder JSON → clues → model → CSV skor (dimajukan dari 19 Nov) | 🟡 | Jalan di JSON dari `app/main.py --fake` |
| 15 Nov | Ganti stub dengan modul asli setiap kali satu modul jadi; smoke test tetap hijau | 🟡 | Test hijau setiap kali ada merge |
| 22 Nov | Menyambungkan semuanya di `app/main.py` di laptop NVIDIA: live → clues → model → log (dibantu Louise dan Freya) | 🔴 | Skor muncul di log ≤ beberapa detik setelah satu percobaan, untuk 3 task |
| 25 Nov | Membandingkan rescore dengan skor live di 27 percobaan dry run | 🟡 | 27/27 sama |
| 27 Nov | Code freeze, tag `v1.0`, install ulang dari nol di laptop live | 🟢 | Install bersih → dry run mini lolos |

### Marty: dataset, baseline, model
| Due | Tugas | Level | Selesai kalau |
|---|---|---|---|
| 8 Nov | Milestone 1 | 🟢 | Baris pertama output: `files: 4661 \| children: 106 \| attempts: 1555` |
| 11 Nov | `data/load_b.py`: B_skel.zip → format bersama; buang klip < 30 frame; titik tengah di neck, skala pakai panjang torso; **jangan dipotong di 300 frame** | 🟡 | 3,359 file task 1–3 (1,205 / 1,056 / 1,098) |
| 13 Nov | `model/baseline.py` per task, GroupKFold per anak | 🟢 | Angka mendekati Card 1 (macro-F1 0.19 / 0.20 / 0.26, kappa 0.00) dan cek kebocoran anak PASS |
| 15 Nov | `common/pick.py: pick_child()` untuk file dengan 2+ orang (28% file B): kunci orang di awal, lalu ikuti neck yang paling dekat. **Fungsi yang sama dipakai Louise untuk live** | 🔴 | 5 file multi-orang dicek visual pakai replay Theresia |
| 16 Nov | `model/train_rf.py` disiapkan dengan clues task 1 dulu | 🟡 | Jalan tanpa error, keluar F1 dan kappa |
| 19 Nov | **Gate:** RF per task dengan tabel clues lengkap mengalahkan baseline; simpan model, buat `model.predict()` | 🟡 | Macro-F1 dan kappa > baseline di 3 task; file `.joblib` ada; `predict` mengembalikan skor dan keyakinan |
| 22 Nov | `model/task_classifier.py`: tebak task (1/2/3) dan kaki dari tabel clues, GroupKFold per anak | 🟡 | Akurasi dan confusion matrix dilaporkan; kalau akurasinya rendah, app pakai tombol saja |
| 22 Nov | Confusion matrix per task dan per fold diserahkan ke Revallo; cek waktu `predict` < 1 detik | 🟢 | Ada file CSV dan PNG |

### Revallo: movement clues dan evaluasi
| Due | Tugas | Level | Selesai kalau |
|---|---|---|---|
| 8 Nov | Milestone 1 | 🟢 | Baris pertama sama |
| 11 Nov | Lembar "hitungan mata" untuk 9 klip dataset (3 per task) pakai replay Theresia: detik satu kaki, jumlah hop, kaki rapat atau tidak | 🟢 | Ada tabel 9 baris |
| 14 Nov | Clues task 1 di `features/clues.py`: waktu terlama di satu kaki (detik), sway (dalam panjang torso), berapa kali kaki turun | 🟡 | 3 klip cocok dengan hitungan mata |
| 17 Nov | Clues task 2 (jumlah hop, jeda terlama) dan task 3 (kaki rapat, waktu di udara, ukuran lompatan) | 🔴 | 6 klip cocok dengan hitungan mata |
| 18 Nov | Tabel clues untuk 3,359 file, satu baris per file, dikirim ke Marty | 🟡 | Tidak ada nilai kosong, atau setiap pengisian tercatat |
| 19 Nov | `features/clues.py: explain(task, clues)` → teks alasan singkat untuk log, mis. "4,2 s satu kaki, 0× turun" | 🟢 | Teks untuk 3 klip contoh cocok dengan lembar hitungan mata |
| 22 Nov | Cek clues di **klip tim kita sendiri** (27 klip rekaman Theresia lewat mode offline Louise); tuning threshold | 🟡 | Sebagian besar klip cocok dengan hitungan mata; klip yang beda didaftar |
| 25 Nov | `eval/report.py`: confusion matrix per task, angka per fold, salah tebak dibandingkan dengan aturan skor | 🟡 | Ada 1 laporan untuk 3 task |

### Louise: live pose module (RTMPose) — jalur kritis
| Due | Tugas | Level | Selesai kalau |
|---|---|---|---|
| 8 Nov | Milestone 1 | 🟢 | Baris pertama sama |
| 10 Nov | rtmlib `BodyWithFeet` (balanced) jalan di webcam Mac-nya sendiri (CPU). Tulis script `live/bench_fps.py`, lalu Marty, Revallo, Freya, dan Theresia menjalankannya dengan CUDA di laptop masing-masing. Pilih laptop live | 🟡 | Ada tabel fps 4 laptop NVIDIA dan laptop live sudah dipilih |
| 11 Nov | `live/convert.py`: `HALPE_TO_BODY25`, dicek terhadap `halpe26.py` di mmpose; plot 1 frame berlabel untuk cek kiri dan kanan | 🟢 | Plot benar (tangan kanan = titik 4) |
| 15 Nov | `live/capture.py` dan `pose.py`: thread capture 720p30, auto-framing mati, gambar tidak di-mirror, timestamp per frame, resample ke 30 fps, output Try; **plus mode `--video file`** untuk klip rekaman | 🔴 | Klip rekaman → file Try yang valid |
| 19 Nov | Pilih anak saat live: pakai `pick_child()` dari Marty, kunci orang di garis start, tahan kalau ada orang lain masuk frame | 🔴 | Tes dengan 2 orang di frame: track tidak pindah orang |
| 22 Nov | Performa **di laptop live** (duduk bareng pemilik laptop): fps dengan video saving aktif, plus tes di ruangan redup; bantu Daryn menyambungkan semuanya | 🟡 | ≥ 15 fps saat semua aktif (idealnya 30) |

### Freya: app dan testing mode
| Due | Tugas | Level | Selesai kalau |
|---|---|---|---|
| 8 Nov | Milestone 1 | 🟢 | Baris pertama sama |
| 11 Nov | Kerangka GUI (`app/ui.py`, `session.py`) dengan titik palsu: tombol 1/2/3, L/R, Space, Enter, Backspace, N, V; state machine siap / merekam / konfirmasi | 🟡 | Semua tombol bekerja, state benar |
| 13 Nov | Satu JSON per percobaan (`K03_task2_try4_L.json`: timestamp, 26 titik mentah, 25 titik hasil konversi) dan `log.csv` (anak, umur, task, percobaan, kaki, tebakan task, dikoreksi, skor, keyakinan, alasan) | 🟢 | 5 percobaan palsu → 5 JSON dan 5 baris log |
| 15 Nov | Video saving di thread sendiri, hanya kalau V aktif, nama file pakai kode anak | 🟡 | fps tidak turun dibandingkan V mati |
| 19 Nov | Testing mode: tidak ada skor di layar; input kode anak dan umur dalam bulan; tampilkan cek kesiapan dari Theresia sebelum anak pertama; semua field log lengkap | 🟢 | Layar tidak pernah menampilkan skor; masalah kesiapan ditulis dengan kata-kata biasa |
| 21 Nov | Mode otomatis: tampilkan siap / merekam / tebakan task dan kaki; Enter = konfirmasi, 1/2/3 dan L/R = koreksi, Backspace = batal; Start/Stop manual tetap ada | 🟡 | Kedua mode menulis JSON dan baris log yang sama |
| 22 Nov | Tahan crash: setiap percobaan langsung tersimpan, tombol tidak dobel; bantu Daryn menyambungkan semuanya | 🟡 | Kill app di tengah → percobaan sebelumnya aman |
| 25 Nov | **Pimpin dry run** 3 orang × 3 task × 3 percobaan, mode otomatis dengan operator mengoreksi | 🟡 | 27 JSON dan 27 baris log; jumlah percobaan yang tertangkap benar oleh mode otomatis |
| (opsional) | Demo mode: skor dan stick figure di layar | 🟢 | Hanya setelah CP6 |

### Theresia: replay, klip tes, cek kesiapan, mulai/selesai otomatis, privasi
| Due | Tugas | Level | Selesai kalau |
|---|---|---|---|
| 8 Nov | Milestone 1 | 🟢 | Baris pertama sama |
| 10 Nov | `viz/replay.py`: file Try → mp4 stick figure dengan label nomor titik (dipakai Revallo, Marty, dan Louise untuk cek visual) | 🟢 | Bisa memutar 1 file dataset dan 1 file palsu |
| 13 Nov | **Rekam klip tim** (720p30, auto-framing mati): 3 orang × 3 task × 3 percobaan, ditambah lembar hitungan mata. Ini data tes untuk Louise dan Revallo | 🟡 | 27 video dan 1 tabel hitungan mata |
| 17 Nov | `live/quality.py`: cek kesiapan dari titik (badan masuk frame, fps, cahaya dari confidence titik, kaki di area start) | 🟢 | 3 setup jelek dari klip tim masing-masing dapat pesan yang benar |
| 19 Nov | `live/segment.py`: mulai dan selesai otomatis dari titik (diam ±1 s = siap, gerak = mulai, diam ±1,5 s atau keluar = selesai), tuning pakai 27 klip tim | 🟡 | Mulai dan selesai ketemu di sebagian besar dari 27 klip; yang meleset didaftar |
| 19 Nov | Sekitar 20 klip stick figure group B untuk kalibrasi KEE, skor ahli disembunyikan | 🟢 | 20 mp4 dan kunci jawaban terpisah |
| 22 Nov | `viz/blur.py`: blur wajah pakai titik kepala RTMPose (hidung, mata, telinga), lalu cek frame per frame | 🔴 | 3 video tim: tidak ada frame dengan wajah yang kelihatan |
| 25 Nov | Replay dari JSON dry run (untuk cek bug visual) | 🟢 | 27 replay terbuat |

## Kerja dengan AI coding (Claude, ChatGPT, Codex, dll.)
Risikonya kalau 6 orang masing-masing melempar tugas ke AI tanpa aturan bersama: tiap AI menebak format sendiri, mengubah file orang lain, atau mengarang API. Ujungnya bagian-bagian tidak nyambung pas integrasi tanggal 22 Nov. Supaya itu tidak terjadi:

1. **Satu file konteks untuk semua AI di root repo.**
   - `AGENTS.md` dibaca Codex dan banyak tool lain. `CLAUDE.md` cukup berisi satu baris `@AGENTS.md`, jadi isinya cuma satu sumber.
   - Isinya:
     - format bersama;
     - signature fungsi;
     - tabel folder → pemilik;
     - versi library;
     - aturan proyek: detik dan panjang torso, bukan pixel; group B saja; bukan diagnosis; deteksi pose di gambar yang tidak di-mirror.
   - Untuk ChatGPT versi web: tempel isi `AGENTS.md` di awal chat, atau taruh di Project instructions.
2. **Kartu tugas = prompt.**
   - Setiap baris tugas di atas ditulis jadi satu kartu: tujuan, input → output, signature, file yang **boleh** diubah, kriteria "Selesai kalau", dan hal yang perlu diwaspadai.
   - Kartu itu langsung ditempel ke AI, jadi orang tidak menulis prompt sendiri dari nol.
3. **Contract test adalah pagarnya.**
   - Tugas AI: membuat `tests/contracts/test_<bagian>.py` jadi hijau tanpa mengubah test-nya.
   - Kalau test atau signature memang perlu diubah, minta ke Daryn. Jangan biarkan AI mengubahnya diam-diam.
4. **Batas folder.**
   - Setiap orang dan AI-nya hanya mengubah folder miliknya.
   - Folder `common/` dan `tests/contracts/` hanya diubah lewat Daryn.
   - Di GitHub, pasang CODEOWNERS dan branch protection: perubahan di folder orang lain butuh review dari pemiliknya.
5. **PR kecil, satu tugas = satu branch = satu PR.**
   - Di deskripsi PR, tempel bukti "Selesai kalau" (angka, plot, atau screenshot).
   - Merge harus setelah smoke test hijau.
6. **Jangan percaya angka atau API karangan AI.**
   - Setiap angka harus keluar dari script yang benar-benar dijalankan (aturan no. 2 di CLAUDE.md).
   - API rtmlib dan indeks Halpe dicek ke source aslinya (repo rtmlib, `halpe26.py` di mmpose), bukan dari jawaban AI.
   - Beri tahu AI versi library yang ada di `requirements.txt`.
7. **Pemilik harus bisa menjelaskan kodenya.**
   - Di setiap checkpoint, pemilik menjelaskan 2–3 menit cara kerja bagiannya. Kalau tidak bisa, bagian itu belum dihitung Done.
   - Ini juga persiapan untuk tanya jawab saat presentasi.
8. **Data anak tidak pernah masuk ke AI.**
   - Titik, video, log, kode anak, dan umur dari hari testing tidak boleh ditempel atau di-upload ke chat AI mana pun.
   - Alasannya: consent form bilang data disimpan di drive terenkripsi dan dihapus paling lambat 25 Des.
   - Untuk debugging, pakai data group B (CC0), data palsu, atau klip tim kita sendiri.
9. **Formatter yang sama:** `ruff format` dijalankan sebelum commit, supaya diff dari AI tidak penuh perubahan spasi.

## Ringkasan beban (Hard / Medium / Easy)
| Orang | 🔴 | 🟡 | 🟢 |
|---|---|---|---|
| Daryn | 1 | 4 | 4 |
| Marty | 1 | 4 | 3 |
| Revallo | 1 | 4 | 3 |
| Louise | 2 | 2 | 2 |
| Freya | 0 | 5 | 4 |
| Theresia | 1 | 2 | 5 |

Louise memegang jalur kritis. Bebannya diringankan dengan tiga cara: `pick_child()` dikerjakan Marty, klip tes direkam Theresia, dan cek kesiapan serta mulai/selesai otomatis dipegang Theresia (di-tuning pakai klip itu).

## Risiko utama dan rencana cadangan
- **Louise dan Daryn tidak bisa tes live sendiri (Mac):** jadwalkan sesi bareng di laptop live tanggal 15, 19, dan 22 Nov. Mode `--video file` membuat sebagian besar tes bisa dilakukan tanpa GPU.
- **rtmlib tidak bisa jalan di CUDA (CP1, 10 Nov):** coba onnxruntime versi lain dulu. Kalau masih gagal, pakai MediaPipe sesuai CLAUDE.md, dengan mapping Card 3.
- **Hop count task 2 jelek (CP3):** RF tetap jalan dengan clues yang ada. Revallo tuning lagi di 22 Nov memakai klip tim.
- **RF tidak mengalahkan baseline pada 19 Nov:** catat hasilnya apa adanya dan lanjutkan integrasi (app tetap bisa jalan). Marty mencoba fitur atau setting lain sampai 22 Nov.
- **Mulai/selesai otomatis meleset di ruangan ramai:** pakai Start/Stop manual di hari testing; angka dari dry run yang menentukan. Operator selalu bisa membatalkan atau mengoreksi.
- **Classifier task sering salah:** operator mengonfirmasi setiap tebakan; kalau akurasinya rendah, kembali ke tombol 1/2/3 dan L/R.
- **fps < 15 dengan video aktif:** kecilkan resolusi simpan video, atau pakai mode `lightweight`. Keduanya harus diukur dulu, bukan ditebak.

## Verifikasi
- Setiap checkpoint dicek dengan kolom "Selesai kalau". Pemilik tugas menulis "Done + tanggal".
- Final 27 Nov: install bersih di laptop live → dry run 27 percobaan → `rescore.py` cocok 27/27 → tag `v1.0`.
