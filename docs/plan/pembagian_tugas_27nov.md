# Pembagian tugas coding: produk siap 27 Nov 2026

## Konteks
Rencana lama di Google Doc (tab *Coding plan to Dec 6*) targetnya: app jalan end-to-end tanggal 29 Nov, dry run 3 Des, dan semua siap 6 Des. Sekarang targetnya dimajukan: **produk harus siap 27 Nov**, jadi waktunya tinggal 20 hari (7–27 Nov). Fokusnya coding saja. Presentasi, laporan, video, dan persiapan hari testing nanti saja.

**Hari testing: Minggu 29 Nov** di 新竹市東區親子館, dua hari setelah code freeze (diputuskan 8 Nov). Tab lama *Coding plan to Dec 6* dan tab *MILESTONES* sudah dihapus dari Doc.

**Definisi "siap" 27 Nov:** di laptop live (laptop Freya, NVIDIA), app bisa melakukan semua ini:
- satu jendela GUI: cek kesiapan, deteksi mulai/selesai otomatis, tebak task dan kaki, operator konfirmasi atau koreksi, lalu skor dan alasannya masuk `log.csv`, untuk 3 task;
- testing mode tanpa skor dan tanpa timeline di layar;
- video tersimpan kalau V aktif;
- `rescore.py` memberi skor yang sama dengan skor live;
- dry run 3 orang × 3 task × 3 percobaan lolos (27 JSON + 27 baris log).

**Di luar target 27 Nov:** TCN (tidak dikerjakan), demo mode (Easy, hanya kalau ada waktu sisa), laporan, dan slide.

## Alur produk (diputuskan 8 Nov)
- **Satu jendela GUI** di laptop live (`python -m app.main`). Tanpa installer, web app, atau app HP.
- **Sebelum anak pertama:** cek kesiapan (badan masuk frame, fps ≥ 15, cahaya cukup, kaki di area start), dengan pesan perbaikan.
- **Setiap percobaan:** anak diam ±1 s di garis start (siap) → bergerak (mulai) → diam ±1,5 s atau keluar area (selesai) → app menebak task dan kaki → operator tekan Enter atau mengoreksi (1/2/3, L/R, Backspace = batal) → skor 0/1/2. Mode manual Start/Stop tetap ada; dry run menentukan mode untuk hari testing.
- **Timeline momen:** setiap percobaan punya catatan kejadian yang diukur dari titik, misalnya kaki turun di detik 2,0 dan goyang di detik 4,1. Nilai ahli di dataset hanya satu per percobaan, jadi timeline diukur, bukan dipelajari. Skor resmi tetap 0/1/2, skala yang sama dengan KEE. Timeline disimpan di JSON dan disebut di alasan; isinya hanya kejadian, tidak pernah menyebut anak gagal.
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
  - `features.clues.timeline(task, points, t) -> momen`
  - `features.clues.compute(task, points, t) -> dict`
  - `model.predict(task, clues) -> 0/1/2`
  - `app.session.save_try(...)`
- **Checkpoint tiap Rabu, Minggu, dan Kamis** (cek 15 menit). Kalau ada yang belum selesai di checkpoint, harus ada nama orang yang membantu dan tanggal barunya.
- **Level:** 🔴 Hard = di jalur kritis atau sulit · 🟡 Medium · 🟢 Easy = kecil, atau boleh mundur 1–2 hari.
- **Laptop:** Daryn dan Louise pakai Mac (tanpa CUDA), dan pekerjaan mereka (training model, clues) tidak butuh GPU. Yang punya NVIDIA: Marty, Revallo, Freya, Theresia.
  - **Laptop live: laptop Freya.** Revallo melakukan cek 10 menit di laptop itu pada 10 Nov.
  - Revallo mengembangkan modul live di laptop NVIDIA.
  - Daryn integrasi di Mac pakai file rekaman, lalu tes di laptop live.
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

### Daryn: repo, format, training model, integrasi
| Due | Tugas | Level | Selesai kalau |
|---|---|---|---|
| 8 Nov | Repo GitHub (private), folder layout, requirements.txt dengan versi pasti, README cara install, plus scripts/summarize_group_B.py untuk Milestone 1 | 🟢 | Satu teman clone, install, dan dapat baris pertama Milestone 1 tanpa tanya |
| 8 Nov | common/format.py: dataclass Try, save/load, validate, generator percobaan palsu, plus examples/fake_try.npz | 🟢 | validate(load(fake)) lolos |
| 8 Nov | AGENTS.md dan CLAUDE.md di root repo, plus template kartu tugas | 🟢 | Teman menempel satu kartu ke AI-nya dan AI bisa menjelaskan aturan repo dengan benar |
| 9 Nov | Stub untuk semua fungsi dan app/main.py --fake dalam mode otomatis dan manual; contract test, smoke test, CI GitHub Actions, CODEOWNERS | 🟡 | python -m app.main --fake menulis 1 JSON dan 1 baris log; test lulus dengan stub; CI hijau di main |
| 9 Nov | Timeline: stub dan contract test features.clues.timeline; events tersimpan di JSON tiap percobaan | 🟢 | Test hijau; JSON dari --fake punya events |
| 10 Nov | rescore.py versi awal: folder JSON → clues → model → CSV skor | 🟡 | Jalan di JSON dari app/main.py --fake |
| 15 Nov | Ganti stub dengan modul asli setiap kali satu modul jadi; smoke test tetap hijau | 🟡 | Test hijau setiap kali ada merge |
| 16 Nov | model/train_rf.py disiapkan dengan clues task 1 dulu | 🟡 | Jalan tanpa error, keluar F1 dan kappa |
| 19 Nov | Gate: random forest per task dengan tabel clues lengkap mengalahkan baseline. Panjang video (duration_s) tidak masuk model. Untuk task 2 uji juga forest per detik dan pakai yang lebih tinggi. Simpan model, buat model.predict() | 🟡 | Macro-F1 dan kappa di atas baseline di 3 task; file .joblib ada; predict mengembalikan skor dan keyakinan |
| 22 Nov | Menyambungkan semuanya di app/main.py di laptop live: live → clues → model → log (dibantu Revallo dan Freya) | 🔴 | Skor muncul di log dalam beberapa detik setelah satu percobaan, untuk 3 task |
| 25 Nov | Membandingkan rescore dengan skor live di 27 percobaan dry run | 🟡 | 27/27 sama |
| 27 Nov | Code freeze, tag v1.0, install ulang dari nol di laptop live | 🟢 | Install bersih, lalu dry run mini lolos |

### Marty: dataset, baseline, classifier task
| Due | Tugas | Level | Selesai kalau |
|---|---|---|---|
| 8 Nov | Milestone 1 | 🟢 | Baris pertama output: files: 4661 \| children: 106 \| attempts: 1555 |
| 11 Nov | data/load_b.py: B_skel.zip → format bersama; buang klip di bawah 30 frame; titik tengah di neck, skala pakai panjang torso; jangan dipotong di 300 frame | 🟡 | 3,359 file task 1–3 (1,205 / 1,056 / 1,098) |
| 13 Nov | model/baseline.py per task, GroupKFold per anak | 🟢 | Angka mendekati Card 1 (macro-F1 0.19 / 0.20 / 0.26, kappa 0.00) dan cek kebocoran anak PASS |
| 15 Nov | common/pick.py: pick_child() untuk file dengan 2+ orang (28% file B): kunci orang di awal, lalu ikuti neck yang paling dekat. Fungsi yang sama dipakai Revallo untuk live | 🔴 | 5 file multi-orang dicek visual pakai replay Freya |
| 22 Nov | model/task_classifier.py: tebak task (1/2/3) dan kaki dari tabel clues, GroupKFold per anak | 🟡 | Akurasi dan confusion matrix dilaporkan; kalau akurasinya rendah, app pakai tombol saja |
| 22 Nov | Confusion matrix per task dan per fold dari random forest Daryn, diserahkan ke Louise; cek waktu predict di bawah 1 detik | 🟢 | Ada file CSV dan PNG |

### Louise: timeline momen, ukuran gerakan, evaluasi
| Due | Tugas | Level | Selesai kalau |
|---|---|---|---|
| 8 Nov | Milestone 1 | 🟢 | Baris pertama sama |
| 14 Nov | Timeline dan clues task 1 di features/clues.py: momen (dua kaki, stabil, goyang), waktu terlama di satu kaki (detik), sway (panjang torso), kaki turun, waktu goyang. Timeline dibuat dulu, clues dihitung dari timeline | 🟡 | 3 klip cocok dengan hitungan mata, termasuk detik kaki turun |
| 17 Nov | Timeline dan clues task 2: momen (hop, jeda, dua kaki), jumlah hop, jeda terlama | 🔴 | 3 klip cocok dengan hitungan mata |
| 18 Nov | Tabel clues untuk 3,359 file, satu baris per file (kolom task 3 dari Revallo), dikirim ke Daryn dan Marty | 🟡 | Tidak ada nilai kosong, atau setiap pengisian tercatat |
| 19 Nov | features/clues.py: explain(task, clues, events) → teks alasan singkat untuk log, mis. 4,2 s satu kaki, kaki turun di detik 2,0 | 🟢 | Teks untuk 3 klip contoh cocok dengan lembar hitungan mata |
| 22 Nov | Cek clues dan timeline di klip tim kita sendiri (27 rekaman Theresia lewat mode --video Revallo); tuning batas | 🟡 | Sebagian besar klip cocok dengan hitungan mata; klip yang beda didaftar |
| 25 Nov | eval/report.py: confusion matrix per task, angka per fold, salah tebak dibandingkan dengan aturan skor | 🟡 | Ada 1 laporan untuk 3 task |

### Revallo: modul pose live (jalur kritis), clues lompat jauh
| Due | Tugas | Level | Selesai kalau |
|---|---|---|---|
| 8 Nov | Milestone 1 | 🟢 | Baris pertama sama |
| 10 Nov | Cek 10 menit di laptop Freya (laptop live): rtmlib BodyWithFeet (balanced) dengan CUDA dan ada orang di gambar (live/bench_fps.py) | 🟢 | CUDA terpakai dan fps minimal 15 |
| 11 Nov | live/convert.py: cek HALPE_TO_BODY25 terhadap halpe26.py di mmpose; plot 1 frame berlabel untuk cek kiri dan kanan | 🟢 | Plot benar (pergelangan tangan kanan = titik 4) |
| 13 Nov | live/capture.py dan pose.py: thread capture 720p30, auto-framing mati, gambar tidak di-mirror, timestamp per frame, resample ke 30 fps, output Try; plus mode --video untuk klip rekaman (dimajukan dari 15 Nov) | 🔴 | Klip rekaman → file Try yang valid |
| 15 Nov | Video saving di thread sendiri, hanya kalau V aktif, nama file pakai kode anak | 🟡 | fps tidak turun dibandingkan V mati |
| 15 Nov | Jalankan 27 klip tim lewat mode --video dan serahkan titiknya ke Louise dan Theresia | 🟢 | 27 file Try yang valid |
| 17 Nov | Timeline dan clues task 3, di fungsi sendiri di features/clues.py: momen (sebelum, di udara, sesudah), kaki rapat, waktu di udara, ukuran lompatan. Task 3 paling lemah di uji pertama, jadi prioritas setelah capture | 🟡 | 3 klip cocok dengan hitungan mata |
| 19 Nov | Pilih anak saat live: pakai pick_child() dari Marty, kunci orang di garis start, tahan kalau ada orang lain masuk gambar | 🔴 | Tes dengan 2 orang di gambar: track tidak pindah orang |
| 22 Nov | Performa di laptop live (bersama Freya): fps dengan video saving aktif, tes ruangan redup kalau sempat; bantu Daryn menyambungkan semuanya | 🟡 | Minimal 15 fps saat semua aktif (idealnya 30) |

### Freya: replay, jendela app, testing mode
| Due | Tugas | Level | Selesai kalau |
|---|---|---|---|
| 8 Nov | Milestone 1 | 🟢 | Baris pertama sama |
| 10 Nov | viz/replay.py: file Try → mp4 stick figure dengan label nomor titik (dipakai Louise, Marty, dan Theresia untuk cek visual) | 🟢 | Bisa memutar 1 file dataset dan 1 file palsu |
| 11 Nov | Kerangka GUI (app/ui.py, session.py) dengan titik palsu: tombol 1/2/3, L/R, Space, Enter, Backspace, N, V; state machine siap / merekam / konfirmasi | 🟡 | Semua tombol bekerja, state benar |
| 13 Nov | Satu JSON per percobaan (K03_task2_try4_L.json: timestamp, 26 titik mentah, 25 titik hasil konversi, events) dan log.csv (anak, umur, task, percobaan, kaki, tebakan task, dikoreksi, skor, keyakinan, alasan) | 🟢 | 5 percobaan palsu → 5 JSON dan 5 baris log |
| 13 Nov | Replay: garis warna di bawah stick figure yang menampilkan timeline, satu warna per state, untuk cek momen dengan mata | 🟢 | Percobaan dengan events buatan menampilkan warna yang benar di detik yang benar |
| 19 Nov | Testing mode: tidak ada skor dan timeline di layar; input kode anak dan umur dalam bulan; tampilkan cek kesiapan dari Theresia sebelum anak pertama; semua field log lengkap | 🟢 | Layar tidak pernah menampilkan skor atau timeline; masalah kesiapan ditulis dengan kata-kata biasa |
| 21 Nov | Mode otomatis: tampilkan siap / merekam / tebakan task dan kaki; Enter = konfirmasi, 1/2/3 dan L/R = koreksi, Backspace = batal; Start/Stop manual tetap ada | 🟡 | Kedua mode menulis JSON dan baris log yang sama |
| 22 Nov | Tahan crash: setiap percobaan langsung tersimpan, tombol tidak dobel; bantu Daryn menyambungkan semuanya | 🟡 | Kill app di tengah dan percobaan sebelumnya aman |
| 25 Nov | Pimpin dry run: 3 orang × 3 task × 3 percobaan, mode otomatis dengan operator mengoreksi | 🟡 | 27 JSON dan 27 baris log |
| (opsional) | Demo mode: skor, stick figure, dan timeline di layar | 🟢 | Hanya setelah CP6 |

### Theresia: klip tes, cek kesiapan, mulai/selesai otomatis, privasi, tabel hari testing
| Due | Tugas | Level | Selesai kalau |
|---|---|---|---|
| 8 Nov | Milestone 1 | 🟢 | Baris pertama sama |
| 11 Nov | Lembar hitungan mata untuk 9 klip dataset (3 per task) pakai replay Freya: detik satu kaki, detik kaki turun, jumlah hop, kaki rapat atau tidak | 🟢 | Ada tabel 9 baris |
| 13 Nov | Rekam klip tim (720p30, auto-framing mati): 3 orang × 3 task × 3 percobaan, sebagian dengan orang lain lewat, ditambah lembar hitungan mata. Ini data tes untuk Revallo dan Louise | 🟡 | 27 video dan 1 tabel hitungan mata |
| 17 Nov | live/quality.py: cek kesiapan dari titik (badan masuk gambar, fps, cahaya dari confidence titik, kaki di area start) | 🟢 | 3 setup jelek dari klip tim masing-masing dapat pesan yang benar |
| 17 Nov | live/start_area.py: operator menandai area start di gambar kamera; cek kesiapan dan mulai otomatis memakainya | 🟢 | Area tersimpan dan dipakai cek kesiapan |
| 19 Nov | live/segment.py: mulai dan selesai otomatis dari titik (diam sekitar 1 s = siap, gerak = mulai, diam sekitar 1,5 s atau keluar = selesai); tuning pakai 27 klip tim | 🟡 | Mulai dan selesai ketemu di sebagian besar dari 27 klip; yang meleset didaftar |
| 19 Nov | Sekitar 20 klip stick figure group B untuk kalibrasi KEE, tersebar di 3 task dan 3 skor, skor ahli disembunyikan | 🟢 | 20 mp4 dan kunci jawaban terpisah |
| 22 Nov | viz/blur.py: blur wajah pakai titik kepala RTMPose (hidung, mata, telinga), lalu cek frame per frame | 🔴 | 3 video tim: tidak ada frame dengan wajah yang kelihatan |
| 25 Nov | eval/dry_run_count.py: hitung berapa dari 27 percobaan dry run yang tertangkap benar oleh mode otomatis. Hasilnya menentukan mode untuk 29 Nov | 🟢 | Satu angka per task |
| 25 Nov | eval/agreement.py: tabel app vs KEE dan KEE vs KEE, dan 20 klip latihan KEE vs kunci jawaban. Dibuat dan diuji dengan data palsu saja | 🟡 | Tabel keluar dari file skor palsu |
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
| Daryn | 1 | 6 | 5 |
| Marty | 1 | 2 | 3 |
| Louise | 1 | 4 | 2 |
| Revallo | 2 | 3 | 4 |
| Freya | 0 | 4 | 6 |
| Theresia | 1 | 3 | 7 |

Revallo memegang jalur kritis. Bebannya diringankan dengan tiga cara: `pick_child()` dikerjakan Marty, klip tes direkam Theresia, dan cek kesiapan serta mulai/selesai otomatis dipegang Theresia (di-tuning pakai klip itu).

## Risiko utama dan rencana cadangan
- **Daryn tidak bisa tes live di Mac:** jadwalkan sesi bareng di laptop live tanggal 15, 19, dan 22 Nov. Mode `--video file` membuat sebagian besar tes bisa dilakukan tanpa GPU.
- **rtmlib tidak bisa jalan di CUDA (CP1, 10 Nov):** coba onnxruntime versi lain dulu. Kalau masih gagal, pakai MediaPipe sesuai CLAUDE.md, dengan mapping Card 3.
- **Hop count task 2 jelek (CP3):** RF tetap jalan dengan clues yang ada. Louise tuning lagi di 22 Nov memakai klip tim.
- **RF tidak mengalahkan baseline pada 19 Nov:** catat hasilnya apa adanya dan lanjutkan integrasi (app tetap bisa jalan). Daryn dan Marty mencoba fitur atau setting lain sampai 22 Nov.
- **Task 3 tetap lemah** (uji kasar pertama: macro-F1 0,40, tebakan jawaban terbanyak 0,27): Revallo mengerjakannya lebih dulu setelah capture. Hasilnya dilaporkan apa adanya; percobaan yang tidak yakin ditulis "perlu dicek guru".
- **Momen di timeline tidak cocok dengan mata** (titik RTMPose live beda dari OpenPose di dataset): cek di 9 klip hitungan mata dan 27 klip tim, lalu tuning. Kalau satu state tidak bisa diandalkan, state itu tidak dipakai di alasan.
- **Mulai/selesai otomatis meleset di ruangan ramai:** pakai Start/Stop manual di hari testing; angka dari dry run yang menentukan. Operator selalu bisa membatalkan atau mengoreksi.
- **Classifier task sering salah:** operator mengonfirmasi setiap tebakan; kalau akurasinya rendah, kembali ke tombol 1/2/3 dan L/R.
- **fps < 15 dengan video aktif:** kecilkan resolusi simpan video, atau pakai mode `lightweight`. Keduanya harus diukur dulu, bukan ditebak.

## Verifikasi
- Setiap checkpoint dicek dengan kolom "Selesai kalau". Status ada di papan proyek (https://github.com/users/djwofficial/projects/3); PR dengan `Closes #<issue>` menutup tugasnya.
- Final 27 Nov: install bersih di laptop live → dry run 27 percobaan → `rescore.py` cocok 27/27 → tag `v1.0`.
