"""
Dashboard Streamlit: Guitar Tuner & Cek Chord Gitar berbasis DTW
Jalankan dari folder project (yang berisi nsynth-valid/ dan nsynth-test/ atau arsip .tar.gz-nya):
    streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import tuner_core as tc

st.set_page_config(page_title="DTW Guitar Tuner & Cek Chord", page_icon="🎸", layout="wide")

FOLDER_DATA = "."
FILE_HASIL = "hasil_analisis.pkl"
BIRU, JINGGA, HIJAU, MERAH, ABU, UNGU = "#004998", "#F5AA2D", "#1E8E3E", "#C62828", "#6B7280", "#7B1FA2"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


# =============================================================================== data hasil analisis
@st.cache_resource(show_spinner=False)
def muat_hasil_cache(path, mtime):
    return tc.muat_hasil(path)


def ambil_hasil():
    if not os.path.exists(FILE_HASIL):
        with st.status("Menjalankan analisis pertama kali (hanya sekali, hasilnya disimpan)...", expanded=True) as s:
            R = tc.jalankan_analisis(FOLDER_DATA, log=s.write)
            tc.simpan_hasil(R, FILE_HASIL)
            s.update(label="Analisis selesai", state="complete")
    return muat_hasil_cache(FILE_HASIL, os.path.getmtime(FILE_HASIL))


R = ambil_hasil()
T1, T2, A = R["tuner"], R["chord"], R["app"]


# =============================================================================== helper
def tampil(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def pct(v):
    return f"{v:.1%}" if pd.notna(v) else "-"


def akurasi_per_variasi(hasil, metode, kolom_benar, variasi):
    tab = pd.DataFrame({v: [(hasil.loc[hasil.variasi == v, m] == hasil.loc[hasil.variasi == v, kolom_benar]).mean()
                            for m in metode] for v in variasi}, index=metode)
    tab["rata-rata"] = [(hasil[m] == hasil[kolom_benar]).mean() for m in metode]
    return tab


def fmt_tabel(df, persen=(), desimal=(), p=()):
    f = {c: "{:.1%}" for c in persen if c in df.columns}
    f.update({c: "{:.3f}" for c in desimal if c in df.columns})
    f.update({c: "{:.4f}" for c in p if c in df.columns})
    return df.style.format(f, na_rep="-")


def fig_bar_variasi(tab, variasi, judul, acak, baseline=None):
    fitur = ["semiton", "chroma", "MFCC"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
    for a, nf in zip(ax, fitur):
        for i, (nj, w) in enumerate(zip(["Euclid padding", "Euclid resampling", "DTW"], [ABU, JINGGA, BIRU])):
            a.bar(np.arange(len(variasi)) + (i - 1) * 0.25, tab.loc[f"{nf} | {nj}", variasi], 0.25, color=w, label=nj)
        if baseline:
            a.plot(range(len(variasi)), tab.loc[baseline, variasi], "D--", color=MERAH, ms=5, label=baseline)
        a.axhline(acak, ls=":", color="gray")
        a.set_xticks(range(len(variasi))); a.set_xticklabels(variasi); a.set_ylim(0, 1.05); a.set_title(f"Fitur: {nf}")
    ax[0].set_ylabel("akurasi")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle(judul, y=1.02); fig.tight_layout()
    return fig


def fig_ci(tab):
    t = tab.sort_values("akurasi")
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(t) + 1))
    warna = [BIRU if "DTW" in m else MERAH if ("YIN" in m or "biner" in m) else JINGGA for m in t.index]
    ax.errorbar(t.akurasi, range(len(t)), xerr=[t.akurasi - t["CI 95% bawah"], t["CI 95% atas"] - t.akurasi],
                fmt="none", ecolor="gray", capsize=3)
    ax.scatter(t.akurasi, range(len(t)), c=warna, zorder=3, s=40)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index); ax.set_xlim(0, 1.02)
    ax.set_xlabel("akurasi (titik) dan selang kepercayaan 95% Wilson (garis)")
    fig.tight_layout()
    return fig


def fig_konfusi(benar, prediksi, kelas, judul):
    K = pd.crosstab(pd.Series(benar, name="b"), pd.Series(prediksi, name="p")).reindex(index=kelas, columns=kelas, fill_value=0)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(K.values, cmap="Blues")
    for i in range(len(kelas)):
        for j in range(len(kelas)):
            if K.values[i, j]:
                ax.text(j, i, K.values[i, j], ha="center", va="center", fontsize=8,
                        color="white" if K.values[i, j] > K.values.max() / 2 else "black")
    ax.set_xticks(range(len(kelas))); ax.set_xticklabels(kelas); ax.set_yticks(range(len(kelas))); ax.set_yticklabels(kelas)
    ax.set_xlabel("prediksi"); ax.set_ylabel("sebenarnya"); ax.set_title(judul, fontsize=9)
    fig.tight_layout()
    return fig


def fig_warping(T, F, jalur, judul, label_y="frame template", label_x="frame uji"):
    C = np.sqrt(((T[:, None] - F[None]) ** 2).sum(-1))
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.imshow(C, origin="lower", aspect="auto", cmap="Blues_r")
    if len(jalur):
        ax.plot(jalur[:, 1], jalur[:, 0], color=JINGGA, lw=2.5)
    ax.set_title(judul, fontsize=9); ax.set_xlabel(label_x); ax.set_ylabel(label_y)
    fig.tight_layout()
    return fig


def fig_fitur(F, judul, sumbu="semiton"):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.imshow(F.T, origin="lower", aspect="auto", cmap="magma")
    if sumbu == "semiton":
        idx = [k for k, m in enumerate(tc.MIDI_BIN) if m % 12 == 0]
        ax.set_yticks(idx); ax.set_yticklabels([tc.nama_nada(tc.MIDI_BIN[k]) for k in idx])
    else:
        ax.set_yticks(range(12)); ax.set_yticklabels(tc.NAMA_NADA, fontsize=7)
    ax.set_xlabel("frame (≈0,13 detik)"); ax.set_title(judul, fontsize=9)
    fig.tight_layout()
    return fig


def fig_gauge(cent):
    c = float(np.clip(cent, -50, 50)) if np.isfinite(cent) else 0.0
    fig, ax = plt.subplots(figsize=(5.2, 3), subplot_kw={"projection": "polar"})
    ax.set_thetamin(0); ax.set_thetamax(180); ax.set_theta_zero_location("W"); ax.set_theta_direction(-1)
    sudut = lambda v: np.deg2rad((v + 50) / 100 * 180)
    for a, b, w in [(-50, -15, MERAH), (-15, -5, JINGGA), (-5, 5, HIJAU), (5, 15, JINGGA), (15, 50, MERAH)]:
        th = np.linspace(sudut(a), sudut(b), 30)
        ax.fill_between(th, 0.82, 1.0, color=w, alpha=0.85)
    ax.plot([sudut(c), sudut(c)], [0, 0.95], color="black", lw=3)
    ax.scatter([0], [0], color="black", s=40, zorder=5)
    ax.set_xticks([sudut(v) for v in [-50, -25, -5, 0, 5, 25, 50]])
    ax.set_xticklabels(["-50", "-25", "-5", "0", "+5", "+25", "+50"], fontsize=8)
    ax.set_yticks([]); ax.set_ylim(0, 1); ax.spines["polar"].set_visible(False); ax.grid(False)
    ax.set_title("selisih (cent)", fontsize=9, pad=2)
    return fig


def fig_jarak(jarak, label, terpilih, judul):
    fig, ax = plt.subplots(figsize=(5, 3))
    warna = [BIRU if l == terpilih else "#C9D6E8" for l in label]
    ax.bar(label, jarak, color=warna)
    ax.set_ylabel("jarak DTW (makin kecil makin mirip)"); ax.set_title(judul, fontsize=9)
    fig.tight_layout()
    return fig


def fig_gelombang_spektrum(x, f_target=None, f0=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    ax[0].plot(np.arange(len(x)) / tc.SR, x, lw=0.5, color=BIRU)
    ax[0].set_xlabel("detik"); ax[0].set_title("Gelombang suara (setelah dipangkas & difilter)", fontsize=9)
    seg = x[: int(1.5 * tc.SR)]
    f = np.fft.rfftfreq(len(seg), 1 / tc.SR); S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    ax[1].semilogy(f, S + 1e-9, color=BIRU, lw=0.8); ax[1].set_xlim(0, 1200); ax[1].set_xlabel("Hz")
    if f_target:
        for k in range(1, 7):
            ax[1].axvline(k * f_target, color=JINGGA, ls=":", lw=1)
    if f0 and np.isfinite(f0):
        ax[1].axvline(f0, color=MERAH, lw=1.2)
    ax[1].set_title("Spektrum 1,5 detik pertama (garis kuning = senar target & harmoniknya, merah = f0 terukur)", fontsize=9)
    fig.tight_layout()
    return fig


def fig_profil(profil, target):
    nada = tc.pc(tc.CHORD[target])
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(12), profil, color=[BIRU if k in nada else "#C9D6E8" for k in range(12)])
    for k in nada:
        ax.axvspan(k - 0.5, k + 0.5, color=JINGGA, alpha=0.12)
    ax.axhline(0.3, ls=":", color=MERAH, lw=1); ax.axhline(0.5, ls="--", color=ABU, lw=1)
    ax.set_xticks(range(12)); ax.set_xticklabels(tc.NAMA_NADA); ax.set_ylim(0, 1.05)
    ax.set_ylabel("energi relatif"); ax.set_title(f"Profil 12 nada (biru/kuning = nada chord {target})", fontsize=9)
    fig.tight_layout()
    return fig


# =============================================================================== sidebar
st.sidebar.title("🎸 DTW Guitar Lab")
halaman = st.sidebar.radio("Halaman", ["1. Deskripsi project", "2. Hasil analisis: tuner",
                                       "3. Hasil analisis: cek chord", "4. Coba tuner (rekam)",
                                       "5. Coba cek chord (rekam)"])
st.sidebar.divider()
st.sidebar.caption(f"Data analisis: **{R['mode']}** · dibuat {R['dibuat']}")
if R["mode"] == "sintetis":
    st.sidebar.warning("Data NSynth tidak ditemukan; hasil memakai data sintetis dan tidak untuk dilaporkan.")
if st.sidebar.button("Jalankan ulang analisis"):
    if os.path.exists(FILE_HASIL):
        os.remove(FILE_HASIL)
    st.cache_resource.clear()
    st.rerun()


# =============================================================================== 1. deskripsi
if halaman.startswith("1"):
    st.title("Guitar Tuner & Cek Chord Gitar dengan Dynamic Time Warping")
    st.caption("Project Data Mining · pengembangan studi kasus DTW Minggu 11 (pengenalan ucapan angka)")
    st.markdown("""
Project ini menerapkan **Dynamic Time Warping (DTW)** untuk dua masalah gitar yang nyata:

- **Tuner**: satu senar dipetik, sistem menebak senar mana dan berapa *cent* lagi nadanya harus dinaikkan atau diturunkan.
- **Cek chord**: satu chord digenjreng, sistem menebak chord apa, menyatakan benar atau salah, dan memberi petunjuk nada yang bermasalah.

Metodenya sama dengan studi kasus di kelas: **audio → deret vektor fitur → jarak (Euclid vs DTW) → 1-NN ke template**.
Yang berbeda adalah fiturnya. MFCC (dipakai di studi kasus ucapan) menangkap warna suara, sedangkan untuk musik yang penting adalah **tinggi nada**.
""")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Bagian 1: Tuner")
        st.markdown("""
- **Kelas**: 6 senar lepas (E2, A2, D3, G3, B3, E4)
- **Fitur utama**: semiton (energi per semitone, membedakan oktaf)
- **Baseline**: YIN (estimasi frekuensi dasar)
- **Variasi uji**: asli, derau 15 dB, pendek 0,8 detik, fals ±35 cent
- **Evaluasi**: akurasi + CI 95%, F1 per senar, McNemar, ketepatan cent
""")
    with c2:
        st.subheader("Bagian 2: Cek chord")
        st.markdown("""
- **Kelas**: 8 chord terbuka (C, D, E, G, A, Am, Em, Dm)
- **Fitur utama**: chroma (12 nama nada, tanpa oktaf)
- **Baseline**: template chord biner
- **Variasi uji**: genjrengan turun, naik, lambat, derau; chord salah (nada salah, senar mati)
- **Evaluasi**: akurasi + CI 95%, McNemar, verifikasi (ROC-AUC, sensitivitas, spesifisitas)
""")
    st.subheader("Alur metode")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EEF3FA", color="#004998", fontname="Helvetica"];
  a [label="Audio\\n(petikan / genjrengan)"]; b [label="Pra-proses\\npangkas sunyi, filter"];
  c [label="Fitur per frame\\nsemiton / chroma / MFCC"]; d [label="Jarak ke template\\nEuclid atau DTW"];
  e [label="1-NN\\ntemplate terdekat"]; f [label="Senar + cent\\n(tuner)"]; g [label="Chord + benar/salah\\n(cek chord)"];
  a -> b -> c -> d -> e; e -> f; e -> g;
}""")
    st.subheader("DTW dalam satu rumus")
    st.latex(r"D(i,j)=c(i,j)+\min\{D(i-1,j-1),\ D(i-1,j),\ D(i,j-1)\},\qquad c(i,j)=\lVert \mathbf{x}_i-\mathbf{y}_j\rVert")
    st.markdown("Jarak DTW dinormalkan dengan $n+m$. Euclid membandingkan frame ke-$i$ dengan frame ke-$i$, sedangkan DTW boleh "
                "memasangkan frame secara lentur sehingga perbedaan tempo atau panjang bunyi tidak dihukum.")
    st.subheader("Data")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Sumber data", "NSynth" if R["mode"] == "nsynth" else "Sintetis")
    k2.metric("Bunyi uji tuner", f"{len(T1['hasil'])}", f"{T1['n_rekaman']} rekaman × 4 variasi", delta_color="off")
    k3.metric("Genjrengan uji chord", f"{len(T2['hasil'])}", f"+ {T2['n_salah']} chord salah", delta_color="off")
    k4.metric("Gitar uji", f"{T1['n_gitar_uji']} (tuner) / {T2['n_gitar_uji']} (chord)")
    st.markdown("""
**NSynth** (Google Magenta, CC BY 4.0): nada tunggal berdurasi 4 detik dari ribuan instrumen. Template diambil dari satu gitar di split
*valid*, data uji dari gitar lain di split *test* (instrumen tidak tumpang tindih). Genjrengan chord dirakit dari nada-nada gitar
NSynth dengan jeda 10–100 ms antarsenar.
""")
    with st.expander("Cara memakai dashboard"):
        st.markdown("""
- **Halaman 2 & 3**: hasil evaluasi dari notebook (akurasi, selang kepercayaan, matriks konfusi, uji statistik).
- **Halaman 4**: tekan tombol mikrofon, petik satu senar ±2 detik, lalu hentikan. Hasil senar dan cent muncul otomatis.
- **Halaman 5**: pilih chord tujuan, tekan mikrofon, genjreng chord, lalu hentikan.
- Bila mikrofon tidak bisa dipakai, unggah file **WAV** sebagai gantinya.
""")


# =============================================================================== 2. hasil tuner
elif halaman.startswith("2"):
    st.title("Hasil analisis Bagian 1: Tuner (6 senar lepas)")
    hasil, acc = T1["hasil"], T1["akurasi"]
    tab_var = akurasi_per_variasi(hasil, T1["metode"], "pitch", tc.VARIASI)
    med_asli = np.nanmedian(np.abs(hasil.loc[hasil.variasi == "asli", "cent"]))
    med_fals = np.nanmedian(np.abs(hasil.loc[hasil.variasi == "fals", "cent"]))
    _, _, p_dtw = tc.mcnemar(hasil.pitch.to_numpy(), hasil["semiton | DTW"].to_numpy(), hasil["semiton | Euclid resampling"].to_numpy())

    k = st.columns(5)
    k[0].metric("semiton | DTW", pct(acc.loc["semiton | DTW", "akurasi"]))
    k[1].metric("semiton | Euclid resampling", pct(acc.loc["semiton | Euclid resampling", "akurasi"]))
    k[2].metric("YIN (baseline)", pct(acc.loc["YIN (baseline tuner)", "akurasi"]))
    k[3].metric("McNemar DTW vs Euclid", f"p = {p_dtw:.3f}")
    k[4].metric("Median |cent| (asli / fals)", f"{med_asli:.1f} / {med_fals:.1f}", "target 0 / 35", delta_color="off")

    tabs = st.tabs(["Akurasi per variasi", "Akurasi & CI 95%", "Matriks konfusi", "Per senar", "Uji McNemar",
                    "DTW & Sakoe–Chiba", "Contoh kesalahan"])
    with tabs[0]:
        tampil(fig_bar_variasi(tab_var, tc.VARIASI, "Akurasi 1-NN per variasi (garis titik = tebakan acak 1/6)", 1 / 6))
        st.dataframe(fmt_tabel(tab_var, persen=tab_var.columns), use_container_width=True)
        st.info("Semiton paling stabil di semua variasi. Chroma sedikit lebih rendah karena tidak membedakan oktaf (E2 ↔ E4). "
                "MFCC mendekati tebakan acak karena merepresentasikan warna suara, bukan tinggi nada.")
    with tabs[1]:
        tampil(fig_ci(acc))
        st.dataframe(fmt_tabel(acc, persen=["akurasi", "CI 95% bawah", "CI 95% atas"], desimal=["F1 makro"]),
                     use_container_width=True)
    with tabs[2]:
        m = st.selectbox("Metode", T1["metode"], index=T1["metode"].index("semiton | DTW"))
        c1, c2 = st.columns([1, 1])
        with c1:
            tampil(fig_konfusi(hasil.label, hasil[m].map(tc.SENAR), tc.LABEL, m))
        with c2:
            e2e4 = int(((hasil.label == "E2") & (hasil[m] == 64)).sum() + ((hasil.label == "E4") & (hasil[m] == 40)).sum())
            st.metric("E2 ↔ E4 tertukar", e2e4)
            st.caption("Pasangan E2 dan E4 bernama nada sama (E) dan berjarak dua oktaf. Fitur chroma tidak dapat membedakannya.")
    with tabs[3]:
        m = st.selectbox("Metode ", T1["metode"], index=T1["metode"].index("semiton | DTW"))
        st.dataframe(fmt_tabel(T1["per_kelas"][m], desimal=["presisi", "recall", "F1"]), use_container_width=True)
        st.dataframe(T1["jumlah_uji"], use_container_width=True)
        st.caption("Tabel kedua: jumlah rekaman uji per senar dan jenis gitar (sebelum diberi 4 variasi).")
    with tabs[4]:
        st.markdown("Uji McNemar eksak: hanya menghitung bunyi yang ditebak benar oleh satu metode tetapi salah oleh metode lain.")
        st.dataframe(fmt_tabel(T1["mcnemar"], p=["p-value"]), use_container_width=True)
        st.markdown("**Hanya variasi `pendek`** (panjang uji jauh berbeda dari template):")
        st.dataframe(fmt_tabel(T1["mcnemar_pendek"], p=["p-value"]), use_container_width=True)
    with tabs[5]:
        c1, c2 = st.columns(2)
        with c1:
            T = A["F_templ"]["semiton"][tc.PITCH.index(45)]; F = T1["contoh_pendek"]["F"]
            jarak, _, jalur = tc.dtw(T, F)
            tampil(fig_warping(T, F, jalur, f"Template A2 vs uji A2 (pendek 0,8 detik)\nDTW ternormalisasi = {jarak / (len(T) + len(F)):.3f}"))
        with c2:
            st.markdown("**Jendela Sakoe–Chiba** (fitur semiton)")
            st.dataframe(fmt_tabel(T1["sakoe"], persen=["akurasi", "akurasi 'pendek'"], desimal=["waktu (detik)"]),
                         use_container_width=True)
            st.caption("Jendela sempit mempercepat komputasi. Jika panjang dua deret sangat berbeda, jendela yang terlalu sempit "
                       "memaksa jalur mengikuti diagonal sehingga DTW kembali mirip Euclid resampling.")
    with tabs[6]:
        m = st.selectbox("Metode  ", T1["metode"], index=T1["metode"].index("semiton | DTW"))
        salah = hasil[hasil[m] != hasil.pitch][["instrumen", "sumber", "velocity", "variasi", "label", m]].copy()
        salah[m] = salah[m].map(tc.SENAR)
        st.write(f"{len(salah)} dari {len(hasil)} bunyi salah ditebak.")
        st.dataframe(salah.rename(columns={m: "prediksi", "label": "sebenarnya"}), use_container_width=True)
        if len(salah):
            st.caption("Bila kesalahan menumpuk di satu gitar dan muncul di semua variasi, penyebabnya adalah rekaman/gitar itu "
                       "sendiri (warna suara jauh dari template), bukan variasinya.")


# =============================================================================== 3. hasil chord
elif halaman.startswith("3"):
    st.title("Hasil analisis Bagian 2: Cek chord (8 chord)")
    hasil2, acc2, verif = T2["hasil"], T2["akurasi"], T2["verif"]
    tab_var2 = akurasi_per_variasi(hasil2, T2["metode"], "chord", tc.VARIASI2)
    hilang = [c for c in tc.NAMA_CHORD if c not in set(hasil2.chord)]
    _, _, p_dtw = tc.mcnemar(hasil2.chord.to_numpy(), hasil2["chroma | DTW"].to_numpy(), hasil2["chroma | Euclid resampling"].to_numpy())
    terbaik_ver = verif["ROC-AUC"].idxmax()

    k = st.columns(5)
    k[0].metric("chroma | DTW", pct(acc2.loc["chroma | DTW", "akurasi"]))
    k[1].metric("chroma | Euclid resampling", pct(acc2.loc["chroma | Euclid resampling", "akurasi"]))
    k[2].metric("Template biner (baseline)", pct(acc2.loc["template biner (baseline)", "akurasi"]))
    k[3].metric("McNemar DTW vs Euclid", f"p = {p_dtw:.3f}")
    k[4].metric("Verifikasi: AUC terbaik", f"{verif.loc[terbaik_ver, 'ROC-AUC']:.3f}", terbaik_ver, delta_color="off")
    if hilang:
        st.warning(f"Chord {', '.join(hilang)} tidak ada di data uji: tidak ada gitar uji yang memiliki semua nadanya di NSynth. "
                   "Akurasi dan F1 makro dihitung dari chord yang tersedia saja.")

    tabs = st.tabs(["Akurasi per variasi", "Akurasi & CI 95%", "Matriks konfusi", "Per chord", "Uji McNemar",
                    "Jalur warping & Sakoe–Chiba", "Verifikasi benar/salah"])
    with tabs[0]:
        tampil(fig_bar_variasi(tab_var2, tc.VARIASI2, "Akurasi klasifikasi chord per variasi (garis titik = tebakan acak 1/8)",
                               1 / 8, baseline="template biner (baseline)"))
        st.dataframe(fmt_tabel(tab_var2, persen=tab_var2.columns), use_container_width=True)
    with tabs[1]:
        tampil(fig_ci(acc2))
        st.dataframe(fmt_tabel(acc2, persen=["akurasi", "CI 95% bawah", "CI 95% atas"], desimal=["F1 makro"]),
                     use_container_width=True)
    with tabs[2]:
        m = st.selectbox("Metode", T2["metode"], index=T2["metode"].index("chroma | DTW"))
        c1, c2 = st.columns([1, 1])
        with c1:
            tampil(fig_konfusi(hasil2.chord, hasil2[m], tc.NAMA_CHORD, m))
        with c2:
            for a, b in [("A", "Am"), ("E", "Em"), ("D", "Dm")]:
                n = int(((hasil2.chord == a) & (hasil2[m] == b)).sum() + ((hasil2.chord == b) & (hasil2[m] == a)).sum())
                st.metric(f"{a} ↔ {b} tertukar", n)
            st.caption("Pasangan mayor/minor hanya berbeda satu nada (setengah nada), sehingga paling sulit dibedakan.")
    with tabs[3]:
        m = st.selectbox("Metode ", T2["metode"], index=T2["metode"].index("chroma | DTW"))
        st.dataframe(fmt_tabel(T2["per_kelas"][m], desimal=["presisi", "recall", "F1"]), use_container_width=True)
    with tabs[4]:
        st.dataframe(fmt_tabel(T2["mcnemar"], p=["p-value"]), use_container_width=True)
        st.markdown("**Hanya variasi `lambat`** (urutan dan waktu munculnya nada paling berbeda dari template):")
        st.dataframe(fmt_tabel(T2["mcnemar_lambat"], p=["p-value"]), use_container_width=True)
    with tabs[5]:
        c1, c2 = st.columns(2)
        cl = T2["contoh_lambat"]; T = A["F_templ2"]["chroma"][tc.NAMA_CHORD.index(cl["chord"])]; F = cl["F"]
        jarak, _, jalur = tc.dtw(T, F)
        with c1:
            tampil(fig_fitur(F, f"Chroma uji {cl['chord']} (genjrengan lambat)", "chroma"))
        with c2:
            tampil(fig_warping(T, F, jalur, f"Template {cl['chord']} vs uji {cl['chord']} lambat\n"
                                            f"DTW ternormalisasi = {jarak / (len(T) + len(F)):.3f}"))
        st.markdown("**Jendela Sakoe–Chiba** (fitur chroma)")
        st.dataframe(fmt_tabel(T2["sakoe"], persen=["akurasi", "akurasi 'lambat'"], desimal=["waktu (detik)"]),
                     use_container_width=True)
    with tabs[6]:
        st.markdown("""
Pertanyaan: *"Saya mau main chord X, sudah benar belum?"* Skor = jarak ke template chord tujuan; di atas ambang dinyatakan **salah**.
Ambang ditentukan lintas gitar dengan indeks Youden. Baris **aturan terdekat = target** adalah aturan yang dipakai aplikasi di halaman 5:
benar jika chord paling mirip sama dengan chord tujuan.
""")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            for (s, (fpr, tpr)), w in zip(T2["roc"].items(), [BIRU, JINGGA, HIJAU, MERAH]):
                ax.plot(fpr, tpr, color=w, lw=2, label=f"{s} (AUC {verif.loc[s, 'ROC-AUC']:.3f})")
            r = verif.loc["aturan terdekat = target"]
            ax.scatter([1 - r.spesifisitas], [r.sensitivitas], color="black", marker="*", s=120, zorder=5,
                       label="aturan terdekat = target")
            ax.plot([0, 1], [0, 1], ":", color="gray")
            ax.set_xlabel("1 - spesifisitas (alarm palsu)"); ax.set_ylabel("sensitivitas (kesalahan terdeteksi)")
            ax.set_title("Kurva ROC verifikasi chord", fontsize=10); ax.legend(fontsize=7)
            fig.tight_layout(); tampil(fig)
        with c2:
            sk = T2["skor_ver"]; kel = tc.VARIASI2 + ["nada salah", "senar mati"]
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            ax.boxplot([sk.loc[sk.variasi == k_, "chroma | DTW"] for k_ in kel])
            ax.set_xticks(range(1, len(kel) + 1)); ax.set_xticklabels(kel, fontsize=8)
            ax.axvspan(len(tc.VARIASI2) + 0.5, len(kel) + 0.5, color=MERAH, alpha=0.07)
            ax.set_ylabel("jarak DTW ke template chord tujuan")
            ax.set_title("Sebaran skor chroma | DTW (area merah = chord salah)", fontsize=10)
            fig.tight_layout(); tampil(fig)
        st.dataframe(fmt_tabel(verif, persen=["sensitivitas", "CI sens. bawah", "CI sens. atas", "spesifisitas",
                                              "CI spes. bawah", "CI spes. atas", "akurasi verifikasi"], desimal=["ROC-AUC"]),
                     use_container_width=True)
        st.markdown("**Proporsi dinyatakan SALAH per kelompok bunyi** (chord benar: idealnya 0%; chord salah: idealnya 100%)")
        st.dataframe(fmt_tabel(T2["rinci"], persen=T2["rinci"].columns), use_container_width=True)
        st.info("Nada salah lebih mudah terdeteksi daripada senar mati: nada dari senar yang mati sering masih terdengar dari "
                "senar lain di oktaf berbeda, dan chroma tidak membedakan oktaf.")


# =============================================================================== input audio
def input_audio(kunci):
    sumber = None
    if hasattr(st, "audio_input"):
        sumber = st.audio_input("Tekan mikrofon untuk merekam, tekan lagi untuk berhenti", key=f"{kunci}_rekam")
    else:
        st.warning("Versi Streamlit belum mendukung perekaman (perlu ≥ 1.40). Gunakan unggah file.")
    with st.expander("atau unggah file WAV"):
        unggah = st.file_uploader("File WAV", type=["wav"], key=f"{kunci}_unggah")
    sumber = sumber or unggah
    if sumber is None:
        return None
    try:
        x = tc.muat(sumber.getvalue())
    except Exception as e:
        st.error(f"File tidak bisa dibaca sebagai WAV ({e}). Ubah dulu ke WAV, misalnya dengan ffmpeg.")
        return None
    if len(x) < 0.4 * tc.SR:
        st.warning("Rekaman terlalu pendek. Biarkan bunyi berlangsung ±2 detik.")
        return None
    return x


# =============================================================================== 4. tuner live
if halaman.startswith("4"):
    st.title("🎸 Coba tuner: rekam satu senar")
    c_mode, c_info = st.columns([1, 2])
    with c_mode:
        mode = st.radio("Mode", ["Otomatis (DTW menebak senar)", "Manual (pilih senar)"])
        manual = st.selectbox("Senar", tc.LABEL, format_func=lambda s: f"Senar {tc.NOMOR_SENAR[s]}: {s}") \
            if mode.startswith("Manual") else None
    with c_info:
        st.markdown("""
**Cara pakai:** redam senar lain dengan telapak tangan, tekan mikrofon, petik **satu** senar, biarkan berbunyi ±2 detik, lalu hentikan.

**Cara kerja:** DTW (fitur semiton) mencocokkan bunyi dengan 6 template senar. Setelah senarnya diketahui, frekuensi dasar
diukur dengan menjumlahkan energi di harmoniknya (±1 semitone dari senar), lalu dihitung selisihnya dalam *cent*
(100 cent = 1 semitone; ±5 cent = pas).
""")
    x = input_audio("tuner")
    if x is not None:
        r = tc.tuner_live(x, A, manual)
        s = tc.SENAR[r["pitch"]]
        c1, c2, c3 = st.columns([1.1, 1.3, 1.3])
        with c1:
            st.markdown(f"### Senar {tc.NOMOR_SENAR[s]}: **{s}**")
            st.metric("Frekuensi terukur", f"{r['f0']:.1f} Hz", f"target {tc.midi_ke_hz(r['pitch']):.1f} Hz", delta_color="off")
            st.metric("Selisih", f"{r['cent']:+.1f} cent")
            (st.success if r["status"] == "PAS" else st.warning)(r["status"].upper())
            if manual and tc.SENAR[r["pitch_dtw"]] != manual:
                st.info(f"DTW menilai bunyi ini paling mirip senar **{tc.SENAR[r['pitch_dtw']]}**. Pastikan senar yang dipetik benar.")
            if abs(r["cent"]) > 90:
                st.caption("Selisih mendekati batas pencarian; kemungkinan senar sangat fals atau senar lain ikut berbunyi.")
        with c2:
            tampil(fig_gauge(r["cent"]))
        with c3:
            tampil(fig_jarak(r["jarak"], tc.LABEL, tc.SENAR[r["pitch_dtw"]], "Jarak DTW ke 6 template senar"))
        st.audio(np.clip(r["x"] / (np.abs(r["x"]).max() + 1e-9), -1, 1), sample_rate=tc.SR)
        with st.expander("Lihat di balik layar", expanded=False):
            tampil(fig_gelombang_spektrum(r["x"], tc.midi_ke_hz(r["pitch"]), r["f0"]))
            c1, c2 = st.columns(2)
            with c1:
                tampil(fig_fitur(r["F"], "Fitur semiton rekaman"))
            with c2:
                T = A["F_templ"]["semiton"][tc.PITCH.index(r["pitch"])]
                tampil(fig_warping(T, r["F"], r["jalur"], f"Jalur warping: template {s} vs rekaman", label_x="frame rekaman"))
            st.caption(f"Sebagai pembanding, YIN rentang lebar mendeteksi {r['yin_lebar']:.1f} Hz. Bila angka ini jauh dari "
                       "senar target (misalnya ±82 Hz pada senar lain), kemungkinan senar E rendah ikut berdengung.")


# =============================================================================== 5. cek chord live
if halaman.startswith("5"):
    st.title("🎶 Coba cek chord: rekam satu genjrengan")
    c1, c2 = st.columns([1, 2])
    with c1:
        target = st.selectbox("Chord yang ingin dimainkan", tc.NAMA_CHORD, index=tc.NAMA_CHORD.index("G"))
        st.markdown(f"**Posisi jari** (senar 6 → 1): `{tc.POSISI_JARI[target]}`")
        st.markdown("**Nada**: " + ", ".join(tc.nama_nada(p) for p in tc.CHORD[target]))
        st.markdown("**Kelas nada**: " + ", ".join(tc.NAMA_NADA[k] for k in sorted(tc.pc(tc.CHORD[target]))))
        with st.expander("Pengaturan diagnosis"):
            b_kurang = st.slider("Ambang 'nada kurang'", 0.1, 0.6, 0.3, 0.05)
            b_lebih = st.slider("Ambang 'nada berlebih'", 0.3, 0.9, 0.5, 0.05)
    with c2:
        st.markdown("""
**Cara pakai:** pilih chord, tekan mikrofon, genjreng chord sekali, biarkan berbunyi ±2 detik, lalu hentikan.

**Cara kerja:** DTW (fitur chroma) mencocokkan genjrengan dengan 8 template chord. Dinyatakan **BENAR** bila chord paling mirip
sama dengan chord tujuan. Petunjuk *nada kurang/berlebih* membandingkan profil 12 nada rekaman dengan nada chord; ini heuristik,
jadi gunakan sebagai petunjuk, bukan vonis.
""")
    x = input_audio("chord")
    if x is not None:
        r = tc.cek_chord_live(x, target, A, b_kurang, b_lebih)
        c1, c2, c3 = st.columns([1, 1.3, 1.3])
        with c1:
            st.markdown(f"### Target: **{target}**")
            st.metric("Paling mirip", r["terdekat"])
            if r["benar"]:
                st.success("BENAR")
            else:
                st.error(f"SALAH: genjrengan lebih mirip chord {r['terdekat']}")
            if r["kurang"]:
                st.warning("Cek nada kurang: " + ", ".join(r["kurang"]))
            if r["lebih"]:
                st.warning("Cek nada berlebih: " + ", ".join(r["lebih"]))
            if not r["kurang"] and not r["lebih"]:
                st.caption("Tidak ada nada yang mencurigakan.")
        with c2:
            tampil(fig_jarak(r["jarak"], tc.NAMA_CHORD, r["terdekat"], "Jarak DTW ke 8 template chord"))
        with c3:
            tampil(fig_profil(r["profil"], target))
        st.audio(np.clip(r["x"] / (np.abs(r["x"]).max() + 1e-9), -1, 1), sample_rate=tc.SR)
        with st.expander("Lihat di balik layar", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                tampil(fig_fitur(r["F"], "Fitur chroma rekaman", "chroma"))
            with c2:
                T = A["F_templ2"]["chroma"][tc.NAMA_CHORD.index(target)]
                tampil(fig_warping(T, r["F"], r["jalur"], f"Jalur warping: template {target} vs rekaman", label_x="frame rekaman"))
            tampil(fig_gelombang_spektrum(r["x"]))
