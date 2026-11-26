"""Analiza datasetu Iconclass AI Test Set: wykresy PNG (16:9, pod slajdy) i summary.txt w analysis/plots/.

Uruchomienie: python analysis/dataset_eda.py [ścieżka do .zip | data.json | URL]
Bez argumentu bierze data/iconclass_dataset.zip (albo $CARIS_DATA), a gdy go nie ma, czyta data.json
prosto z iconclass.org (HTTP Range, bez pobierania 3,3 GB obrazów).
"""

import io
import json
import random
import re
import sys
import textwrap
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import iconclass
from caris.data import DATA, SEED, SIZES
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

URL = "https://iconclass.org/testset/779ba2ca9e977c58d818e3823a676973.zip"
LOCAL_ZIP = DATA / "iconclass_dataset.zip"
OUT = Path(__file__).parent / "plots"

DIVISIONS = {
    "0": "0 Sztuka abstrakcyjna",
    "1": "1 Religia i magia",
    "2": "2 Natura",
    "3": "3 Człowiek",
    "4": "4 Społeczeństwo i kultura",
    "5": "5 Idee i pojęcia abstrakcyjne",
    "6": "6 Historia",
    "7": "7 Biblia",
    "8": "8 Literatura",
    "9": "9 Mitologia i starożytność",
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ACCENT, ACCENT_DARK, NEUTRAL = "#2a78d6", "#17437a", "#c5cad3"
INK, MUTED, FAINT, GRID = "#16181d", "#5b6070", "#8a8f9c", "#e6e8ec"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f7fb", "#9cc0ec", ACCENT, ACCENT_DARK])
SLIDE = (13.33, 7.5)
INSTALLED_FONTS = {f.name for f in font_manager.fontManager.ttflist}

plt.rcParams.update({
    "font.family": next(f for f in ("Helvetica Neue", "Segoe UI", "Arial", "DejaVu Sans") if f in INSTALLED_FONTS),
    "figure.dpi": 100, "savefig.dpi": 160, "savefig.bbox": "tight", "savefig.pad_inches": 0.35,
    "figure.facecolor": "white", "font.size": 13, "text.color": INK,
    "axes.edgecolor": FAINT, "axes.labelcolor": MUTED, "axes.labelsize": 13, "axes.labelpad": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 8, "ytick.major.pad": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "legend.frameon": False, "legend.fontsize": 12,
})


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        head = urllib.request.Request(url, method="HEAD")
        self.size = int(urllib.request.urlopen(head).headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.pos + n, self.size)
        if end <= self.pos:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
        data = urllib.request.urlopen(req).read()
        self.pos += len(data)
        return data


def load(source):
    if source.endswith(".json"):
        return json.loads(Path(source).read_text()), None
    zf = zipfile.ZipFile(HttpRangeFile(source) if source.startswith("http") else open(source, "rb"))
    return json.loads(zf.read("data.json")), zf


ICONCLASS = iconclass.init()


def label(code, width=60):
    parts = []
    for part in code.split(":"):
        try:
            parts.append(re.sub(r"\s\((?!\+)[^()]*\)", "", ICONCLASS[part]()))
        except Exception:
            parts.append(part)
    text = " + ".join(parts)
    return text if len(text) <= width else text[: width - 1] + "…"


def base_notation(code):
    return re.split(r"[(:]", code)[0].strip()


def num(n):
    return f"{n:,.0f}".replace(",", " ")


def dec(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x):
    if 0 < x < 0.01:
        return "<0,01%"
    return dec(x, 2 if x < 1 else 1) + "%"


def wrap(text, width=14):
    return textwrap.fill(text, width, break_long_words=False)


def figure(title, subtitle, size=SLIDE, **kw):
    fig, ax = plt.subplots(figsize=size, **kw)
    fig.subplots_adjust(top=0.80)
    fig._titles = (title, subtitle)
    return fig, ax


def save(fig, name, note="Źródło: Iconclass AI Test Set (iconclass.org/testset)"):
    title, subtitle = fig._titles
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    boxes = [ax.get_tightbbox().transformed(inv) for ax in fig.axes if ax.get_visible()]
    x0, y1, y0 = min(b.x0 for b in boxes), max(b.y1 for b in boxes), min(b.y0 for b in boxes)
    fig.text(x0, y1 + 0.115, title, fontsize=24, fontweight="bold", color=INK, va="bottom")
    fig.text(x0, y1 + 0.05, subtitle, fontsize=14, color=MUTED, va="bottom")
    fig.text(x0, y0 - 0.06, note, fontsize=10, color=FAINT, va="top")
    fig.savefig(OUT / name)
    plt.close(fig)
    print("  ", OUT / name)


def hbar(labels, values, title, subtitle, name, fmt=num, highlight=1):
    fig, ax = figure(title, subtitle, size=(13.33, max(5.5, 0.42 * len(labels) + 2)))
    colors = [ACCENT_DARK if i < highlight else ACCENT for i in range(len(values))]
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.62)
    ax.bar_label(bars, [fmt(v) for v in values[::-1]], padding=6, color=INK, fontsize=12)
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelcolor=INK)
    ax.margins(x=0.1, y=0.01)
    save(fig, name)


def vbar(xs, values, title, subtitle, xlabel, labels=None, xticklabels=None):
    fig, ax = figure(title, subtitle)
    bars = ax.bar(range(len(xs)), values, color=ACCENT, width=0.68)
    if labels:
        ax.bar_label(bars, labels, padding=5, color=INK, fontsize=12)
    ax.set_xticks(range(len(xs)), xticklabels or [str(x) for x in xs])
    ax.set_xlabel(xlabel)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v)))
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.12)
    return fig, ax


def heatmap(ax, matrix, xlabels, ylabels):
    im = ax.imshow(matrix, cmap=BLUES, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.tick_params(axis="x", labeltop=True, labelbottom=False)
    ax.tick_params(labelcolor=INK)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=11,
                    color="white" if v > 50 else INK, fontweight="bold" if i == j and len(matrix) == len(row) else "normal")
    return im


def donut(ax, values, colors, center_big, center_small):
    wedges, _ = ax.pie(values, colors=colors, startangle=90, counterclock=False,
                       wedgeprops={"width": 0.38, "linewidth": 3, "edgecolor": "white"})
    ax.text(0, 0.08, center_big, ha="center", va="center", fontsize=30, fontweight="bold", color=INK)
    ax.text(0, -0.16, center_small, ha="center", va="center", fontsize=13, color=MUTED)
    return wedges


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else (str(LOCAL_ZIP) if LOCAL_ZIP.exists() else URL)
    print("Źródło:", source)
    raw, zf = load(source)
    OUT.mkdir(exist_ok=True)

    empty = sum(c.strip() == "" for v in raw.values() for c in v)
    malformed = sorted({c for v in raw.values() for c in v if c.strip() and not c[0].isdigit()})
    data = {k: [c.strip() for c in v if c.strip() and c[0].isdigit()] for k, v in raw.items()}
    data = {k: v for k, v in data.items() if v}
    codes = [c for v in data.values() for c in v]
    n_works, n_codes = len(data), len(codes)
    divs = sorted(DIVISIONS)

    code_div = Counter(c[0] for c in codes)
    top = [d for d, _ in code_div.most_common(7)]
    rest = [d for d in divs if d not in top]
    fig, ax = figure("Kategorie Iconclass w kodach",
                     "Udział każdej kategorii (pierwsza cyfra kodu) we wszystkich przypisanych kodach")
    wedges = donut(ax, [code_div[d] for d in top] + [sum(code_div[d] for d in rest)], SERIES,
                   num(n_codes), "przypisanych kodów")
    legend = [f"{DIVISIONS[d]}   {pct(100 * code_div[d] / n_codes)}" for d in top]
    legend.append("Pozostałe   " + pct(100 * sum(code_div[d] for d in rest) / n_codes) + "\n" + "\n".join(
        f"   {DIVISIONS[d]}   {pct(100 * code_div[d] / n_codes)}" for d in rest))
    ax.legend(wedges, legend, loc="center left", bbox_to_anchor=(1.05, 0.5), labelspacing=0.9, handlelength=1.2)
    save(fig, "01_kategorie_kody_pie.png")

    work_div = Counter(d for v in data.values() for d in {c[0] for c in v})
    order = sorted(divs, key=lambda d: -work_div[d])
    hbar([DIVISIONS[d] for d in order], [100 * work_div[d] / n_works for d in order],
         "Ile dzieł ma kod z danej kategorii",
         "Odsetek dzieł z co najmniej jednym kodem z kategorii; dzieło może należeć do kilku", "02_kategorie_dziela.png",
         fmt=pct)

    with_br = sum(any("(" in c for c in v) for v in data.values())
    fig, ax = figure("Ponad połowa dzieł ma kod z nawiasem",
                     "Dzieła z co najmniej jednym kodem z nawiasem, np. 25G41(ROSE) albo 61B(+52)")
    wedges = donut(ax, [with_br, n_works - with_br], [ACCENT, NEUTRAL], pct(100 * with_br / n_works),
                   "dzieł z nawiasem")
    ax.legend(wedges, [f"z nawiasem   {num(with_br)}", f"bez nawiasu   {num(n_works - with_br)}"],
              loc="center left", bbox_to_anchor=(1.05, 0.5), labelspacing=1.2)
    save(fig, "03_nawiasy_dziela_pie.png")

    kinds = {
        "Nazwa własna w nawiasie\n25G41(ROSE) = kwiat: róża": sum(bool(re.search(r"\([^+.)]", c)) for c in codes),
        "Klucz (+…) doprecyzowujący\n61B(+52) = postać hist. + portret popiersie": sum("(+" in c for c in codes),
        "Kod złożony A:B\n61B:31D14 = postać hist. + dorosły mężczyzna": sum(":" in c for c in codes),
        "Podwojona litera (wariant)\n31AA = postać kobieca, 57AA = przeciwieństwa": sum(
            bool(re.match(r"\d\d([A-Z])\1", c)) for c in codes),
        "Nawias bez nazwy (...)\n11H(...) = święty, imię nieznane": sum("(...)" in c for c in codes),
    }
    hbar(list(kinds), list(kinds.values()), "Rozszerzenia notacji Iconclass",
         "Ile kodów używa danego rozszerzenia; jeden kod może mieć kilka naraz", "04_nawiasy_typy.png")

    names = Counter(m.group(1) for c in codes if (m := re.search(r"\(([^+.)][^)]*)\)", c)))
    first = {}
    for c in codes:
        if (m := re.search(r"\(([^+.)][^)]*)\)", c)) and m.group(1) not in first:
            first[m.group(1)] = c
    top_names = names.most_common(20)
    hbar([f"{n}  ·  {label(first[n].split(':')[0], 40).split(':')[0]}" for n, _ in top_names],
         [v for _, v in top_names], "Najczęstsze nazwy w nawiasach",
         "Nazwa z nawiasu i kod nadrzędny, do którego ją dopisano (opis z pakietu iconclass)",
         "05_nawiasy_top_nazwy.png", highlight=3)

    mean_len = {d: sum(len(base_notation(c)) for c in codes if c[0] == d) / code_div[d] for d in divs}
    order = sorted(divs, key=lambda d: -mean_len[d])
    hbar([DIVISIONS[d] for d in order], [mean_len[d] for d in order], "Średnia długość kodu wg kategorii",
         "Liczba znaków bez nawiasu i bez części po dwukropku; dłuższy kod = głębiej w hierarchii",
         "06_dlugosc_kodu_kategorie.png", fmt=dec, highlight=0)

    lengths = Counter(min(len(base_notation(c)), 10) for c in codes)
    xs = list(range(1, 11))
    fig, ax = vbar(xs, [lengths[x] for x in xs], "Rozkład długości kodu",
                   "Większość kodów ma 3–7 znaków; dłuższe niż 9 to pojedyncze wyjątki i literówki",
                   "liczba znaków kodu (bez nawiasu i części po dwukropku)",
                   labels=[pct(100 * lengths[x] / n_codes) for x in xs], xticklabels=[*map(str, xs[:-1]), "10+"])
    ax.set_ylabel("liczba kodów")
    save(fig, "07_dlugosc_kodu_rozklad.png")

    per_work = Counter(min(len(v), 15) for v in data.values())
    xs = list(range(1, 16))
    mean_codes = n_codes / n_works
    fig, ax = vbar(xs, [per_work[x] for x in xs], "Ile kodów ma jedno dzieło",
                   f"Średnio {dec(mean_codes)} kodu na dzieło; model musi zwrócić kilka kodów naraz",
                   "liczba kodów na dzieło", xticklabels=[*map(str, xs[:-1]), "15+"])
    ax.axvline(mean_codes - 1, color=INK, linestyle=(0, (4, 4)), linewidth=1.3)
    ax.text(mean_codes - 0.85, ax.get_ylim()[1] * 0.95, f"średnia {dec(mean_codes)}", color=INK, va="top")
    ax.set_ylabel("liczba dzieł")
    save(fig, "08_kody_na_dzielo.png")

    freq = Counter(codes)
    ranked = sorted(freq.values(), reverse=True)
    singles, under5 = sum(v == 1 for v in ranked), sum(v < 5 for v in ranked)
    fig, ax = figure("Długi ogon: większość kodów jest rzadka",
                     "Kody posortowane od najczęstszego; obie osie w skali logarytmicznej")
    ranks = range(1, len(ranked) + 1)
    ax.fill_between(ranks, ranked, color=ACCENT, alpha=0.12, step="post")
    ax.step(ranks, ranked, color=ACCENT, linewidth=2.2, where="post")
    ax.set(xscale="log", yscale="log", xlabel="pozycja kodu w rankingu", ylabel="liczba wystąpień")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: num(v)))
    ax.spines["left"].set_visible(True)
    ax.text(0.97, 0.93, f"{num(len(ranked))} unikalnych kodów\n{pct(100 * singles / len(ranked))} występuje tylko raz\n"
            f"{pct(100 * under5 / len(ranked))} występuje mniej niż 5 razy", transform=ax.transAxes, ha="right",
            va="top", fontsize=15, linespacing=1.6, color=INK)
    save(fig, "09_dlugi_ogon.png")

    top_codes = freq.most_common(20)
    hbar([f"{label(c)}  [{c}]" for c, _ in top_codes], [v for _, v in top_codes], "20 najczęstszych kodów",
         "Liczba dzieł z danym kodem; dominują portrety z kolekcji IIHIM", "10_top_kody.png", highlight=2)

    sets = [{c[0] for c in v} for v in data.values()]
    matrix = [[100 * sum(a in s and b in s for s in sets) / work_div[a] for b in divs] for a in divs]
    fig, ax = figure("Które kategorie występują razem",
                     "Wiersz A, kolumna B: jaki % dzieł z kategorią A ma też kategorię B", size=(13.33, 9))
    heatmap(ax, matrix, divs, [DIVISIONS[d] for d in divs])
    save(fig, "11_wspolwystepowanie_kategorii.png")

    source_of = {k: re.split(r"[_-]", k)[0] for k in data}
    src = Counter(source_of.values())
    top_src = src.most_common(10)
    hbar([s for s, _ in top_src], [100 * v / n_works for _, v in top_src], "Jedna kolekcja to 72% datasetu",
         "Odsetek dzieł z 10 największych kolekcji (prefiks nazwy pliku)", "12_zrodla.png", fmt=pct)

    top6 = [s for s, _ in src.most_common(6)]
    profile = []
    for s in top6:
        works = [v for k, v in data.items() if source_of[k] == s]
        profile.append([100 * sum(any(c[0] == d for c in v) for v in works) / len(works) for d in divs])
    fig, ax = figure("Każda kolekcja ma inny profil",
                     "Jaki % dzieł kolekcji ma kod z danej kategorii", size=(13.33, 7))
    heatmap(ax, profile, [wrap(DIVISIONS[d], 12) for d in divs], [f"{s}  ({num(src[s])})" for s in top6])
    save(fig, "13_kolekcje_kategorie.png")

    keys = sorted(k for k, v in raw.items() if any(c.strip() for c in v))
    random.Random(SEED).shuffle(keys)
    groups = {"test": keys[:SIZES["test"]], "val": keys[SIZES["test"]:SIZES["test"] + SIZES["val"]],
              "train": keys[SIZES["test"] + SIZES["val"]:]}
    fig, ax = figure("Zbiory val i test przypominają resztę danych",
                     f"Odsetek dzieł z kategorią w podziale z caris split (seed {SEED})")
    for i, (name, ks) in enumerate(groups.items()):
        share = [100 * sum(any(c[0] == d for c in data.get(k, [])) for k in ks) / len(ks) for d in divs]
        ax.bar([x + (i - 1) * 0.27 for x in range(10)], share, 0.25, color=SERIES[i], label=f"{name}  ({num(len(ks))} dzieł)")
    ax.set_xticks(range(10), [wrap(DIVISIONS[d], 12) for d in divs], fontsize=11)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    save(fig, "14_split_train_val_test.png")

    depths = range(1, 9)
    classes, enough = [], []
    for k in depths:
        per_class = Counter(t for v in data.values() for t in {base_notation(c)[:k] for c in v})
        classes.append(len(per_class))
        enough.append(100 * sum(n >= 20 for n in per_class.values()) / len(per_class))
    fig, ax = vbar(list(depths), classes, "Im dokładniejszy kod, tym więcej rzadkich klas",
                   "Liczba klas po obcięciu kodu do N znaków; pod liczbą: % klas z co najmniej 20 dziełami",
                   "N, czyli długość obciętego kodu (np. 25G41 przy N=2 daje 25, przy N=3 daje 25G)",
                   labels=[f"{num(c)}\n{pct(e)}" for c, e in zip(classes, enough)])
    ax.set_yscale("log")
    ax.set_ylabel("liczba klas (skala log.)")
    ax.tick_params(axis="y", which="minor", length=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v)))
    ax.set_ylim(top=max(classes) * 6)
    save(fig, "15_obciecie_kodu.png")

    if zf:
        members = {Path(m).name: m for m in zf.namelist() if m.endswith(".jpg")}
        rng = random.Random(0)
        per_div = 3
        fig, axes = plt.subplots(per_div, 10, figsize=(22, 8.5))
        fig._titles = ("Przykładowe dzieła z każdej kategorii",
                       "Losowe dzieła, których wszystkie kody należą do jednej kategorii")
        for j, d in enumerate(divs):
            pure = [k for k, v in data.items() if all(c[0] == d for c in v) and k in members]
            pool = pure if len(pure) >= per_div else [k for k, v in data.items() if any(c[0] == d for c in v)]
            for i, key in enumerate(rng.sample(pool, per_div)):
                ax = axes[i][j]
                ax.imshow(plt.imread(io.BytesIO(zf.read(members[key])), format="jpg"))
                ax.set_axis_off()
                if i == 0:
                    ax.set_title(wrap(DIVISIONS[d], 14), fontsize=13, fontweight="bold", loc="center", pad=10)
        save(fig, "16_przyklady_kategorii.png")

    summary = [
        f"Źródło: {source}",
        f"Dzieła: {num(len(raw))} (z poprawnym kodem: {num(n_works)})",
        f"Przypisane kody: {num(n_codes)}, unikalne: {num(len(freq))}, średnio na dzieło: {dec(mean_codes)}",
        f"Dzieła z kodem z nawiasem: {num(with_br)} ({pct(100 * with_br / n_works)})",
        *(f"{k.splitlines()[0]}: {num(v)}" for k, v in kinds.items()),
        f"Kody występujące raz: {num(singles)} ({pct(100 * singles / len(freq))} unikalnych)",
        f"Brudne dane: {empty} pustych kodów, {len(malformed)} kodów spoza notacji: {malformed}",
        "",
        "Kategoria | kody | % kodów | dzieła | % dzieł | śr. długość",
        *(f"{DIVISIONS[d]} | {num(code_div[d])} | {pct(100 * code_div[d] / n_codes)} | {num(work_div[d])} | "
          f"{pct(100 * work_div[d] / n_works)} | {dec(mean_len[d])}" for d in divs),
    ]
    (OUT / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
