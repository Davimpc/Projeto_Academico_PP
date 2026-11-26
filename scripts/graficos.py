#!/usr/bin/env python3
"""Gráficos e tabela-resumo do mergesort paralelo — CCO085 IESB 2026/2.

Lê os CSVs gerados por scripts/medir.sh (pasta results/) e escreve em graficos/:
  tempo.png, speedup.png, eficiencia.png   escalabilidade forte (OpenMP, MPI árvore, MPI gather)
  mpi_overhead.png                         tempo do MPI decomposto em comunicação x cálculo
  escalabilidade_fraca.png                 escalabilidade fraca (n proporcional a p)
  cutoff.png                               sensibilidade do OpenMP ao cutoff
e results/resumo.csv com mediana, speedup, eficiência e a métrica de Karp-Flatt.

Requisitos: python3, pandas, matplotlib.   (pip install pandas matplotlib)
Uso:        python3 scripts/graficos.py [--amdahl F]
            --amdahl F   desenha também a curva de Amdahl para a fração serial F (0 < F < 1),
                         por exemplo a estimada no perfilamento.

Definições (p = threads ou processos):
  Speedup      S(p) = T_baseline(n) / T(p)       (T_baseline = código do professor, mesmo n)
  Eficiência   E(p) = S(p) / p
  Karp-Flatt   e(p) = (1/S - 1/p) / (1 - 1/p)    fração serial EFETIVA, inferida das medidas:
               se e(p) cresce com p, o que limita é overhead (comunicação, memória, desbalanceamento);
               se e(p) é ~constante, o limite é uma parte serial fixa (Amdahl puro).
Todas as estatísticas usam a MEDIANA das repetições (robusta a picos de ruído).
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTADOS = "results"
SAIDA = "graficos"


def carregar(nome):
    caminho = os.path.join(RESULTADOS, nome)
    if not os.path.exists(caminho):
        print(f"(aviso) {caminho} não existe: gráficos que dependem dele serão pulados")
        return None
    df = pd.read_csv(caminho)
    return df if len(df) else None


def milhar(n):
    """1234567 -> '1.234.567' (só o número; não mexe em vírgulas do texto)."""
    return f"{int(n):,}".replace(",", ".")


def salvar(fig, nome):
    os.makedirs(SAIDA, exist_ok=True)
    caminho = os.path.join(SAIDA, nome)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print("gerado", caminho)


def mediana(df, chaves):
    return df.groupby(chaves, as_index=False)["tempo_s"].median()


def montar_series(omp, mpi, base):
    """Devolve dict {rótulo: DataFrame[n, p, tempo, speedup, eficiencia, karp_flatt]}."""
    tbase = None
    if base is not None:
        tbase = base.groupby("n")["tempo_s"].median().to_dict()
    series = {}

    def processa(rotulo, df, col_p):
        m = mediana(df, ["n", col_p]).rename(columns={col_p: "p"})
        ref = tbase if tbase is not None else {
            n: g.loc[g["p"] == g["p"].min(), "tempo_s"].iloc[0] for n, g in m.groupby("n")}
        m["speedup"] = m.apply(lambda r: ref[r["n"]] / r["tempo_s"], axis=1)
        m["eficiencia"] = m["speedup"] / m["p"]
        m["karp_flatt"] = m.apply(
            lambda r: (1 / r["speedup"] - 1 / r["p"]) / (1 - 1 / r["p"]) if r["p"] > 1 else float("nan"),
            axis=1)
        series[rotulo] = m

    if omp is not None:
        processa("OpenMP (tasks)", omp, "threads")
    if mpi is not None:
        for modo, rot in (("arvore", "MPI (merge em árvore)"), ("gather", "MPI (Gatherv + merge)")):
            sub = mpi[mpi["modo"] == modo]
            if len(sub):
                processa(rot, sub, "processos")
    return series


def grafico_metrica(series, base, coluna, titulo, ylabel, arquivo, amdahl=None, ideal=False, ylim0=False):
    tamanhos = sorted({n for m in series.values() for n in m["n"].unique()})
    if not tamanhos:
        return
    fig, eixos = plt.subplots(1, len(tamanhos), figsize=(5.2 * len(tamanhos), 4.2), squeeze=False)
    for ax, n in zip(eixos[0], tamanhos):
        pmax = 1
        for rot, m in series.items():
            s = m[m["n"] == n].sort_values("p")
            if len(s):
                ax.plot(s["p"], s[coluna], marker="o", label=rot)
                pmax = max(pmax, s["p"].max())
        ps = sorted({p for m in series.values() for p in m[m["n"] == n]["p"]})
        if ideal and ps:
            ax.plot(ps, ps, "k--", linewidth=1, label="ideal (S = p)")
        if amdahl and ps and coluna == "speedup":
            xs = list(range(1, int(pmax) + 1))
            ax.plot(xs, [1 / (amdahl + (1 - amdahl) / x) for x in xs], ":", color="gray",
                    label=f"Amdahl (f = {amdahl:g})")
        if coluna == "eficiencia":
            ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
        ax.set_title(f"n = {milhar(n)}")
        ax.set_xlabel("threads / processos (p)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ps)
        ax.set_xticklabels([str(p) for p in ps])
        if ylim0:
            ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    eixos[0][0].legend(fontsize=8)
    fig.suptitle(titulo)
    salvar(fig, arquivo)


def grafico_overhead(mpi):
    if mpi is None:
        return
    n = mpi["n"].max()
    sub = mpi[mpi["n"] == n]
    modos = [m for m in ("arvore", "gather") if m in set(sub["modo"])]
    if not modos:
        return
    fig, eixos = plt.subplots(1, len(modos), figsize=(5.6 * len(modos), 4.2), squeeze=False)
    partes = [("t_scatter", "Scatterv (distribuir)"), ("t_local", "ordenação local"),
              ("t_comb_comm", "comunicação da combinação (inclui espera)"),
              ("t_comb_merge", "merge da combinação")]
    for ax, modo in zip(eixos[0], modos):
        m = sub[sub["modo"] == modo].groupby("processos")[[c for c, _ in partes]].median().sort_index()
        base = pd.Series(0.0, index=m.index)
        for col, rot in partes:
            ax.bar([str(p) for p in m.index], m[col], bottom=base, label=rot)
            base = base + m[col]
        ax.set_title(f"MPI, modo {modo} (n = {milhar(n)})")
        ax.set_xlabel("processos (p)")
        ax.set_ylabel("tempo (s), máximo entre ranks")
        ax.grid(alpha=0.3, axis="y")
    eixos[0][0].legend(fontsize=7)
    fig.suptitle("Decomposição do tempo MPI (cada parcela é o MÁXIMO entre ranks; a soma pode exceder o total)")
    salvar(fig, "mpi_overhead.png")


def grafico_fraca(omp, mpi):
    series = {}
    if omp is not None:
        series["OpenMP (tasks)"] = mediana(omp, ["n", "threads"]).rename(columns={"threads": "p"})
    if mpi is not None:
        series["MPI (árvore)"] = mediana(mpi, ["n", "processos"]).rename(columns={"processos": "p"})
    if not series:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ref_p1 = None
    for rot, m in series.items():
        m = m.sort_values("p")
        ax1.plot(m["p"], m["tempo_s"], marker="o", label=rot)
        t1 = m.loc[m["p"] == m["p"].min(), "tempo_s"].iloc[0]
        ax2.plot(m["p"], t1 / m["tempo_s"], marker="o", label=rot)
        if ref_p1 is None:
            n0 = m.loc[m["p"] == m["p"].min(), "n"].iloc[0]
            ref_p1 = (n0, t1, m["p"].tolist(), m["n"].tolist())
    if ref_p1:
        n0, t1, ps, ns = ref_p1
        ax1.plot(ps, [t1] * len(ps), "k--", linewidth=1, label="ideal ingênuo (tempo constante)")
        # o trabalho do mergesort cresce como n log n: com n = n0*p, o tempo "ideal" por processo cresce com log(n)
        esperado = [t1 * math.log2(n) / math.log2(n0) for n in ns]
        ax1.plot(ps, esperado, "k:", linewidth=1.2, label="ideal real (n log n): T1·log n / log n0")
        ax2.plot(ps, [t1 / e for e in esperado], "k:", linewidth=1.2, label="eficiência máxima (n log n)")
    ax1.set_title("Tempo (n = n0·p)")
    ax1.set_ylabel("tempo (s)")
    ax2.set_title("Eficiência fraca  E = T(1, n0) / T(p, n0·p)")
    ax2.set_ylabel("eficiência")
    ax2.set_ylim(bottom=0)
    ps_todos = sorted({int(p) for m in series.values() for p in m["p"]})
    for ax in (ax1, ax2):
        ax.set_xlabel("threads / processos (p)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ps_todos)
        ax.set_xticklabels([str(p) for p in ps_todos])
        ax.minorticks_off()
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Escalabilidade fraca (como o trabalho é n·log n, o tempo constante NÃO é o ideal)")
    salvar(fig, "escalabilidade_fraca.png")


def grafico_cutoff(cut):
    if cut is None:
        return
    m = cut.groupby("cutoff")["tempo_s"].agg(["median", "min", "max"]).sort_index()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(m.index, m["median"], marker="o")
    ax.fill_between(m.index, m["min"], m["max"], alpha=0.2, label="mín–máx")
    n = int(cut["n"].iloc[0]); t = int(cut["threads"].iloc[0])
    ax.set_title(f"Sensibilidade ao cutoff (OpenMP, n = {milhar(n)}, {t} threads)")
    ax.set_xlabel("cutoff (elementos abaixo dos quais não se criam tasks)")
    ax.set_ylabel("tempo (s)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(m.index))
    ax.set_xticklabels([str(int(c)) for c in m.index], rotation=45)
    ax.minorticks_off()
    ax.grid(alpha=0.3)
    ax.legend()
    salvar(fig, "cutoff.png")


def escrever_resumo(series):
    if not series:
        return
    linhas = []
    for rot, m in series.items():
        for _, r in m.sort_values(["n", "p"]).iterrows():
            linhas.append({"versao": rot, "n": int(r["n"]), "p": int(r["p"]),
                           "tempo_mediano_s": round(r["tempo_s"], 6),
                           "speedup": round(r["speedup"], 3),
                           "eficiencia": round(r["eficiencia"], 3),
                           "karp_flatt": round(r["karp_flatt"], 3) if r["p"] > 1 else ""})
    df = pd.DataFrame(linhas)
    caminho = os.path.join(RESULTADOS, "resumo.csv")
    df.to_csv(caminho, index=False)
    print("gerado", caminho)
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amdahl", type=float, default=None, help="fração serial F para a curva de Amdahl")
    args = ap.parse_args()
    if args.amdahl is not None and not 0 < args.amdahl < 1:
        sys.exit("--amdahl precisa estar entre 0 e 1")

    base = carregar("baseline.csv")
    omp = carregar("openmp_escalabilidade.csv")
    mpi = carregar("mpi_escalabilidade.csv")
    series = montar_series(omp, mpi, base)

    if series:
        grafico_metrica(series, base, "tempo_s", "Escalabilidade forte: tempo", "tempo mediano (s)", "tempo.png")
        grafico_metrica(series, base, "speedup", "Escalabilidade forte: speedup sobre o baseline do professor",
                        "speedup", "speedup.png", amdahl=args.amdahl, ideal=True, ylim0=True)
        grafico_metrica(series, base, "eficiencia", "Escalabilidade forte: eficiência (S/p)",
                        "eficiência", "eficiencia.png", ylim0=True)
    grafico_overhead(mpi)
    grafico_fraca(carregar("openmp_fraca.csv"), carregar("mpi_fraca.csv"))
    grafico_cutoff(carregar("openmp_cutoff.csv"))
    escrever_resumo(series)


if __name__ == "__main__":
    main()
