#!/usr/bin/env bash
# Coleta de dados de desempenho do mergesort — CCO085 IESB 2026/2
#
# Gera CSVs em results/, no formato do medir_speedup.sh do professor: as 3 primeiras colunas
# (threads|processos, repeticao, tempo_s) seguem a convenção dele; colunas extras vêm no FIM.
#
# Experimentos (todos conferem o checksum contra o gabarito; se divergir, o script ABORTA):
#   baseline : código do professor (build/baseline_prof)         -> results/baseline.csv
#   forte    : n fixo, varia threads / processos                 -> results/openmp_escalabilidade.csv
#                                                                    results/mpi_escalabilidade.csv
#   fraca    : n proporcional a p (N_BASE por thread/processo)   -> results/openmp_fraca.csv
#                                                                    results/mpi_fraca.csv
#   cutoff   : varia o cutoff do OpenMP (n e threads fixos)      -> results/openmp_cutoff.csv
#
# Uso:   ./scripts/medir.sh                 (roda tudo)
#        ./scripts/medir.sh forte           (só um experimento: baseline forte fraca cutoff)
# Ajustes por variável de ambiente (valores padrão entre parênteses):
#   TAMANHOS ("5000000 10000000 20000000")   REP (6)   N_BASE (2500000)
#   CUTOFFS  ("256 1024 4096 16384 65536 262144 1048576")
#   NUCLEOS  (nproc)    MPIRUN (mpirun)    MPIFLAGS ("--use-hwthread-cpus --oversubscribe")
#
# REP = total de execuções por configuração; a PRIMEIRA é descartada (aquecimento), como no
# script do professor. Com REP=6 sobram 5 medidas por configuração.
# As faixas de threads/processos são potências de 2 até NUCLEOS (e o próprio NUCLEOS).
# Nada é medido acima de NUCLEOS: oversubscribe não entra na curva de speedup.
set -euo pipefail
cd "$(dirname "$0")/.."

TAMANHOS=${TAMANHOS:-"5000000 10000000 20000000"}
REP=${REP:-6}
N_BASE=${N_BASE:-2500000}
CUTOFFS=${CUTOFFS:-"256 1024 4096 16384 65536 262144 1048576"}
NUCLEOS=${NUCLEOS:-$(nproc)}
MPIRUN=${MPIRUN:-mpirun}
MPIFLAGS=${MPIFLAGS:---use-hwthread-cpus --oversubscribe}
CUTOFF_PADRAO=16384
SAIDA=results
EXPERIMENTOS=("$@"); [ ${#EXPERIMENTOS[@]} -eq 0 ] && EXPERIMENTOS=(baseline forte fraca cutoff)

for b in baseline_prof seq_check omp_mergesort mpi_mergesort; do
    [ -x "build/$b" ] || { echo "Falta build/$b: rode 'make' primeiro."; exit 1; }
done
[ "$REP" -ge 2 ] || { echo "REP precisa ser >= 2 (a 1a execucao e descartada)."; exit 1; }
mkdir -p "$SAIDA"

CPU=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ *//' | tr ',' ' ' || true)
CPU=${CPU:-desconhecido}

campo() {  # campo <chave> <linha>  -> valor
    echo "$2" | tr ' ' '\n' | grep "^$1=" | head -1 | cut -d= -f2
}

declare -A GAB
preparar() {  # preparar <n>: calcula o gabarito (checksum do seq_check) uma vez por n
    [ -n "${GAB[$1]:-}" ] || GAB[$1]=$(campo checksum "$(build/seq_check "$1")")
}
conferir() {  # conferir <linha> <n>: aborta se a versao paralela divergir do gabarito
    local ord chk; ord=$(campo ordenado "$1"); chk=$(campo checksum "$1")
    if [ "$ord" != "SIM" ] || [ "$chk" != "${GAB[$2]}" ]; then
        echo "ERRO: resultado diferente do gabarito (n=$2): $1" >&2; exit 1
    fi
}

lista_p() {  # 1 2 4 8 ... ate NUCLEOS (e o proprio NUCLEOS, se nao for potencia de 2)
    local p=1 l=""
    while [ "$p" -le "$NUCLEOS" ]; do l="$l $p"; p=$((p * 2)); done
    local ultimo=${l##* }
    [ "$ultimo" -ne "$NUCLEOS" ] && l="$l $NUCLEOS"
    echo $l
}

# ---------------------------------------------------------------------------------------------
# Cabecalhos
# ---------------------------------------------------------------------------------------------
CAB_OMP="threads,repeticao,tempo_s,n,cutoff,nucleos_logicos,cpu"
CAB_MPI="processos,repeticao,tempo_s,n,modo,t_scatter,t_local,t_comb_comm,t_comb_merge,nucleos_logicos,cpu"

medir_omp() {  # medir_omp <arquivo> <threads> <n> <cutoff>
    local arq=$1 t=$2 n=$3 c=$4 out tempo
    for r in $(seq 1 "$REP"); do
        out=$(OMP_NUM_THREADS=$t build/omp_mergesort "$n" "$c" 2>/dev/null)
        conferir "$out" "$n"
        tempo=$(campo tempo "$out")
        [ "$r" -gt 1 ] && echo "$t,$((r - 1)),$tempo,$n,$c,$NUCLEOS,$CPU" >> "$arq"
    done
    return 0
}

medir_mpi() {  # medir_mpi <arquivo> <processos> <n> <modo>
    local arq=$1 p=$2 n=$3 modo=$4 out linha det
    for r in $(seq 1 "$REP"); do
        out=$($MPIRUN $MPIFLAGS -np "$p" build/mpi_mergesort "$n" "$modo" 2>&1)
        linha=$(echo "$out" | grep '^n=' | head -1)
        det=$(echo "$out" | grep '^\[mpi\]' | head -1)
        conferir "$linha" "$n"
        [ "$r" -gt 1 ] && echo "$p,$((r - 1)),$(campo tempo "$linha"),$n,$modo,$(campo t_scatter "$det"),$(campo t_local "$det"),$(campo t_comb_comm "$det"),$(campo t_comb_merge "$det"),$NUCLEOS,$CPU" >> "$arq"
    done
    return 0
}

# ---------------------------------------------------------------------------------------------
# Experimentos
# ---------------------------------------------------------------------------------------------
exp_baseline() {
    echo "== baseline (codigo do professor)"
    local arq=$SAIDA/baseline.csv
    echo "n,repeticao,tempo_s,nucleos_logicos,cpu" > "$arq"
    for n in $TAMANHOS; do
        for r in $(seq 1 "$REP"); do
            local out; out=$(build/baseline_prof "$n")
            [ "$(campo ordenado "$out")" = "SIM" ] || { echo "ERRO: baseline nao ordenou" >&2; exit 1; }
            [ "$r" -gt 1 ] && echo "$n,$((r - 1)),$(campo tempo "$out"),$NUCLEOS,$CPU" >> "$arq"
        done
        echo "   n=$n concluido"
    done
}

exp_forte() {
    echo "== escalabilidade FORTE (n fixo)"
    echo "$CAB_OMP" > "$SAIDA/openmp_escalabilidade.csv"
    echo "$CAB_MPI" > "$SAIDA/mpi_escalabilidade.csv"
    for n in $TAMANHOS; do
        preparar "$n"
        for p in $(lista_p); do
            medir_omp "$SAIDA/openmp_escalabilidade.csv" "$p" "$n" "$CUTOFF_PADRAO"
            for modo in arvore gather; do
                medir_mpi "$SAIDA/mpi_escalabilidade.csv" "$p" "$n" "$modo"
            done
            echo "   n=$n p=$p concluido"
        done
    done
}

exp_fraca() {
    echo "== escalabilidade FRACA (n = $N_BASE * p)"
    echo "$CAB_OMP" > "$SAIDA/openmp_fraca.csv"
    echo "$CAB_MPI" > "$SAIDA/mpi_fraca.csv"
    for p in $(lista_p); do
        local n=$((N_BASE * p))
        preparar "$n"
        medir_omp "$SAIDA/openmp_fraca.csv" "$p" "$n" "$CUTOFF_PADRAO"
        medir_mpi "$SAIDA/mpi_fraca.csv" "$p" "$n" arvore
        echo "   p=$p n=$n concluido"
    done
}

exp_cutoff() {
    local n=${TAMANHOS##* }   # o maior (ultimo) tamanho da lista
    echo "== estudo de CUTOFF (OpenMP, n=$n, threads=$NUCLEOS)"
    echo "cutoff,repeticao,tempo_s,n,threads,nucleos_logicos,cpu" > "$SAIDA/openmp_cutoff.csv"
    preparar "$n"
    for c in $CUTOFFS; do
        local tmp; tmp=$(mktemp)
        medir_omp "$tmp" "$NUCLEOS" "$n" "$c"
        # reordena as colunas: cutoff,repeticao,tempo_s,n,threads,nucleos,cpu
        awk -F, -v OFS=, '{print $5,$2,$3,$4,$1,$6,$7}' "$tmp" >> "$SAIDA/openmp_cutoff.csv"
        rm -f "$tmp"
        echo "   cutoff=$c concluido"
    done
}

for e in "${EXPERIMENTOS[@]}"; do
    case "$e" in
        baseline) exp_baseline ;;
        forte)    exp_forte ;;
        fraca)    exp_fraca ;;
        cutoff)   exp_cutoff ;;
        *) echo "Experimento desconhecido: $e (use: baseline forte fraca cutoff)"; exit 1 ;;
    esac
done

echo
echo "CPU: $CPU | nucleos logicos usados como limite: $NUCLEOS"
echo "Dados brutos em $SAIDA/. Para os graficos: python3 scripts/graficos.py"
