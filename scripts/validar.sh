#!/usr/bin/env bash
# Valida que TODAS as versoes produzem o mesmo vetor ordenado do gabarito.
#
# Gabarito = checksum impresso por build/seq_check (logica do professor).
# Para cada tamanho n, compara o checksum de:
#   - OpenMP com 1, 2, 4, 8 threads
#   - MPI com -np 1, 2, 3, 4, 5, 8  (3 e 5 nao dividem a maioria dos n)
# Tamanhos incluem casos de borda (n=1, 2, 3) e n primo (nao divisivel por p).
#
# Uso: ./scripts/validar.sh [n1 n2 ...]      (padrao: lista abaixo)
# Saida: codigo 0 se tudo passou, 1 se alguma versao divergiu.
set -u
cd "$(dirname "$0")/.."

TAMANHOS=("$@")
[ ${#TAMANHOS[@]} -eq 0 ] && TAMANHOS=(1 2 3 7 1000 99991 1000003 5000000)
THREADS=(1 2 4 8)
PROCS=(1 2 3 4 5 8)
MPIRUN=${MPIRUN:-mpirun}
MPIFLAGS=${MPIFLAGS:---oversubscribe}   # permite -np maior que os nucleos

falhas=0; testes=0

campo() {  # campo <chave> <linha>  -> valor
    echo "$2" | tr ' ' '\n' | grep "^$1=" | head -1 | cut -d= -f2
}

verificar() {  # verificar <rotulo> <n> <gabarito> <linha_de_saida>
    local rot=$1 n=$2 gab=$3 saida=$4
    local ord chk; ord=$(campo ordenado "$saida"); chk=$(campo checksum "$saida")
    testes=$((testes+1))
    if [ "$ord" = "SIM" ] && [ "$chk" = "$gab" ]; then
        printf "  OK     %-18s n=%s\n" "$rot" "$n"
    else
        printf "  FALHOU %-18s n=%s  (ordenado=%s checksum=%s esperado=%s)\n" \
               "$rot" "$n" "${ord:-?}" "${chk:-?}" "$gab"
        falhas=$((falhas+1))
    fi
}

[ -x build/seq_check ] || { echo "Falta build/seq_check: rode 'make seq'"; exit 2; }

for n in "${TAMANHOS[@]}"; do
    ref=$(build/seq_check "$n")
    gab=$(campo checksum "$ref")
    echo "n=$n  gabarito=$gab"
    testes=$((testes+1))
    if [ "$(campo ordenado "$ref")" != "SIM" ]; then
        echo "  FALHOU seq_check nao ordenou"; falhas=$((falhas+1))
    fi

    if [ -x build/omp_mergesort ]; then
        for t in "${THREADS[@]}"; do
            out=$(OMP_NUM_THREADS=$t build/omp_mergesort "$n")
            verificar "openmp t=$t" "$n" "$gab" "$out"
        done
    fi
    if [ -x build/mpi_mergesort ]; then
        for p in "${PROCS[@]}"; do
            out=$($MPIRUN $MPIFLAGS -np "$p" build/mpi_mergesort "$n" | grep '^n=' | head -1)
            verificar "mpi np=$p" "$n" "$gab" "$out"
        done
    fi
done

[ -x build/omp_mergesort ] || echo "(aviso) OpenMP ainda nao compilado: nada a validar"
[ -x build/mpi_mergesort ] || echo "(aviso) MPI ainda nao compilado: nada a validar"
echo
echo "Resumo: $((testes-falhas))/$testes verificacoes OK"
[ $falhas -eq 0 ] && echo "VALIDACAO: PASSOU" || { echo "VALIDACAO: FALHOU ($falhas)"; exit 1; }
