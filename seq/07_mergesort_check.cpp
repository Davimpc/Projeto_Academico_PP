/* seq_check: mesma logica de calculo do baseline do professor
 * (merge e msort copiados sem alteracao). Diferencas, todas FORA da regiao
 * cronometrada: usa o gerador comum e imprime o checksum do vetor ordenado,
 * que serve de GABARITO para validar as versoes OpenMP e MPI.
 *
 * Compilar: make seq        Executar: ./build/seq_check 20000000
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../common/comum.h"
using namespace std;

static void merge(vector<int>& a, vector<int>& tmp, int lo, int mid, int hi) {
    int i = lo, j = mid, k = lo;
    while (i < mid && j < hi) tmp[k++] = (a[i] <= a[j]) ? a[i++] : a[j++];
    while (i < mid) tmp[k++] = a[i++];
    while (j < hi)  tmp[k++] = a[j++];
    for (int t = lo; t < hi; t++) a[t] = tmp[t];
}
static void msort(vector<int>& a, vector<int>& tmp, int lo, int hi) {
    if (hi - lo <= 1) return;
    int mid = (lo + hi) / 2;
    msort(a, tmp, lo, mid);
    msort(a, tmp, mid, hi);
    merge(a, tmp, lo, mid, hi);
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 20000000;
    vector<int> a(n), tmp(n);
    gerar_entrada(a);

    double t0 = agora();
    msort(a, tmp, 0, n);
    double t1 = agora();

    imprimir_resultado(n, esta_ordenado(a.data(), n), t1 - t0, checksum(a.data(), n));
    return 0;
}
