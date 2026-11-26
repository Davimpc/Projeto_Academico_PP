/* Utilitarios compartilhados pelas tres versoes (seq_check, OpenMP, MPI).
 *
 * gerar_entrada(): copia FIEL do gerador do codigo do professor
 *   (LCG, semente 12345). Nao alterar: a entrada define o resultado correto.
 * checksum(): resumo do vetor, usado para provar que todas as versoes
 *   produzem exatamente o mesmo vetor ordenado.
 */
#ifndef COMUM_H
#define COMUM_H
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <vector>

static inline double agora() {
    timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Mesma sequencia do baseline: a[i] depende do estado anterior do gerador,
 * entao a geracao e inerentemente sequencial. */
static inline void gerar_entrada(std::vector<int>& a) {
    uint64_t s = 12345;
    for (size_t i = 0; i < a.size(); i++) {
        s = s * 6364136223846793005ULL + 1;
        a[i] = (int)(s >> 33);
    }
}

/* FNV-1a de 64 bits sobre os bytes de cada elemento, na ordem do vetor.
 * Sensivel a ordem e ao conteudo: detecta elemento perdido, duplicado,
 * trocado ou fora de posicao. */
static inline uint64_t checksum(const int* a, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        uint32_t v = (uint32_t)a[i];
        for (int b = 0; b < 4; b++) {
            h ^= (v >> (8 * b)) & 0xFF;
            h *= 1099511628211ULL;
        }
    }
    return h;
}

static inline bool esta_ordenado(const int* a, size_t n) {
    for (size_t i = 1; i < n; i++) if (a[i - 1] > a[i]) return false;
    return true;
}

/* Linha de saida padrao, igual em todas as versoes (facilita os scripts):
 *   n=<n> ordenado=SIM|NAO tempo=<s> checksum=0x<hex> */
static inline void imprimir_resultado(size_t n, bool ok, double tempo, uint64_t h) {
    printf("n=%zu ordenado=%s tempo=%.6f checksum=0x%016llx\n",
           n, ok ? "SIM" : "NAO", tempo, (unsigned long long)h);
}
#endif
