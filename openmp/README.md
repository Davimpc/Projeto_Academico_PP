# Versão OpenMP — a fazer

Arquivo esperado: `07_mergesort_omp.cpp` (o `Makefile` o detecta sozinho).

Contrato:
- `#include "../common/comum.h"`; gerar a entrada com `gerar_entrada(a)`.
- Cronometrar so a ordenacao com `agora()`; imprimir com `imprimir_resultado(...)`.
- Numero de threads vem de `OMP_NUM_THREADS`.
- Ponto de partida: `parallel` + `single` + `task`, `taskwait` antes do merge, com cutoff.
- Registrar no relatorio: por que nao ha corrida, por que critical/atomic/reduction nao se aplicam.
