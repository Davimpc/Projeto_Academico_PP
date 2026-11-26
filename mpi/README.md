# Versão MPI — a fazer

Arquivo esperado: `07_mergesort_mpi.cpp` (o `Makefile` o detecta sozinho).

Contrato:
- `#include "../common/comum.h"`; o rank 0 gera a entrada com `gerar_entrada(a)`.
- `MPI_Scatterv` (n pode nao dividir por p), ordenacao local, combinacao dos resultados.
- Tempo = distribuir + ordenar + recolher, maximo entre ranks (`MPI_Reduce` com `MPI_MAX`).
- Somente o rank 0 imprime a linha de resultado com `imprimir_resultado(...)`.
- Resultado identico para qualquer `-np`, inclusive np > n e n nao divisivel por p.
