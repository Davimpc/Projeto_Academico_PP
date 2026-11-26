# Mergesort paralelo — CCO085 (IESB 2026/2)

Paralelização do mergesort fornecido pelo professor (problema nº 7) com
OpenMP (tasks) e MPI (Scatterv + merge). Autores em `AUTORES.md`.

## Estrutura

```
seq/      codigo do professor (INTACTO) e seq_check (gabarito com checksum)
openmp/   07_mergesort_omp.cpp   versao OpenMP (tasks com cutoff)
mpi/      07_mergesort_mpi.cpp   versao MPI (Scatterv + ordenacao local + merge em arvore ou Gatherv)
common/   comum.h: gerador de entrada identico ao do professor, checksum, saida padrao
scripts/  validar.sh (corretude), medir.sh (gera os CSVs), graficos.py (figuras e tabela-resumo)
results/  CSVs gerados por medir.sh
graficos/ figuras para o relatorio
```

## Como rodar do zero (ambiente oficial da disciplina)

Requisito: Docker. Na pasta do ambiente do professor (`ambiente_pp`), com este projeto em
`codigos/projeto-mergesort`:

```bash
docker compose build              # so na primeira vez
docker compose run --rm pp-lab    # abre um shell em /lab
cd codigos/projeto-mergesort      # ATENCAO: o make da raiz (/lab) e o do professor, nao o deste projeto

make                              # compila baseline, seq_check, OpenMP e MPI
make validar                      # compara o checksum de TODAS as versoes com o gabarito
make medir                        # gera results/*.csv (alguns minutos; use uma maquina com varios nucleos)
make graficos                     # gera graficos/*.png e results/resumo.csv
```

Sem Docker: Linux com `g++`, OpenMPI (`mpicxx`, `mpirun`), `make`, e (para os graficos)
`python3` com `pandas` e `matplotlib`. O contêiner do professor nao traz pandas/matplotlib:
gere os graficos fora dele (`python3 scripts/graficos.py` na pasta do projeto, no seu computador)
ou instale com `pip install --break-system-packages pandas matplotlib`.

Execucao manual:

```bash
./build/baseline_prof 20000000                       # codigo do professor
OMP_NUM_THREADS=4 ./build/omp_mergesort 20000000     # OpenMP (2o argumento opcional: cutoff)
mpirun -np 4 ./build/mpi_mergesort 20000000          # MPI, merge em arvore
mpirun -np 4 ./build/mpi_mergesort 20000000 gather   # MPI, Gatherv + merge no rank 0
```

## Regra de ouro: o resultado nao pode mudar

- `seq/07_mergesort.cpp` **nao pode ser alterado** (regra do professor).
- Toda versao usa `common/comum.h` (mesmo gerador, mesmo checksum) e imprime em stdout:
  `n=<n> ordenado=SIM|NAO tempo=<s> checksum=0x<hex>`
- `scripts/validar.sh` exige `ordenado=SIM` **e** checksum igual ao do `seq_check`
  para OpenMP com 1/2/4/8 threads e MPI com -np 1/2/3/4/5/8, incluindo n pequenos
  (1, 2, 3, 7) e n que nao divide por p (99991, 1000003).
- O checksum e necessario: a verificacao original so testa a ordenacao e nao
  detecta elemento perdido ou duplicado.
- `scripts/medir.sh` tambem confere o checksum de **cada** execucao medida e aborta se divergir.

## Medicao (scripts/medir.sh)

- Cronometra so a regiao de ordenacao, como o baseline (geracao e verificacao ficam fora).
  No MPI o tempo inclui distribuir, ordenar e combinar, e e o maximo entre os ranks.
- `REP` execucoes por configuracao (padrao 6); a **primeira e descartada** (aquecimento).
- Faixas de threads/processos: potencias de 2 ate o numero de nucleos logicos (`nproc`) e o proprio
  `nproc`. **Nao se mede acima de `nproc`**: oversubscribe nao entra na curva de speedup.
- Variaveis de ambiente: `TAMANHOS`, `REP`, `N_BASE` (escalabilidade fraca), `CUTOFFS`, `NUCLEOS`.
  Experimentos isolados: `bash scripts/medir.sh forte` (ou `baseline`, `fraca`, `cutoff`).
- Formato dos CSVs: as 3 primeiras colunas seguem o script do professor
  (`threads|processos, repeticao, tempo_s`); as demais (n, cutoff/modo, tempos por fase, nucleos, cpu)
  vem no fim. O CPU e o numero de nucleos ficam registrados em cada linha.

| Arquivo | Conteudo |
|---|---|
| `baseline.csv` | codigo do professor, por n |
| `openmp_escalabilidade.csv` | escalabilidade forte OpenMP |
| `mpi_escalabilidade.csv` | escalabilidade forte MPI (arvore e gather), com tempo por fase |
| `openmp_fraca.csv`, `mpi_fraca.csv` | escalabilidade fraca (n = `N_BASE` * p) |
| `openmp_cutoff.csv` | sensibilidade ao cutoff |
| `resumo.csv` | mediana, speedup, eficiencia e Karp-Flatt (gerado por `graficos.py`) |


