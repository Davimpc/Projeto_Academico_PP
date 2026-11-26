# Makefile — Projeto Mergesort (CCO085, IESB 2026/2)
# Ambiente de referencia: Linux, g++ e OpenMPI (mpicxx / mpirun).
#
#   make            compila tudo o que ja existe
#   make seq        baseline do professor + seq_check (gabarito)
#   make openmp     versao OpenMP   (openmp/07_mergesort_omp.cpp)
#   make mpi        versao MPI      (mpi/07_mergesort_mpi.cpp)
#   make validar    compara checksum de todas as versoes com o gabarito
#   make medir      gera os CSVs de desempenho em results/   (demora alguns minutos)
#   make graficos   gera as figuras em graficos/ a partir dos CSVs (precisa de pandas e matplotlib)
#   make clean

CXX      ?= g++
MPICXX   ?= mpicxx
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
BUILD    := build

OMP_SRC := openmp/07_mergesort_omp.cpp
MPI_SRC := mpi/07_mergesort_mpi.cpp

TARGETS := $(BUILD)/baseline_prof $(BUILD)/seq_check
ifneq ($(wildcard $(OMP_SRC)),)
TARGETS += $(BUILD)/omp_mergesort
endif
ifneq ($(wildcard $(MPI_SRC)),)
TARGETS += $(BUILD)/mpi_mergesort
endif

.PHONY: all seq openmp mpi validar medir graficos clean
all: $(TARGETS)

seq: $(BUILD)/baseline_prof $(BUILD)/seq_check
openmp: $(BUILD)/omp_mergesort
mpi: $(BUILD)/mpi_mergesort

$(BUILD):
	mkdir -p $(BUILD)

# Codigo do professor, compilado exatamente como no cabecalho dele (-O2).
$(BUILD)/baseline_prof: seq/07_mergesort.cpp | $(BUILD)
	$(CXX) -O2 -o $@ $<

$(BUILD)/seq_check: seq/07_mergesort_check.cpp common/comum.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BUILD)/omp_mergesort: $(OMP_SRC) common/comum.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $<

$(BUILD)/mpi_mergesort: $(MPI_SRC) common/comum.h | $(BUILD)
	$(MPICXX) $(CXXFLAGS) -DOMPI_SKIP_MPICXX -o $@ $<

# 'bash scripts/...' (e nao ./scripts/...) evita depender do bit de executavel, que se perde
# ao passar pelo Windows.
validar: all
	bash scripts/validar.sh

medir: all
	bash scripts/medir.sh

graficos:
	python3 scripts/graficos.py

clean:
	rm -rf $(BUILD)
