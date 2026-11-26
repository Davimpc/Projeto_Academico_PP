/*-------------------------------------------------------------------------------------------------------------------------
 * File Name : 07_mergesort_omp.cpp
 * Course    : Programação Paralela (CCO085) - IESB 2026/2
 * Project   : Projeto Acadêmico - Problema nº 7: Ordenação (mergesort)
 * Objective : Mergesort paralelo em memória compartilhada com OpenMP TASKS.
 * Authors   : ver AUTORES.md
 *
 * Como compilar (Linux / WSL):
 *   g++ -O2 -std=c++17 -fopenmp -o omp_mergesort openmp/07_mergesort_omp.cpp
 *   (ou simplesmente: make openmp)
 *
 * Como executar:
 *   OMP_NUM_THREADS=4 ./build/omp_mergesort 20000000          (n elementos)
 *   OMP_NUM_THREADS=4 ./build/omp_mergesort 20000000 16384    (n e cutoff)
 *
 * Saída (a mesma de todas as versões do projeto):
 *   n=<n> ordenado=SIM|NAO tempo=<s> checksum=0x<hex>
 *
 *=========================================================================================================================
 * A IDEIA: DIVISÃO E CONQUISTA JÁ É UM PARALELISMO ESPERANDO ACONTECER
 *=========================================================================================================================
 *   O mergesort do professor faz:   msort(esquerda);  msort(direita);  merge(esquerda, direita);
 *
 *   As duas chamadas recursivas trabalham em METADES DIFERENTES do vetor: não se tocam. Então podem
 *   rodar ao mesmo tempo. O merge, porém, precisa das DUAS metades prontas: só começa depois delas.
 *
 *   Analogia: um chefe tem uma pilha de 1.000 provas para colocar em ordem alfabética. Ele divide a
 *   pilha em duas e entrega uma metade a cada auxiliar (cada auxiliar pode dividir a sua de novo).
 *   Quando os dois terminam, o chefe INTERCALA as duas pilhas já ordenadas. Ele não pode intercalar
 *   antes de os auxiliares terminarem: isso é o TASKWAIT.
 *
 *=========================================================================================================================
 * POR QUE TASK E NÃO "parallel for" NEM "sections"?
 *=========================================================================================================================
 *   - parallel for: precisa de um laço com número de iterações conhecido. A recursão não é um laço.
 *   - sections:     o número de blocos é FIXO e escrito no código. Na recursão, quantas tarefas existem
 *                   só se sabe em tempo de execução (cada chamada cria novas).
 *   - task:         cada chamada recursiva vira uma "tarefa" que vai para uma FILA; as threads livres
 *                   pegam tarefas da fila. Serve para trabalho que se descobre dinamicamente.
 *
 *   Estrutura:   #pragma omp parallel   -> cria o time de threads
 *                #pragma omp single     -> UMA thread começa a recursão (se todas começassem, a
 *                                          ordenação seria repetida T vezes)
 *                #pragma omp task       -> "esta chamada pode ser feita por qualquer thread livre"
 *                #pragma omp taskwait   -> "espere as tarefas filhas terminarem antes de seguir"
 *
 *=========================================================================================================================
 * CUTOFF (corte): O QUE FAZ A DIFERENÇA ENTRE ACELERAR E DESACELERAR
 *=========================================================================================================================
 *   Criar uma task custa tempo (alocar, colocar na fila, escalonar). Se o pedaço for pequeno (ex.: 8
 *   elementos), o custo de criar a task é MAIOR que o de ordenar o pedaço. Por isso, abaixo de um
 *   tamanho mínimo (o cutoff) paramos de criar tasks e ordenamos com o msort sequencial.
 *
 *   Cutoff pequeno demais -> tasks demais, overhead domina.
 *   Cutoff grande demais  -> poucas tasks, threads ficam ociosas (desbalanceamento).
 *   O padrão (16384) é um ponto de partida; o segundo argumento do programa permite varrer o valor
 *   e fazer o estudo de sensibilidade no relatório.
 *
 *=========================================================================================================================
 * CONDIÇÕES DE CORRIDA: POR QUE NÃO HÁ CRITICAL, ATOMIC NEM REDUCTION
 *=========================================================================================================================
 *   Uma corrida exige duas threads acessando o MESMO endereço, com pelo menos uma escrita, sem ordem
 *   definida entre elas. Aqui isso não acontece:
 *
 *   1. As duas tasks filhas escrevem em INTERVALOS DISJUNTOS: [lo, mid) e [mid, hi), tanto em 'a'
 *      quanto em 'tmp'. Nenhum endereço é escrito por duas tasks.
 *   2. O merge de um nó só roda depois do 'taskwait', que garante que as duas filhas terminaram
 *      (e o OpenMP garante que as escritas delas ficam visíveis para quem passou pelo taskwait).
 *   3. Não existe nenhuma variável escalar compartilhada sendo atualizada (nenhum contador, soma ou
 *      acumulador). Por isso:
 *        - reduction: serve para combinar um valor acumulado por todas as threads (ex.: uma soma).
 *                     Não há valor acumulado aqui.
 *        - atomic:    protege UMA atualização simples de uma variável compartilhada. Não há.
 *        - critical:  protege um bloco inteiro contra execução simultânea. Seria necessário só se
 *                     houvesse região compartilhada; usá-lo no merge serializaria tudo e mataria o
 *                     ganho de desempenho.
 *   A sincronização correta aqui é a de DEPENDÊNCIA (taskwait), não a de exclusão mútua.
 *
 *   Falsos compartilhamentos (false sharing): só poderia ocorrer nas bordas entre blocos vizinhos
 *   (uma linha de cache compartilhada por duas threads). Com cutoff de milhares de elementos, as
 *   bordas são uma fração desprezível do trabalho.
 *
 *=========================================================================================================================
 * O QUE LIMITA O SPEEDUP (para a discussão com Amdahl)
 *=========================================================================================================================
 *   O merge do topo (que junta as duas metades de n/2 elementos) é feito por UMA thread só, o do
 *   nível abaixo por no máximo duas, o seguinte por no máximo quatro, e assim por diante. Esse
 *   "funil" no fim da recursão limita o ganho, mesmo com infinitas threads:
 *
 *     - TRABALHO total (T1)      ~ n * log2(n)        (log2(n) níveis, cada um move n elementos)
 *     - CAMINHO CRÍTICO (T_inf)  ~ n + n/2 + n/4 + ... ~ 2n   (um merge por nível, cada vez menor,
 *                                                       mesmo que todos os merges de um nível
 *                                                       rodem em paralelo)
 *     - Speedup máximo ~ T1 / T_inf ~ log2(n) / 2      (para n = 20 milhões: log2(n) ~ 24, ou seja
 *                                                       em torno de 12, e menos na prática)
 *
 *   Esse é um limite parecido com o de Amdahl, mas que vem da ESTRUTURA do algoritmo e não de um
 *   trecho serial isolado. Soma-se a ele a pressão sobre a memória: o merge só lê e escreve vetores
 *   grandes (poucas contas por byte movido), então a largura de banda satura antes dos núcleos.
 *   Um merge paralelo (particionando os dois vetores) eliminaria o funil, mas não faz parte desta
 *   versão.
 --------------------------------------------------------------------------------------------------------------------------*/

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include "../common/comum.h"
using namespace std;

/* ---------------------------------------------------------------------------------------------
 * merge e msort: CÓPIA SEM ALTERAÇÃO do código do professor (a lógica de cálculo não pode mudar).
 * O msort sequencial é usado nas folhas da recursão, abaixo do cutoff.
 * --------------------------------------------------------------------------------------------- */
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

/* ---------------------------------------------------------------------------------------------
 * msort_paralelo: mesma estrutura do msort, mas as duas chamadas recursivas viram TASKS
 * enquanto o pedaço for maior que o cutoff.
 * --------------------------------------------------------------------------------------------- */
static void msort_paralelo(vector<int>& a, vector<int>& tmp, int lo, int hi, int cutoff) {
    if (hi - lo <= 1) return;

    // Pedaço pequeno: não vale criar task. Ordena sequencialmente, na própria thread.
    if (hi - lo <= cutoff) {
        msort(a, tmp, lo, hi);
        return;
    }

    int mid = lo + (hi - lo) / 2;   // mesmo ponto médio do professor, sem risco de estouro de int

    // Metade esquerda vira uma TASK: qualquer thread livre pode pegá-la.
    // shared(a, tmp): as tasks usam os MESMOS vetores (memória compartilhada); cada uma escreve
    // só no seu intervalo, por isso não há corrida.
    // lo, mid, cutoff são copiados para a task (firstprivate, o padrão para variáveis locais).
    #pragma omp task shared(a, tmp)
    msort_paralelo(a, tmp, lo, mid, cutoff);

    // Metade direita: a thread atual a faz diretamente. Criar uma segunda task só adicionaria
    // overhead, porque a thread atual ficaria parada esperando de qualquer jeito.
    msort_paralelo(a, tmp, mid, hi, cutoff);

    // Barreira de DEPENDÊNCIA: o merge precisa das duas metades prontas.
    // taskwait espera as tasks FILHAS (aqui, a da esquerda) e, enquanto espera, a thread continua
    // útil executando outras tasks da fila.
    #pragma omp taskwait

    merge(a, tmp, lo, mid, hi);
}

int main(int argc, char** argv) {
    int n      = (argc > 1) ? atoi(argv[1]) : 20000000;
    int cutoff = (argc > 2) ? atoi(argv[2]) : 16384;
    if (n < 0) n = 0;
    if (cutoff < 1) cutoff = 1;

    vector<int> a(n), tmp(n);
    gerar_entrada(a);   // igual à do professor; fica FORA da região cronometrada

    int threads_usadas = 1;

    // ---------------------- região cronometrada: só a ordenação, como no baseline ----------------------
    double t0 = agora();

    #pragma omp parallel
    {
        // single: UMA thread inicia a recursão e cria as tasks; as demais ficam à espera de tasks
        // na fila (ao fim do single há uma barreira implícita, que também só libera quando todas
        // as tasks terminaram).
        #pragma omp single
        {
            threads_usadas = omp_get_num_threads();
            msort_paralelo(a, tmp, 0, n, cutoff);
        }
    }

    double t1 = agora();
    // -------------------------------------------------------------------------------------------------

    // Informações de apoio vão para stderr, para a linha de resultado (stdout) ficar idêntica
    // à das outras versões e os scripts continuarem lendo só ela.
    fprintf(stderr, "[omp] threads=%d cutoff=%d procs_logicos=%d\n",
            threads_usadas, cutoff, omp_get_num_procs());

    imprimir_resultado(n, esta_ordenado(a.data(), n), t1 - t0, checksum(a.data(), n));
    return 0;
}

/*
 Como ler o resultado:
  - Speedup = T_baseline / T_omp(p threads). Eficiência = Speedup / p.
  - Use como T_baseline o tempo do código do professor (build/baseline_prof) para o MESMO n.
  - O speedup não chega a p porque: (1) os merges do topo da recursão têm pouco ou nenhum paralelismo;
    (2) o merge move muitos dados e poucas contas, então a largura de banda da memória satura antes
    dos núcleos; (3) criar/escalonar tasks tem custo; (4) com 1 thread o programa ainda paga o custo
    de criar a região paralela, então T_omp(1) é um pouco maior que T_baseline.
  - Rode várias vezes e use a mediana; os tempos variam entre execuções.
*/
