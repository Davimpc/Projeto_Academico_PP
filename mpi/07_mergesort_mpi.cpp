/*-------------------------------------------------------------------------------------------------------------------------
 * File Name : 07_mergesort_mpi.cpp
 * Course    : Programação Paralela (CCO085) - IESB 2026/2
 * Project   : Projeto Acadêmico - Problema nº 7: Ordenação (mergesort)
 * Objective : Mergesort paralelo em memória DISTRIBUÍDA com MPI (decomposição de domínio + merge).
 * Authors   : ver AUTORES.md
 *
 * Como compilar:
 *   mpicxx -O2 -std=c++17 -o mpi_mergesort mpi/07_mergesort_mpi.cpp      (ou: make mpi)
 *
 * Como executar:
 *   mpirun -np 4 ./build/mpi_mergesort 20000000              (n elementos, combinação em ÁRVORE)
 *   mpirun -np 4 ./build/mpi_mergesort 20000000 gather       (combinação por MPI_Gatherv no rank 0)
 *   mpirun -np 8 --oversubscribe ./build/mpi_mergesort ...   (mais processos que núcleos)
 *
 * Saída em stdout (só o rank 0; a mesma de todas as versões do projeto):
 *   n=<n> ordenado=SIM|NAO tempo=<s> checksum=0x<hex>
 * Detalhes de tempo e hardware vão para stderr, em linhas "[mpi] chave=valor ...".
 *
 *=========================================================================================================================
 * A IDEIA (decomposição de DADOS)
 *=========================================================================================================================
 *   Com threads, todo mundo enxerga o vetor inteiro. Com ranks (processos), cada um tem a SUA memória:
 *   o rank 3 não tem o vetor, só o que receber. Então paralelizar = dividir os DADOS e MANDAR cada
 *   pedaço para quem vai trabalhar nele.
 *
 *     1. DISTRIBUIR : o rank 0 tem o vetor e entrega um bloco a cada rank        (MPI_Scatterv)
 *     2. ORDENAR    : cada rank ordena o seu bloco com o msort do professor      (100% paralelo,
 *                     sem nenhuma comunicação)
 *     3. COMBINAR   : os p blocos ordenados precisam virar UM vetor ordenado     (merge)
 *     4. MEDIR      : o tempo do programa é o do rank MAIS LENTO                 (MPI_Reduce, MPI_MAX)
 *
 *   Analogia: 1.000 provas em ordem alfabética. O professor (rank 0) reparte pilhas iguais a 4
 *   monitores. Cada um ordena a sua pilha. No fim, as 4 pilhas ordenadas precisam ser INTERCALADAS
 *   numa só: é a etapa que o mergesort exige e que não se evita.
 *
 *   Por que Scatterv (com "v") e não Scatter? n quase nunca divide por p. Scatterv aceita um
 *   tamanho diferente por rank. A divisão usa  inicio(r) = r*n/p  (em long long, para não estourar):
 *   os blocos diferem em no máximo 1 elemento, inclusive quando p > n (alguns ranks ficam com 0).
 *
 *=========================================================================================================================
 * AS DUAS FORMAS DE COMBINAR (argumento opcional, para comparar no relatório)
 *=========================================================================================================================
 *   "arvore" (padrão): merge em árvore, em ceil(log2 p) rodadas. Em cada rodada, metade dos ranks que
 *      ainda restam ENVIA o seu bloco ordenado para um vizinho, que faz o merge dos dois. Os merges de
 *      uma mesma rodada acontecem EM PARALELO em ranks diferentes. Funciona para qualquer p (não
 *      precisa ser potência de 2). Ex. p=5:
 *         rodada 1: 1->0   3->2          (4 espera)
 *         rodada 2: 2->0                 (4 espera)
 *         rodada 3: 4->0
 *      Caminho crítico do merge ~ n/2 + n/4 + ... : o último merge (n elementos) é de UM rank só.
 *
 *   "gather": MPI_Gatherv junta os p blocos no rank 0, que faz TODOS os merges sozinho
 *      (em árvore também, mas numa só CPU): custo ~ n*log2(p) no rank 0, sem paralelismo.
 *      Comunicação mais simples (uma coletiva), mas o merge vira gargalo serial.
 *
 *   A comparação das duas mostra o trade-off clássico: menos mensagens x mais trabalho serial.
 *
 *=========================================================================================================================
 * COLETIVAS USADAS: MPI_Scatterv (distribuir), MPI_Reduce com MPI_MAX (tempo), e MPI_Gatherv (modo gather).
 * A combinação em árvore usa MPI_Send/MPI_Recv ponto a ponto, porque cada rodada só envolve PARES de ranks.
 *
 * Corretude: o vetor ordenado é ÚNICO (inteiros iguais são indistinguíveis), então qualquer decomposição
 * correta produz exatamente o mesmo vetor do baseline: o checksum não muda com -np.
 --------------------------------------------------------------------------------------------------------------------------*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include <mpi.h>
#include "../common/comum.h"
using namespace std;

/* ---------------------------------------------------------------------------------------------
 * merge e msort: CÓPIA SEM ALTERAÇÃO do código do professor (a lógica de cálculo não pode mudar).
 * Usados na ordenação local E em todos os merges de combinação.
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int meu_rank, total_processos;
    MPI_Comm_rank(MPI_COMM_WORLD, &meu_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processos);
    const int RANK_ROOT = 0;

    int n = (argc > 1) ? atoi(argv[1]) : 20000000;
    if (n < 0) n = 0;
    const bool modo_gather = (argc > 2 && strcmp(argv[2], "gather") == 0);

    // -----------------------------------------------------------------------------------------
    // Divisão em blocos: TODOS os ranks calculam a tabela inteira (é barato e evita comunicação).
    //   quantidades[r]   = quantos elementos o rank r recebe
    //   deslocamentos[r] = onde o bloco do rank r começa no vetor completo
    // -----------------------------------------------------------------------------------------
    vector<int> quantidades(total_processos), deslocamentos(total_processos);
    for (int r = 0; r < total_processos; r++) {
        int inicio_r = (int)((long long)r * n / total_processos);
        int fim_r    = (int)((long long)(r + 1) * n / total_processos);
        quantidades[r]   = fim_r - inicio_r;
        deslocamentos[r] = inicio_r;
    }
    const int meus_elementos = quantidades[meu_rank];

    // Só o root tem o vetor completo (gerado FORA da região cronometrada, como no baseline).
    vector<int> vetor_completo;
    if (meu_rank == RANK_ROOT) {
        vetor_completo.resize(n);
        gerar_entrada(vetor_completo);
    }

    vector<int> bloco(meus_elementos);   // o pedaço deste rank (ordenado depois)
    vector<int> tmp;                     // área de trabalho do merge

    // Cronômetros deste rank (segundos). Separamos comunicação de cálculo para a análise de overhead.
    double t_scatter = 0, t_local = 0, t_comb_comm = 0, t_comb_merge = 0;

    // ===================================== REGIÃO CRONOMETRADA =====================================
    MPI_Barrier(MPI_COMM_WORLD);         // todos largam juntos
    const double inicio = MPI_Wtime();

    // 1. DISTRIBUIR -------------------------------------------------------------------------------
    //    Em ranks que não são o root, o buffer de envio é ignorado (por isso .data() de vetor vazio).
    double c0 = MPI_Wtime();
    MPI_Scatterv(vetor_completo.data(), quantidades.data(), deslocamentos.data(), MPI_INT,
                 bloco.data(), meus_elementos, MPI_INT, RANK_ROOT, MPI_COMM_WORLD);
    t_scatter = MPI_Wtime() - c0;

    // 2. ORDENAR LOCALMENTE (zero comunicação) ------------------------------------------------------
    c0 = MPI_Wtime();
    tmp.resize(meus_elementos);
    msort(bloco, tmp, 0, meus_elementos);
    t_local = MPI_Wtime() - c0;

    // 3. COMBINAR ---------------------------------------------------------------------------------
    vector<int>* resultado = &bloco;     // onde o rank 0 terá o vetor final ordenado

    if (!modo_gather) {
        // ---- árvore: em cada rodada, ranks múltiplos de 2*passo recebem de (rank + passo) ----
        int tamanho_atual = meus_elementos;   // quantos elementos este rank tem AGORA (cresce ao receber)
        for (int passo = 1; passo < total_processos; passo *= 2) {
            if (meu_rank % (2 * passo) == 0) {
                int parceiro = meu_rank + passo;
                if (parceiro < total_processos) {
                    // Quantos elementos o parceiro vai mandar? Soma dos blocos originais dos ranks
                    // que ele já absorveu: [parceiro, min(parceiro+passo, p)). Todos sabem isso
                    // pela tabela 'quantidades', então não precisa de mensagem extra com o tamanho.
                    int fim_grupo = parceiro + passo < total_processos ? parceiro + passo : total_processos;
                    int recebe = 0;
                    for (int r = parceiro; r < fim_grupo; r++) recebe += quantidades[r];

                    // Recebe DIRETO no fim do bloco: [ meu bloco ordenado | bloco do parceiro ordenado ]
                    // e o merge do professor junta as duas metades (mid = tamanho_atual).
                    bloco.resize(tamanho_atual + recebe);
                    c0 = MPI_Wtime();
                    MPI_Recv(bloco.data() + tamanho_atual, recebe, MPI_INT, parceiro, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    t_comb_comm += MPI_Wtime() - c0;

                    c0 = MPI_Wtime();
                    tmp.resize(tamanho_atual + recebe);
                    merge(bloco, tmp, 0, tamanho_atual, tamanho_atual + recebe);
                    t_comb_merge += MPI_Wtime() - c0;
                    tamanho_atual += recebe;
                }
                // sem parceiro (p não é potência de 2): este rank só espera a próxima rodada
            } else {
                // Este rank é o "parceiro" de alguém: envia tudo o que tem e sai da árvore.
                int destino = meu_rank - passo;
                c0 = MPI_Wtime();
                MPI_Send(bloco.data(), tamanho_atual, MPI_INT, destino, 0, MPI_COMM_WORLD);
                t_comb_comm += MPI_Wtime() - c0;
                break;
            }
        }
    } else {
        // ---- gather: o rank 0 recolhe os p blocos e faz todos os merges sozinho ----
        vector<int> reunido;
        if (meu_rank == RANK_ROOT) reunido.resize(n);
        c0 = MPI_Wtime();
        MPI_Gatherv(bloco.data(), meus_elementos, MPI_INT,
                    reunido.data(), quantidades.data(), deslocamentos.data(), MPI_INT,
                    RANK_ROOT, MPI_COMM_WORLD);
        t_comb_comm += MPI_Wtime() - c0;

        if (meu_rank == RANK_ROOT) {
            // Merge bottom-up sobre as fronteiras dos blocos: junta pares vizinhos até sobrar um.
            c0 = MPI_Wtime();
            tmp.resize(n);
            vector<int> fronteiras(deslocamentos);   // início de cada bloco
            fronteiras.push_back(n);                 // e o fim do último
            while (fronteiras.size() > 2) {
                vector<int> proximas;
                size_t i = 0;
                for (; i + 2 < fronteiras.size(); i += 2) {
                    merge(reunido, tmp, fronteiras[i], fronteiras[i + 1], fronteiras[i + 2]);
                    proximas.push_back(fronteiras[i]);
                }
                if (i + 1 < fronteiras.size()) {     // sobrou um bloco sem par: passa adiante
                    proximas.push_back(fronteiras[i]);
                }
                proximas.push_back(n);
                fronteiras.swap(proximas);
            }
            t_comb_merge += MPI_Wtime() - c0;
            bloco.swap(reunido);                     // o resultado final fica em 'bloco' no rank 0
        }
    }

    const double meu_total = MPI_Wtime() - inicio;
    // ================================= FIM DA REGIÃO CRONOMETRADA =================================

    // 4. MEDIR: o tempo do programa é o do rank mais LENTO (na prática, o rank 0, que termina por último).
    double meus_tempos[5] = { meu_total, t_scatter, t_local, t_comb_comm, t_comb_merge };
    double piores[5];
    MPI_Reduce(meus_tempos, piores, 5, MPI_DOUBLE, MPI_MAX, RANK_ROOT, MPI_COMM_WORLD);

    if (meu_rank == RANK_ROOT) {
        // Verificação FORA do tempo, como no baseline. O resultado completo está em 'bloco' (n elementos).
        bool ok = ((int)resultado->size() == n) && esta_ordenado(resultado->data(), n);
        imprimir_resultado(n, ok, piores[0], checksum(resultado->data(), n));

        char nome_maquina[MPI_MAX_PROCESSOR_NAME]; int tamanho_nome;
        MPI_Get_processor_name(nome_maquina, &tamanho_nome);
        fprintf(stderr, "[mpi] p=%d modo=%s maquina=%s nucleos_logicos=%u "
                        "t_total=%.6f t_scatter=%.6f t_local=%.6f t_comb_comm=%.6f t_comb_merge=%.6f\n",
                total_processos, modo_gather ? "gather" : "arvore", nome_maquina,
                std::thread::hardware_concurrency(),
                piores[0], piores[1], piores[2], piores[3], piores[4]);
    }

    MPI_Finalize();
    return 0;
}

/*
 Como ler o resultado:
  - Speedup = T_baseline / T_mpi(p). Eficiência = Speedup / p. T_baseline = build/baseline_prof, mesmo n.
  - Cada MAX é de um rank possivelmente diferente; a soma das parcelas pode passar de t_total.
    Ex.: t_local é o do rank com o maior bloco; t_comb_merge, o do rank 0 (que faz o último merge).
  - t_comb_comm no modo árvore MISTURA transferência e ESPERA: um rank que chama Recv antes de o parceiro
    terminar de ordenar fica parado ali. Parte desse tempo é, portanto, desbalanceamento, não rede.
  - O que limita o speedup: (1) o merge final de n elementos é feito por UM rank (funil), (2) o Scatterv
    e as mensagens do merge MOVEM dados (4 bytes por elemento), e essa comunicação cresce com p,
    (3) dentro de uma só máquina a "rede" é a memória: os ranks disputam a mesma largura de banda,
    (4) com -np maior que o número de núcleos (--oversubscribe) os ranks competem por CPU e o tempo
    piora, então essas execuções NÃO devem entrar na curva de speedup.
*/
