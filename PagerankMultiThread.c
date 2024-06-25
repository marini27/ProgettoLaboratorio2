#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <semaphore.h>
#include <fcntl.h>  // For O_* constants
#include <sys/mman.h>  // For shm_open, shm_unlink
#include <unistd.h>  // For ftruncate
#include <bits/mman-linux.h>
#include <time.h>
#include <math.h>
#include <getopt.h>
#include <signal.h>

typedef struct Node {
    int dest;
    struct Node* next;
} Node;

typedef struct {
    int N;
    int *out;
    Node **in;
    double *ranks;
    double *old_ranks;
    double damping;
    double epsilon;
    int maxiter;
    pthread_mutex_t lock;
} grafo;

typedef struct {
    int src;
    int dest;
} Arco;

typedef struct {
    Arco *buffer;
    int size;
    int in;
    int out;
    sem_t empty;
    sem_t full;
    pthread_mutex_t mutex;
} SharedBuffer;

typedef struct {
    pthread_t *aux_threads;
    int num_aux_threads;
    SharedBuffer *task_queue;
    grafo *g;
} ThreadPool;

typedef struct {
    int num_threads;
    grafo *g;
    SharedBuffer *buffer;
    char *filename;
    ThreadPool *pool;
} ThreadData;

typedef struct HashNode {
    int src;
    int dest;
    struct HashNode *next;
} HashNode;

typedef struct {
    int size;
    HashNode **table;
} HashSet;

typedef struct {
    double *ranks;
    int iterations;
    double delta;
} PageRankResult;

typedef struct {
    grafo *g;
    double *ranks;
    double *old_ranks;
    double damping;
    double *sum_dead_ends;
    int start;
    int end;
    double *local_deltas;
    int thread_id;
    int *iterations;
    double epsilon;
    int maxiter;
    pthread_barrier_t *barrier;
    int *stop_flag;
    int num_threads;
    double *delta;
} PagerankThreadData;

typedef struct {
    int *iterations;
    double *ranks;
    int num_nodes;
    pthread_mutex_t *lock;
} SignalThreadData;

volatile sig_atomic_t print_info = 0;

// Funzione di confronto per qsort
int compare(const void *a, const void *b, void *ranks) {
    int idx_a = *(int *)a;
    int idx_b = *(int *)b;
    double rank_a = ((double *)ranks)[idx_a];
    double rank_b = ((double *)ranks)[idx_b];
    return (rank_b > rank_a) - (rank_b < rank_a);
}

// Funzioni per la gestione dell'HashSet
HashSet *createHashSet(int size) {
    HashSet *set = (HashSet *)malloc(sizeof(HashSet));
    set->size = size;
    set->table = (HashNode **)malloc(size * sizeof(HashNode *));
    for (int i = 0; i < size; i++) {
        set->table[i] = NULL;
    }
    return set;
}

int hashFunction(int src, int dest, int size) {
    return (src * 31 + dest) % size;
}

int hashSetContains(HashSet *set, int src, int dest) {
    int hash = hashFunction(src, dest, set->size);
    HashNode *node = set->table[hash];
    while (node != NULL) {
        if (node->src == src && node->dest == dest) {
            return 1;
        }
        node = node->next;
    }
    return 0;
}

void hashSetAdd(HashSet *set, int src, int dest) {
    int hash = hashFunction(src, dest, set->size);
    HashNode *newNode = (HashNode *)malloc(sizeof(HashNode));
    newNode->src = src;
    newNode->dest = dest;
    newNode->next = set->table[hash];
    set->table[hash] = newNode;
}

void freeHashSet(HashSet *set) {
    for (int i = 0; i < set->size; i++) {
        HashNode *node = set->table[i];
        while (node != NULL) {
            HashNode *temp = node;
            node = node->next;
            free(temp);
        }
    }
    free(set->table);
    free(set);
}

//inizializzo il buffer che utilizza la memoria condivisa
void initSharedBuffer(SharedBuffer *sharedBuffer, int size) {
    sharedBuffer->size = size;
    sharedBuffer->in = 0;
    sharedBuffer->out = 0;

    sharedBuffer->buffer = mmap(NULL, size * sizeof(Arco), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (sharedBuffer->buffer == MAP_FAILED) {
        perror("mmap failed");
        exit(EXIT_FAILURE);
    }

    sem_init(&sharedBuffer->empty, 1, size);
    sem_init(&sharedBuffer->full, 1, 0);
    pthread_mutex_init(&sharedBuffer->mutex, NULL);
    printf("Shared buffer initialized with size %d\n", size);
}

//distruggo il buffer condiviso
void destroySharedBuffer(SharedBuffer *sharedBuffer) {
    munmap(sharedBuffer->buffer, sharedBuffer->size * sizeof(Arco));
    sem_destroy(&sharedBuffer->empty);
    sem_destroy(&sharedBuffer->full);
    pthread_mutex_destroy(&sharedBuffer->mutex);
    printf("Shared buffer destroyed\n");
}

//funzione utilizzata dal thread capolettore per inserire un arco nel buffer condiviso
void produce(ThreadPool *pool, Arco arco) {
    SharedBuffer *sharedBuffer = pool->task_queue;
    sem_wait(&sharedBuffer->empty);
    pthread_mutex_lock(&sharedBuffer->mutex);

    sharedBuffer->buffer[sharedBuffer->in] = arco;
    sharedBuffer->in = (sharedBuffer->in + 1) % sharedBuffer->size;

    pthread_mutex_unlock(&sharedBuffer->mutex);
    sem_post(&sharedBuffer->full);
    //printf("Produced arc (%d, %d) by capolettore thread: %ld\n", arco.src, arco.dest, pthread_self());
}

//funzione utilizzata dai thread ausiliari per consumare un arco dal buffer condiviso
Arco consume(ThreadPool *pool) {
    SharedBuffer *sharedBuffer = pool->task_queue;
    sem_wait(&sharedBuffer->full);
    pthread_mutex_lock(&sharedBuffer->mutex);

    Arco arco = sharedBuffer->buffer[sharedBuffer->out];
    sharedBuffer->out = (sharedBuffer->out + 1) % sharedBuffer->size;

    pthread_mutex_unlock(&sharedBuffer->mutex);
    sem_post(&sharedBuffer->empty);
    return arco;
}

//funzione eseguita dai thread ausiliari per consumare gli archi dal buffer condiviso e aggiungerli al grafo
void *thread_function(void *arg) {
    ThreadPool *pool = (ThreadPool *)arg;
    while (1) {
        Arco arco = consume(pool);
        if (arco.src == -1 && arco.dest == -1) {
            break; // Termina il thread
        }
        //printf("Thread %ld: processed arc (%d, %d)\n", pthread_self(), arco.src, arco.dest);

        if (arco.src != arco.dest) {
            pthread_mutex_lock(&pool->g->lock);
            addEdge(pool->g, arco.src, arco.dest);
            pthread_mutex_unlock(&pool->g->lock);
        }
        usleep(1);
    }
    return NULL;
}

//inizializzo il thread pool
void threadPoolInit(ThreadPool *pool, int num_aux_threads, SharedBuffer *task_queue, grafo *g) {
    pool->num_aux_threads = num_aux_threads;
    pool->aux_threads = (pthread_t *)malloc(num_aux_threads * sizeof(pthread_t));
    pool->task_queue = task_queue;
    pool->g = g;
    pthread_mutex_init(&pool->g->lock, NULL);
    printf("Thread pool initialized with %d threads\n", num_aux_threads);
}

//distruggo il thread pool
void threadPoolDestroy(ThreadPool *pool) {

    for (int i = 0; i < pool->num_aux_threads; i++) {
        pthread_join(pool->aux_threads[i], NULL);
    }
    
    free(pool->aux_threads);
    pthread_mutex_destroy(&pool->g->lock);
    printf("Thread pool destroyed\n");
}

//funzione eseguita dal thread capolettore per leggere il file e inserire gli archi nel buffer condiviso
void *funzione_capolettore(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    grafo *g = data->g;
    ThreadPool *pool = data->pool;
    SharedBuffer *task_queue = pool->task_queue;

    FILE *file = fopen(data->filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Errore nell'apertura del file %s\n", data->filename);
        exit(1);
    }

    char line[1024];
    while (fgets(line, sizeof(line), file) != NULL && line[0] == '%');

    int r, c, n;
    sscanf(line, "%d %d %d", &r, &c, &n);
    if (r != c) {
        perror("Error opening file");
        fprintf(stderr, "Errore: Matrice non quadrata\n");
        exit(1);
    }

    int N = r;
    g->N = N;
    g->out = (int *)calloc(N, sizeof(int));
    g->in = (Node **)malloc(N * sizeof(Node *));
    for (int i = 0; i < N; i++) {
        g->in[i] = NULL;
    }

    HashSet *set = createHashSet(N);

    for (int i = 0; i < n; i++) {
        int src, dest;
        fscanf(file, "%d %d", &src, &dest);
        src--;
        dest--;
        if (src < 0 || src >= N || dest < 0 || dest >= N || src == dest) {
            continue;
        }
        if (hashSetContains(set, src, dest)) {
            continue;
        }
        hashSetAdd(set, src, dest);
        Arco arco = { src, dest };
        produce(pool, arco);
    }

    freeHashSet(set);

    for (int i = 0; i < data->num_threads; i++) {
        Arco terminate_signal = { -1, -1 };
        produce(pool, terminate_signal);
    }
    fclose(file);
    printf("File letto\n");
}

//funzione chiamata da threadfunction per aggiungere un arco al grafo
void addEdge(grafo *g, int src, int dest) {
    Node *newNode = (Node *)malloc(sizeof(Node));
    newNode->dest = src;
    newNode->next = g->in[dest];
    g->in[dest] = newNode;
    g->out[src]++;
    //printf("Edge added: %d -> %d\n", src, dest);
}

//funzione per liberare la memoria allocata per il grafo
void freeGraph(grafo *g) {
    for (int i = 0; i < g->N; i++) {
        Node *current = g->in[i];
        while (current != NULL) {
            Node *temp = current;
            current = current->next;
            free(temp);
        }
    }
    free(g->in);
    free(g->out);
    free(g);
    printf("Graph memory freed\n");
}

//funzione eseguita dal thread che gestisce i segnali per stampare le informazioni sul pagerank attuale, dopo aver ricevuto SIGUSR1
void *signal_thread(void *arg) {
    SignalThreadData *data = (SignalThreadData *)arg;
    int sig;
    sigset_t set;

    sigemptyset(&set);
    sigaddset(&set, SIGUSR1);
    fprintf(stderr, "Signal thread started, il mio pid è: \n, %d", getpid());

    while (1) {
        sigwait(&set, &sig);
        fprintf(stderr, "Segnale ricevuto = %d\n", sig);

        if (sig == SIGUSR1) {
            pthread_mutex_lock(data->lock);

            int max_index = 0;
            double max_rank = data->ranks[0];
            for (int i = 1; i < data->num_nodes; i++) {
                if (data->ranks[i] > max_rank) {
                    max_rank = data->ranks[i];
                    max_index = i;
                }
            }

            fprintf(stderr, "Iterazione corrente: %d\n", *data->iterations);
            fprintf(stderr, "Nodo con il maggiore PageRank: %d\n", max_index);
            fprintf(stderr, "Valore del PageRank: %f\n", max_rank);

            pthread_mutex_unlock(data->lock);
        }
    }

    return NULL;
}

//funzione chiamata da pagerank_parallel per calcolare il pagerank in parallelo
void *pagerank_thread(void *arg) {
    PagerankThreadData *data = (PagerankThreadData *)arg;
    grafo *g = data->g;
    double *ranks = data->ranks;
    double *old_ranks = data->old_ranks;
    double damping = data->damping;
    int start = data->start;
    int end = data->end;
    double epsilon = data->epsilon;
    int maxiter = data->maxiter;
    double *local_deltas = data->local_deltas;
    int thread_id = data->thread_id;
    pthread_barrier_t *barrier = data->barrier;
    int *stop_flag = data->stop_flag;
    int *iterations = data->iterations;
    int num_threads = data->num_threads;
    double *sum_dead_ends = data->sum_dead_ends;
    double *delta = data->delta;

    while (1) {
        pthread_barrier_wait(barrier);

        if (*iterations >= maxiter || *stop_flag) {
            pthread_exit(NULL);
        }

        double local_delta = 0.0;
        for (int i = start; i < end; i++) {
            double rank_sum = 0.0;
            Node* current = g->in[i];
            while (current != NULL) {
                rank_sum += old_ranks[current->dest] / g->out[current->dest];
                current = current->next;
            }
            double new_rank = ((1.0 - damping) / g->N) + (damping * (rank_sum + *sum_dead_ends / g->N));
            ranks[i] = new_rank;
            local_delta += fabs(ranks[i] - old_ranks[i]);
        }
        local_deltas[thread_id] = local_delta;

        pthread_barrier_wait(barrier);

        if (thread_id == 0) {
            *delta = 0.0;
            for (int i = 0; i < num_threads; i++) {
                *delta += local_deltas[i];
            }
            if (*delta <= epsilon) {
                *stop_flag = 1;
            }
            (*iterations)++;
            memcpy(old_ranks, ranks, g->N * sizeof(double));

            *sum_dead_ends = 0.0;
            for (int i = 0; i < g->N; i++) {
                if (g->out[i] == 0) {
                    *sum_dead_ends += old_ranks[i];
                }
            }
        }

        pthread_barrier_wait(barrier);
    }

    return NULL;
}

//funzione che divide il lavoro tra i thread per calcolare il pagerank in parallelo
PageRankResult pagerank_parallel(grafo *g, int num_nodes, double damping, double epsilon, int maxiter, int num_threads) {
    printf("Calcolo del pagerank con %d nodi, damping %.2f, epsilon %.2e, maxiter %d, %d threads\n", num_nodes, damping, epsilon, maxiter, num_threads);
    double *ranks = (double *)malloc(num_nodes * sizeof(double));
    double *old_ranks = (double *)malloc(num_nodes * sizeof(double));
    double *sum_dead_ends = (double *)malloc(sizeof(double));
    double *delta = (double *)malloc(sizeof(double));
    double init_rank = 1.0 / num_nodes;
    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    pthread_t signal_thread_id;

    for (int i = 0; i < num_nodes; i++) {
        ranks[i] = init_rank;
    }

    PagerankThreadData *thread_data = (PagerankThreadData *)malloc(num_threads * sizeof(PagerankThreadData));
    double *local_deltas = (double *)malloc(num_threads * sizeof(double));
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, num_threads);
    pthread_mutex_t lock;
    pthread_mutex_init(&lock, NULL);
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGUSR1);
    pthread_sigmask(SIG_BLOCK, &set, NULL);
    int iterations = 0;
    int stop_flag = 0;

    SignalThreadData signal_data = { &iterations, ranks, num_nodes, &lock };
    pthread_create(&signal_thread_id, NULL, signal_thread, &signal_data);

    memcpy(old_ranks, ranks, num_nodes * sizeof(double));
    *sum_dead_ends = 0.0;
    for (int i = 0; i < num_nodes; i++) {
        if (g->out[i] == 0) {
            *sum_dead_ends += old_ranks[i];
        }
    }

    for (int i = 0; i < num_threads; i++) {
        int start = i * (num_nodes / num_threads);
        int end = (i + 1) * (num_nodes / num_threads);
        if (i == num_threads - 1) {
            end = num_nodes;
        }
        thread_data[i].g = g;
        thread_data[i].ranks = ranks;
        thread_data[i].old_ranks = old_ranks;
        thread_data[i].damping = damping;
        thread_data[i].start = start;
        thread_data[i].end = end;
        thread_data[i].local_deltas = local_deltas;
        thread_data[i].thread_id = i;
        thread_data[i].iterations = &iterations;
        thread_data[i].epsilon = epsilon;
        thread_data[i].maxiter = maxiter;
        thread_data[i].barrier = &barrier;
        thread_data[i].stop_flag = &stop_flag;
        thread_data[i].num_threads = num_threads;
        thread_data[i].sum_dead_ends = sum_dead_ends;
        thread_data[i].delta = delta;
        pthread_create(&threads[i], NULL, pagerank_thread, (void *)&thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_cancel(signal_thread_id);
    pthread_join(signal_thread_id, NULL);

    double total_rank = 0.0;
    for (int i = 0; i < num_nodes; i++) {
        total_rank += ranks[i];
    }
    for (int i = 0; i < num_nodes; i++) {
        ranks[i] /= total_rank;
    }

    PageRankResult result;
    result.ranks = ranks;
    result.iterations = iterations;
    result.delta = *delta;

    free(old_ranks);
    free(sum_dead_ends);
    free(threads);
    free(thread_data);
    free(local_deltas);
    pthread_barrier_destroy(&barrier);

    return result;
}

int main(int argc, char *argv[]) {
    // Valori di default
    int K = 3;
    int M = 100;
    double D = 0.9;
    double E = 1e-8;
    int T = 3;
    char *infile = NULL;

    // Parse delle opzioni della linea di comando
    int opt;
    while ((opt = getopt(argc, argv, "k:m:d:e:t:")) != -1) {
        switch (opt) {
            case 'k':
                K = atoi(optarg);
                break;
            case 'm':
                M = atoi(optarg);
                break;
            case 'd':
                D = atof(optarg);
                break;
            case 'e':
                E = atof(optarg);
                break;
            case 't':
                T = atoi(optarg);
                break;
            default:
                fprintf(stderr, "Usage: %s [-k K] [-m M] [-d D] [-e E] [-t T] infile\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    // L'ultimo argomento deve essere il file di input
    if (optind < argc) {
        infile = argv[optind];
    } else {
        fprintf(stderr, "Usage: %s [-k K] [-m M] [-d D] [-e E] [-t T] infile\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Inizializzazione grafo e altre strutture
    grafo *g = (grafo *)malloc(sizeof(grafo));
    SharedBuffer task_queue;
    initSharedBuffer(&task_queue, 1024);
    ThreadData data;
    data.buffer = &task_queue;
    data.g = g;
    data.filename = infile;
    data.num_threads = T;

    ThreadPool pool;
    threadPoolInit(&pool, T, &task_queue, g);
    data.pool = &pool;

    // Creazione thread per la lettura del buffer e creazione del grafo
    for (int i = 0; i < pool.num_aux_threads; i++) {
        if (pthread_create(&pool.aux_threads[i], NULL, thread_function, &pool) != 0) {
            perror("Failed to create thread");
            threadPoolDestroy(&pool);
            return EXIT_FAILURE;
        }
    }
    
    // Creazione del thread capolettore
    pthread_t capolettore_thread;
    pthread_create(&capolettore_thread, NULL, funzione_capolettore, &data);

    //join del thread capolettore e dei thread ausiliari per avere la certezza che il grafo sia stato creato
    pthread_join(capolettore_thread, NULL);
    threadPoolDestroy(&pool);

    // Distruggo il buffer condiviso dopo la creazione del grafo
    destroySharedBuffer(&task_queue);

    // Calcolo del PageRank parallelo
    PageRankResult result = pagerank_parallel(g, g->N, D, E, M, T);

    // Stampa delle statistiche del grafo
    int num_deadends = 0;
    for (int i = 0; i < g->N; i++) {
        if (g->out[i] == 0) {
            num_deadends++;
        }
    }
    printf("Number of nodes: %d\n", g->N);
    printf("Number of dead-end nodes: %d\n", num_deadends);
    printf("Damping factor: %.2f\n", D);
    printf("Epsilon: %.2e\n", E);
    printf("Max iterations: %d\n", M);

    if(result.iterations == M) {
        printf("PageRank did not converge after %d iterations\n", M);
    } else {
        printf("Converged after %d iterations\n", result.iterations);
    }
    
    double sum_ranks = 0.0;
    for (int i = 0; i < g->N; ++i) {
        sum_ranks += result.ranks[i];
    }
    printf("Sum of ranks: %.4f (should be 1)\n", sum_ranks);

    // Stampa dei top K nodi per PageRank
    printf("Top %d nodes:\n", K);
    int *indices = (int *)malloc(g->N * sizeof(int));
    for (int i = 0; i < g->N; i++) {
        indices[i] = i;
    }

    // Ordinamento degli indici usando qsort
    qsort_r(indices, g->N, sizeof(int), compare, result.ranks);

    // Stampa i top K nodi
    for (int i = 0; i < K && i < g->N; ++i) {
        printf("  Node %d: rank = %.6f\n", indices[i], result.ranks[indices[i]]);
    }

    // Libera la memoria allocata
    free(indices);
    free(result.ranks);
    freeGraph(g);

    return EXIT_SUCCESS;
}




