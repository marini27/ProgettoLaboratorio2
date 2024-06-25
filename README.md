# Relazione sulla Gestione dei Thread

Il progetto implementa un sistema di gestione dei thread per eseguire il calcolo del PageRank su un grafo in parallelo. Questa gestione si divide principalmente in tre parti: lettura e costruzione del grafo utilizzando thread, calcolo del PageRank utilizzando thread, e gestione dei segnali per monitorare il processo.

## Lettura e Costruzione del Grafo Utilizzando Thread

### Thread Capolettore

Il thread capolettore (`funzione_capolettore`) legge il file di input e inserisce gli archi validi nel buffer condiviso, producendo i dati affinché i thread ausiliari possano processarli.

### Thread Ausiliari

I thread ausiliari (`thread_function`) consumano gli archi dal buffer condiviso e li aggiungono al grafo. Ogni thread estrae un arco dal buffer, verifica che non sia un segnale di terminazione, e poi lo aggiunge al grafo.

### Funzioni "produce" e "consume"

Le funzioni `produce` e `consume`, eseguite da `funzione_capolettore` e `thread_function`, utilizzano semafori e mutex per garantire l'accesso sicuro da parte di più thread.

## Calcolo del PageRank

I thread per il calcolo del PageRank (`pagerank_thread`) eseguono il calcolo in parallelo. Ogni thread calcola il PageRank per una porzione del grafo. La sincronizzazione tra i thread avviene tramite una barriera (`pthread_barrier_t`). I thread si sincronizzano dopo ogni iterazione del calcolo del PageRank, verificando se la condizione di terminazione è stata raggiunta (convergenza o numero massimo di iterazioni).

## Gestione Segnale SIGUSR1

Un thread dedicato (`signal_thread`) gestisce i segnali, in particolare SIGUSR1, per stampare le informazioni attuali sul calcolo del PageRank. Questo thread attende i segnali e, quando riceve SIGUSR1, stampa lo stato corrente del calcolo.

## Implementazione Client-Server con Gestione dei Thread

Il progetto include anche un'implementazione del paradigma client-server per la gestione del calcolo del PageRank su più file di input in parallelo.

### Gestione dei Thread nel Server

Il server ascolta le connessioni in entrata, gestendo ogni connessione con un thread separato, e si occupa di ricevere i dati del grafo, eseguire un programma esterno per calcolare il PageRank, e inviare i risultati al client.

#### Descrizione del Server

- **Funzione `handle_client`**: Ogni connessione client viene gestita da un nuovo thread avviato dalla funzione `handle_client`. Questo thread:
  - Riceve il numero di nodi e archi del grafo.
  - Scrive gli archi validi in un file temporaneo.
  - Esegue un programma esterno (`./main`) per calcolare il PageRank.
  - Invia il risultato del calcolo al client.
  - Logga le informazioni rilevanti e gestisce eventuali eccezioni.
  
- **Funzione `main`**:
  - Crea e configura il socket del server.
  - Gestisce i segnali per una chiusura pulita del server.
  - Ascolta le connessioni in entrata e avvia un nuovo thread per ogni client.

### Gestione dei Thread nel Client

Il client legge i dati del grafo da file, li invia al server e riceve il risultato del calcolo del PageRank, gestendo il tutto in modo parallelo utilizzando thread.

#### Descrizione del Client

- **Funzione `send_graph`**: Ogni file di input viene processato da un thread separato. Questo thread:
  - Legge il grafo dal file.
  - Si connette al server e invia il numero di nodi e archi, seguiti dagli archi stessi.
  - Riceve e stampa il risultato del calcolo del PageRank dal server.
  - Gestisce eventuali eccezioni e logga gli errori.
  
- **Funzione `main`**: 
  - Legge i nomi dei file di input passati come argomenti.
  - Crea e avvia un thread per ogni file di input, delegando a ciascun thread l'esecuzione della funzione `send_graph`.
  - Attende la terminazione di tutti i thread.

## Conclusione

L'implementazione descritta permette di gestire in modo efficiente e parallelo la ricezione e l'elaborazione dei dati del grafo grazie all'uso dei thread. Il server può gestire più connessioni contemporaneamente, mentre il client può inviare diversi file di input in parallelo, sfruttando al meglio le risorse di calcolo disponibili. La gestione dei thread nel server e nel client garantisce un'elaborazione efficiente e una comunicazione fluida tra le due parti, migliorando significativamente le prestazioni complessive del sistema.
