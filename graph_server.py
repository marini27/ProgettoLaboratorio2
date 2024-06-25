import socket
import threading
import tempfile
import subprocess
import logging
import signal
import os

# Configurazione del logging
logging.basicConfig(filename='server.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Configurazione del server
HOST = '127.0.0.1'
PORT = 56984

# Funzione per gestire la connessione con il client
def handle_client(conn, addr):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name
            
            # Ricezione numero di nodi e archi
            n = int.from_bytes(conn.recv(4), byteorder='little')
            a = int.from_bytes(conn.recv(4), byteorder='little')
            
            valid_arcs = 0
            invalid_arcs = 0
            
            temp_file.write(f"{n} {n} {a}\n".encode())
            
            for _ in range(a):
                origin = int.from_bytes(conn.recv(4), byteorder='little')
                dest = int.from_bytes(conn.recv(4), byteorder='little')
                #stampo arco
                print(f"Arco: {origin} {dest}")
                
                if 1 <= origin <= n and 1 <= dest <= n:
                    temp_file.write(f"{origin} {dest}\n".encode())
                    valid_arcs += 1
                else:
                    invalid_arcs += 1
            
            temp_file.flush()
        
        # Esecuzione del programma pagerank
        result = subprocess.run(['./pagerank', temp_filename], capture_output=True, text=True)
        
        # Inviare il risultato al client
        if result.returncode == 0:
            conn.sendall((0).to_bytes(4, byteorder='little'))
            conn.sendall(result.stdout.encode())
        else:
            conn.sendall(result.returncode.to_bytes(4, byteorder='little'))
            conn.sendall(result.stderr.encode())
        
        # Registrazione delle informazioni nel file di log
        logging.info(f"Number of nodes: {n}, Temp file: {temp_filename}, Invalid arcs: {invalid_arcs}, Valid arcs: {valid_arcs}, Pagerank exit code: {result.returncode}")
        
    except Exception as e:
        logging.error(f"Error handling client {addr}: {e}")
    finally:
        conn.close()
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# Funzione principale del server
def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen()

    # Lista per tracciare i thread attivi
    threads = []

    def signal_handler(sig, frame):
        print('\nBye dal server')
        for t in threads:
            t.join()
        server.close()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print(f"Server listening on {HOST}:{PORT}")
    
    while True:
        conn, addr = server.accept()
        print(f"Connected by {addr}")
        client_thread = threading.Thread(target=handle_client, args=(conn, addr))
        client_thread.start()
        threads.append(client_thread)

if __name__ == "__main__":
    main()
