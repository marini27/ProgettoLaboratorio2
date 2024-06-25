import socket
import threading
import sys
import logging

# Configurazione del logging
logging.basicConfig(filename='client.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Configurazione del server
HOST = '127.0.0.1'
PORT = 56984

# Funzione per inviare il grafo al server e ricevere il risultato
def send_graph(file_name):
    try:
        with open(file_name, 'r') as f:
            lines = f.readlines()
            # Ignora le righe di commento
            data_lines = [line.strip() for line in lines if not line.startswith('%') and line.strip()]
            header = data_lines[0].split()
            n, a = int(header[0]), int(header[2])
            edges = [tuple(map(int, line.split())) for line in data_lines[1:]]
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(n.to_bytes(4, byteorder='little'))
            s.sendall(a.to_bytes(4, byteorder='little'))
            
            for origin, dest in edges:
                s.sendall(origin.to_bytes(4, byteorder='little'))
                s.sendall(dest.to_bytes(4, byteorder='little'))
            
            exit_code = int.from_bytes(s.recv(4), byteorder='little')
            message = s.recv(4096).decode()
        
        print(f"{file_name} Exit code: {exit_code}")
        print(f"{file_name} {message.strip()}")
        print(f"{file_name} Bye")
        
    except Exception as e:
        logging.error(f"Error sending graph from file {file_name}: {e}")

# Funzione principale del client
def main():
    file_names = sys.argv[1:]
    
    threads = []
    
    for file_name in file_names:
        client_thread = threading.Thread(target=send_graph, args=(file_name,))
        client_thread.start()
        threads.append(client_thread)
    
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
