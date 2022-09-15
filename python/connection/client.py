import socket


class Client:

    def __init__(self, host, port):
        self._HOST = host     # The server's hostname or IP address
        self._PORT = port     # The port used by the server
        self._s = None        # The TCP socket object

    def connect(self):
        self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._s.connect((self._HOST, self._PORT))

    def disconnect(self):
        self._s.close()
        self._s = None

    def send(self, message):
        message += "\n"
        self._s.sendall(bytes(message, "ascii"))
