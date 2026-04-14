from http.server import SimpleHTTPRequestHandler
import socketserver
import threading
import time
import os

def serve_image_locally():
    
    handler = SimpleHTTPRequestHandler
    
    #httpd = socketserver.TCPServer(("10.112.104.168", 9090), handler)
    httpd = socketserver.TCPServer(("0.0.0.0", 9090), handler)

    threading.Thread(target=httpd.serve_forever).start()
    return httpd

if __name__ == "__main__":
    
    httpd = serve_image_locally()
    print("Start image server ...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        httpd.shutdown()
        print("Server has been shut down.")
