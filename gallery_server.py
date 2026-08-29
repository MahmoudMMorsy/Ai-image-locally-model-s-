import http.server
import socketserver
import os

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

PORT = 3000
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with socketserver.TCPServer(("0.0.0.0", PORT), CustomHandler) as httpd:
    print(f"Serving gallery server on port {PORT}...")
    httpd.serve_forever()
