#!/usr/bin/env python3
"""Dead-simple static server for the results viewer.

    python web/serve.py            # serves the web/ dir at http://localhost:8000
    python web/serve.py 9000       # ...on port 9000

Serves the directory this file lives in, with caching disabled so a freshly
recorded run.bin is always picked up.
"""
import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
WEB_DIR = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=WEB_DIR, **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    print(f"serving {WEB_DIR} at http://localhost:{PORT}  (Ctrl+C to stop)")
    http.server.HTTPServer(("localhost", PORT), Handler).serve_forever()
