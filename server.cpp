// Minimal static file server for the results viewer, using cpp-httplib
// (single-header, MIT). Serves ./web so you can open the DQN-plays-Mario replay.
//
//   server            # serves ./web at http://localhost:8080
//   server 9000 web   # custom port / directory
//
// (A pure-Python equivalent lives at web/serve.py.)
#include "httplib.h"
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    int port = argc > 1 ? std::atoi(argv[1]) : 8080;
    std::string dir = argc > 2 ? argv[2] : "web";

    httplib::Server svr;
    if (!svr.set_mount_point("/", dir)) {
        std::printf("could not mount directory: %s\n", dir.c_str());
        return 1;
    }
    // Disable caching so a freshly recorded run.bin is always served.
    svr.set_post_routing_handler([](const httplib::Request&, httplib::Response& res) {
        res.set_header("Cache-Control", "no-store");
    });
    std::printf("serving %s at http://localhost:%d  (Ctrl+C to stop)\n", dir.c_str(), port);
    svr.listen("localhost", port);
    return 0;
}
