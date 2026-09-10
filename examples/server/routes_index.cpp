#include "routes.h"
#include "try_on_assets.h"

#include <fstream>
#include <iterator>

void register_index_endpoints(httplib::Server& svr, const SDSvrParams& svr_params, const std::string& index_html) {
    svr.Get("/try-on", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'unsafe-inline'; connect-src 'self'; img-src data:; base-uri 'none'; form-action 'self'; frame-ancestors 'none'");
        res.set_header("X-Content-Type-Options", "nosniff");
        res.set_content(try_on_html, "text/html; charset=utf-8");
    });
    svr.Get("/try-on.js", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("X-Content-Type-Options", "nosniff");
        res.set_content(try_on_js, "text/javascript; charset=utf-8");
    });
    const std::string serve_html_path = svr_params.serve_html_path;
    svr.Get("/", [serve_html_path, index_html](const httplib::Request&, httplib::Response& res) {
        if (!serve_html_path.empty()) {
            std::ifstream file(serve_html_path);
            if (file) {
                std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
                res.set_content(content, "text/html");
            } else {
                res.status = 500;
                res.set_content("Error: Unable to read HTML file", "text/plain");
            }
        } else {
            res.set_content(index_html, "text/html");
        }
    });
}
