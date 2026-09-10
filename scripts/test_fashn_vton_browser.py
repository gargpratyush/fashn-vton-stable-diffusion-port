"""Optional real Chromium test of raw uploads, PNG download and UI cancellation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request

from PIL import Image
from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "checkpoint", "dwpose-dir", "parser-dir", "person", "garment", "prepared", "expected", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--accept-parser-research-license", action="store_true")
    args = parser.parse_args()
    if not args.accept_parser_research_license:
        parser.error("This worn-garment test requires explicit research/evaluation parser consent")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "external_requests": [], "page_errors": []}
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with (args.output / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(args.server), "--diffusion-model", str(args.checkpoint), "--rng", "cpu",
             "--type", "bf16", "--diffusion-fa", "-t", "8", "--listen-ip", "127.0.0.1",
             "--listen-port", str(port), "--try-on-dwpose-dir", str(args.dwpose_dir),
             "--try-on-parser-dir", str(args.parser_dir), "--accept-parser-research-license"],
            stdout=log, stderr=subprocess.STDOUT, cwd=args.output)
        try:
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Server exited during startup; inspect server.log")
                try:
                    with opener.open(url + "/sdcpp/v1/capabilities", timeout=2):
                        break
                except urllib.error.URLError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Server did not become responsive")
                    time.sleep(.1)
            for name, source in (("person", args.person), ("garment", args.garment)):
                with Image.open(source) as image:
                    image.convert("RGB").save(args.output / f"{name}.png")
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                context = browser.new_context(accept_downloads=True)

                def route_request(route):
                    if route.request.url.startswith(url + "/"):
                        route.continue_()
                    else:
                        report["external_requests"].append(route.request.url)
                        route.abort()

                context.route("**/*", route_request)
                page = context.new_page()
                page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
                page.goto(url + "/try-on")
                expect(page.locator("#controls")).to_be_enabled()
                page.select_option("#input-mode", "raw")
                page.set_input_files("#raw-person", args.output / "person.png")
                page.set_input_files("#raw-garment", args.output / "garment.png")
                page.select_option("#photo-type", "model")
                page.fill("#steps", "1")
                page.fill("#cfg", "1")
                started = time.monotonic()
                page.click("#generate")
                expect(page.locator("#results img")).to_have_count(1, timeout=300000)
                page.locator("#results img").evaluate("image => image.decode()")
                with page.expect_download() as pending:
                    page.locator("#results a").click()
                destination = args.output / "browser-one-step.png"
                pending.value.save_as(destination)
                with Image.open(destination) as actual, Image.open(args.expected) as expected:
                    assert actual.mode == expected.mode == "RGB"
                    assert actual.size == expected.size
                    assert actual.tobytes() == expected.tobytes(), "Browser pixels differ from native prepared inference"
                report["generation_seconds"] = time.monotonic() - started
                report["exact_pixels"] = True
                report["output_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
                page.screenshot(path=str(args.output / "browser.png"), full_page=True)
                expect(page.locator("#controls")).to_be_enabled()
                page.fill("#steps", "3")
                page.click("#generate")
                expect(page.locator("#cancel")).to_be_enabled()
                page.click("#cancel")
                expect(page.locator("#status")).to_contain_text(re.compile(r"cancelled", re.I), timeout=180000)
                expect(page.locator("#controls")).to_be_enabled()
                report["ui_cancellation"] = True
                page.select_option("#input-mode", "prepared")
                manifest = json.loads(args.prepared.read_text(encoding="utf-8"))
                page.set_input_files("#files", [args.prepared] + [
                    args.prepared.parent / manifest[key] for key in
                    ("ca_image", "garment_image", "person_pose", "garment_pose")])
                page.fill("#seed", "18446744073709551616")
                page.click("#generate")
                expect(page.locator("#status")).to_contain_text("unsigned 64-bit integer")
                report["prepared_mode_validation"] = True
                assert not report["external_requests"], report["external_requests"]
                assert not report["page_errors"], report["page_errors"]
                browser.close()
                report["passed"] = True
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
