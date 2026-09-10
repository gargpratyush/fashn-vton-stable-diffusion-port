#!/usr/bin/env python3
"""Loopback HTTP regressions for the native FASHN async endpoint."""

import argparse
import base64
import copy
import http.client
import io
import json
import os
import re
import signal
from concurrent.futures import ThreadPoolExecutor
import socket
import subprocess
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


ARGS = None
REPORT = {}

def live_retained_results(records, now):
    return [(expires, size) for expires, size in records if expires > now]


class RetentionAccountingTests(unittest.TestCase):
    def test_expired_results_are_not_subtracted_forever(self):
        self.assertEqual(live_retained_results([(10, 100), (20, 200)], 10), [(20, 200)])
        self.assertEqual(live_retained_results([(10, 100)], 11), [])


class TryOnServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            cls.port = sock.getsockname()[1]
        cls.url = f"http://127.0.0.1:{cls.port}"
        cls.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        cls.log = (ARGS.output / "server.log").open("w", encoding="utf-8")
        cls.command = (
            [str(ARGS.server), "--diffusion-model", str(ARGS.checkpoint), "--rng", "cpu",
             "-t", "8", "--type", ARGS.weight_type, "--listen-ip", "127.0.0.1", "--listen-port", str(cls.port)] +
            (["--diffusion-fa"] if ARGS.flash_attention else []) +
            (["--try-on-dwpose-dir", str(ARGS.dwpose_dir)] if ARGS.dwpose_dir else []) +
            (["--try-on-parser-dir", str(ARGS.parser_dir), "--accept-parser-research-license"] if ARGS.parser_dir else []) +
            (["--try-on-ort-no-arena"] if ARGS.ort_no_arena else []))
        cls.process = None
        cls.addClassCleanup(cls.stop_server)
        cls.start_server()
        cls.inputs = json.loads(ARGS.prepared.read_text(encoding="utf-8"))
        for key in ("ca_image", "garment_image", "person_pose", "garment_pose"):
            data = (ARGS.prepared.parent / cls.inputs[key]).read_bytes()
            cls.inputs[key] = base64.b64encode(data).decode("ascii")

    @classmethod
    def start_server(cls):
        cls.process = subprocess.Popen(cls.command, stdout=cls.log, stderr=subprocess.STDOUT, cwd=ARGS.output,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                raise AssertionError("Server exited during startup; inspect server.log")
            try:
                with cls.opener.open(cls.url + "/sdcpp/v1/capabilities", timeout=5) as response:
                    cls.capabilities = json.load(response)
                    break
            except urllib.error.URLError:
                time.sleep(.1)
        else:
            raise AssertionError("Server did not become responsive")
    @classmethod
    def stop_server(cls):
        if cls.process is not None and cls.process.poll() is None:
            cls.process.terminate()
            try:
                cls.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.process.kill()
                cls.process.wait(timeout=10)
        cls.log.close()

    def request(self, path, body=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        request = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            response = self.opener.open(request, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read())

    def payload(self):
        return {"inputs": copy.deepcopy(self.inputs), "steps": 1, "cfg": 1, "seed": 42}

    def poll(self, path, states, timeout=900):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status, result = self.request(path)
            self.assertEqual(status, 200, result)
            if result["status"] in states:
                return result
            if result["status"] in ("failed", "cancelled", "completed"):
                self.fail(f"Unexpected terminal job: {result}")
            time.sleep(.5)
        self.fail(f"Timed out waiting for {states}")

    def test_01_capabilities(self):
        cap = self.capabilities
        self.assertEqual(cap["supported_modes"], ["try_on"])
        self.assertEqual(cap["current_mode"], "try_on")
        self.assertEqual(cap["output_formats"], ["png"])
        self.assertEqual(cap["samplers"], [])
        self.assertEqual(cap["schedulers"], [])
        self.assertEqual(cap["defaults"]["steps"], 30)
        self.assertEqual(cap["limits"]["min_width"], 576)
        self.assertEqual(cap["limits"]["max_height"], 864)
        self.assertEqual(cap["limits"]["completed_job_ttl_seconds"], 600)
        self.assertTrue(cap["features"]["cancel_generating"])
        self.assertTrue(cap["features"]["progress"])
        self.assertEqual(cap["features"]["raw_image_preprocessing"], bool(ARGS.dwpose_dir))
        self.assertEqual(cap["features"]["raw_parser_modes"], bool(ARGS.parser_dir))
        (ARGS.output / "capabilities.json").write_text(json.dumps(cap, indent=2), encoding="utf-8")
        for route in ("img_gen", "vid_gen"):
            self.assertEqual(self.request("/sdcpp/v1/" + route, {})[0], 400)

    def test_01_startup_options(self):
        for extra in ([], ["--rng", "cpu", "--steps", "2"]):
            result = subprocess.run(
                [str(ARGS.server), "--diffusion-model", str(ARGS.checkpoint), *extra],
                capture_output=True, text=True, timeout=30, cwd=ARGS.output)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("requires --rng cpu" if not extra else "request JSON", result.stdout + result.stderr)

    def test_01_try_on_page(self):
        with self.opener.open(self.url + "/try-on") as response:
            self.assertEqual(response.status, 200)
            self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
            self.assertIn(b"/try-on.js", response.read())
        with self.opener.open(self.url + "/try-on.js") as response:
            self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
            self.assertIn(b"18446744073709551615", response.read())

    def test_01_raw_startup_options(self):
        for extra, expected in (
            (["--try-on-parser-dir", "nonexistent-parser"], r"requires --try-on-dwpose-dir"),
            (["--try-on-dwpose-dir", "nonexistent-pose-models"], r"(requires a build|initialization failed)"),
            (["--try-on-ort-no-arena"], r"requires --try-on-dwpose-dir"),
        ):
            result = subprocess.run(
                [str(ARGS.server), "--diffusion-model", str(ARGS.checkpoint), "--rng", "cpu", *extra],
                capture_output=True, text=True, timeout=45, cwd=ARGS.output)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertRegex(result.stdout + result.stderr, expected)

    def test_01_failed_bind(self):
        result = subprocess.run(
            [str(ARGS.server), "--diffusion-model", str(ARGS.checkpoint), "--rng", "cpu",
             "--type", ARGS.weight_type, "--listen-ip", "::invalid::"],
            capture_output=True, text=True, timeout=30, cwd=ARGS.output)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Failed to listen", result.stdout + result.stderr)

    def test_01_restart_namespaces_jobs(self):
        body = self.payload()
        body.update(steps=30, shift=-20)
        status, before = self.request("/sdcpp/v1/try_on", body)
        self.assertEqual(status, 202)
        self.poll(before["poll_url"], {"failed"})
        self.process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
        self.assertEqual(self.process.wait(timeout=30), 0)
        type(self).start_server()
        self.assertEqual(self.request(before["poll_url"])[0], 404)
        status, after = self.request("/sdcpp/v1/try_on", body)
        self.assertEqual(status, 202)
        self.poll(after["poll_url"], {"failed"})
        expression = r"job_[0-9a-f]+_([0-9a-f]{32})_[0-9a-f]+"
        first, second = re.fullmatch(expression, before["id"]), re.fullmatch(expression, after["id"])
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first.group(1), second.group(1))
        REPORT["restart_job_namespace"] = True

    def test_02_invalid_requests(self):
        cases = []
        for key, value in (("steps", 0), ("steps", True), ("steps", 1.5), ("cfg", -1), ("cfg", 1e100),
                           ("shift", 21), ("skip_cfg_last_n_steps", 2), ("seed", -1), ("seed", 1.5),
                           ("seed", 2**64), ("sample_count", 5), ("output_format", "jpeg")):
            body = self.payload()
            body[key] = value
            cases.append((key, body))
        for key, value in (("schema", "wrong"), ("category", 0), ("person_pose", self.inputs["ca_image"]),
                           ("ca_image", str(ARGS.prepared)), ("ca_image", "https://example.com/image.png"),
                           ("person_pose", "!!!!"), ("person_pose", self.inputs["person_pose"] + " "),
                           ("crop", {"x": 575, "y": 0, "width": 2, "height": 864}),
                           ("person_pose", "A" * (3 * 1024 * 1024 + 1))):
            body = self.payload()
            body["inputs"][key] = value
            cases.append(("inputs." + key, body))
        for mode, shape in (("L", (24, 24)), ("I;16", (576, 864)), ("RGBA", (576, 864))):
            stream = io.BytesIO()
            Image.new(mode, shape).save(stream, format="PNG")
            body = self.payload()
            body["inputs"]["person_pose"] = base64.b64encode(stream.getvalue()).decode()
            cases.append((mode, body))
        for name, body in cases:
            with self.subTest(name=name):
                status, response = self.request("/sdcpp/v1/try_on", body)
                self.assertEqual(status, 400, response)
                self.assertIn("error", response)
        for raw in (b"", b"{", b"[]", b"null", b'{"cfg":NaN}'):
            self.assertEqual(self.request("/sdcpp/v1/try_on", raw=raw)[0], 400)
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            connection.putrequest("POST", "/sdcpp/v1/try_on")
            connection.putheader("Content-Length", str(self.capabilities["limits"]["max_request_bytes"] + 1))
            connection.endheaders()
            self.assertEqual(connection.getresponse().status, 413)
        finally:
            connection.close()
        self.assertEqual(self.request("/sdcpp/v1/jobs/not_a_job")[0], 404)
        REPORT["invalid_request_cases"] = len(cases) + 7

    def test_03_failed_job_and_uint64_seed(self):
        body = self.payload()
        body.update(steps=30, shift=-20, seed=2**64 - 1)
        status, accepted = self.request("/sdcpp/v1/try_on", body)
        self.assertEqual(status, 202, accepted)
        result = self.poll(accepted["poll_url"], {"failed"})
        self.assertEqual(result["error"]["code"], "generation_failed")
        self.assertIsNone(result["result"])

    def test_04_async_generation_queue_and_cancellation(self):
        if not ARGS.inference:
            self.skipTest("--inference not supplied")
        body = self.payload()
        body["inputs"]["ca_image"] = "data:image/png;base64," + body["inputs"]["ca_image"]
        status, accepted = self.request("/sdcpp/v1/try_on", body)
        self.assertEqual(status, 202, accepted)
        self.assertEqual(accepted["kind"], "try_on")
        self.poll(accepted["poll_url"], {"generating"})
        queued = []
        for i in range(self.capabilities["limits"]["max_queue_size"] - 1):
            status, job = self.request("/sdcpp/v1/try_on", self.payload())
            self.assertEqual(status, 202, job)
            queued.append(job["poll_url"])
        self.assertEqual(self.request("/sdcpp/v1/try_on", self.payload())[0], 429)
        for path in queued:
            status, cancelled = self.request(path + "/cancel", {})
            self.assertEqual(status, 200, cancelled)
            self.assertEqual(cancelled["status"], "cancelled")
            self.assertIsNone(cancelled["result"])
        result = self.poll(accepted["poll_url"], {"completed"})
        self.assertEqual(result["progress"]["completed_steps"], 1)
        self.assertEqual(result["progress"]["total_steps"], 1)
        self.assertEqual(result["result"]["output_format"], "png")
        images = result["result"]["images"]
        self.assertEqual(len(images), 1)
        decoded = base64.b64decode(images[0]["b64_json"], validate=True)
        (ARGS.output / "server-one-step.png").write_bytes(decoded)
        with Image.open(io.BytesIO(decoded)) as actual, Image.open(ARGS.expected) as expected:
            self.assertEqual(actual.mode, "RGB")
            np.testing.assert_array_equal(np.array(actual), np.array(expected))
            REPORT["output_shape"] = list(actual.size)
        REPORT["exact_cli_pixel_match"] = True
        REPORT["cancelled_queued_jobs"] = len(queued)
        status, repeat = self.request(accepted["poll_url"] + "/cancel", {})
        self.assertEqual(status, 200)
        self.assertEqual(repeat["status"], "completed")

    def test_05_active_cancellation_progress_and_next_job(self):
        if not ARGS.inference:
            self.skipTest("--inference not supplied")
        body = self.payload()
        body["steps"] = 3
        status, accepted = self.request("/sdcpp/v1/try_on", body)
        self.assertEqual(status, 202)
        path = accepted["poll_url"]
        self.poll(path, {"generating"})
        status, survivor = self.request("/sdcpp/v1/try_on", self.payload())
        self.assertEqual(status, 202)
        deadline = time.monotonic() + 900
        last = 0
        while time.monotonic() < deadline:
            status, current = self.request(path)
            self.assertEqual(status, 200)
            self.assertEqual(current["status"], "generating")
            progress = current["progress"]
            self.assertEqual(progress["total_steps"], 3)
            self.assertGreaterEqual(progress["completed_steps"], last)
            last = progress["completed_steps"]
            if last >= 1:
                break
            time.sleep(.25)
        else:
            self.fail("No step progress received")
        start = time.monotonic()
        status, cancelling = self.request(path + "/cancel", {})
        self.assertEqual(status, 202)
        self.assertTrue(cancelling["cancellation_requested"])
        # Repeated cancellation is idempotent, including a terminal-state race.
        self.assertIn(self.request(path + "/cancel", {})[0], (200, 202))
        cancelled = self.poll(path, {"cancelled"})
        self.assertIsNone(cancelled["result"])
        self.assertEqual(cancelled["error"]["code"], "cancelled")
        self.assertLess(cancelled["progress"]["completed_steps"], 3)
        REPORT["active_cancel_seconds"] = time.monotonic() - start
        REPORT["cancelled_after_steps"] = cancelled["progress"]["completed_steps"]
        result = self.poll(survivor["poll_url"], {"completed"})
        self.assertFalse(result["cancellation_requested"])
        decoded = base64.b64decode(result["result"]["images"][0]["b64_json"], validate=True)
        with Image.open(io.BytesIO(decoded)) as actual, Image.open(ARGS.expected) as expected:
            np.testing.assert_array_equal(np.array(actual), np.array(expected))
        REPORT["next_job_unaffected_by_cancellation"] = True

    def raw_payload(self):
        raw = {"category": "tops", "garment_photo_type": "model", "segmentation_free": True}
        for name, path in (("person_image", ARGS.raw_person), ("garment_image", ARGS.raw_garment)):
            with Image.open(path) as image:
                stream = io.BytesIO()
                image.convert("RGB").save(stream, format="PNG")
                raw[name] = base64.b64encode(stream.getvalue()).decode("ascii")
        return {"raw_inputs": raw, "steps": 1, "cfg": 1, "seed": 42}

    def test_06_raw_contract(self):
        if not ARGS.raw_person or not ARGS.raw_garment:
            self.skipTest("raw photographic fixtures not supplied")
        payload = self.raw_payload()
        if not ARGS.dwpose_dir or not ARGS.parser_dir:
            self.assertEqual(self.request("/sdcpp/v1/try_on", payload)[0], 400)
            if ARGS.dwpose_dir:
                payload["raw_inputs"]["garment_photo_type"] = "flat-lay"
                payload["raw_inputs"]["segmentation_free"] = False
                self.assertEqual(self.request("/sdcpp/v1/try_on", payload)[0], 400)
            return
        cases = []
        for key, value in (("category", "unknown"), ("garment_photo_type", "unknown"),
                           ("segmentation_free", 1), ("person_image", "https://example.com/a.png"),
                           ("person_image", "file.png"), ("person_image", "!!!!"),
                           ("person_image", "A" * (3 * 1024 * 1024 + 1)), ("parser_dir", "client-path")):
            body = copy.deepcopy(payload)
            body["raw_inputs"][key] = value
            cases.append(body)
        body = copy.deepcopy(payload)
        body["inputs"] = self.inputs
        cases.append(body)
        for mode, shape in (("RGBA", (50, 50)), ("L", (50, 50)), ("I;16", (50, 50)),
                            ("RGB", (4097, 1)), ("RGB", (2048, 2049))):
            stream = io.BytesIO()
            Image.new(mode, shape).save(stream, format="PNG")
            body = copy.deepcopy(payload)
            body["raw_inputs"]["person_image"] = base64.b64encode(stream.getvalue()).decode("ascii")
            cases.append(body)
        for index, body in enumerate(cases):
            with self.subTest(index=index):
                status, error = self.request("/sdcpp/v1/try_on", body)
                self.assertEqual(status, 400, error)
        REPORT["raw_invalid_cases"] = len(cases)

    def test_07_raw_preparation_cancellation_and_pixels(self):
        if not (ARGS.inference and ARGS.parser_dir and ARGS.raw_person and ARGS.raw_garment):
            self.skipTest("raw inference and evaluation parser not configured")
        status, accepted = self.request("/sdcpp/v1/try_on", self.raw_payload())
        self.assertEqual(status, 202, accepted)
        started = self.poll(accepted["poll_url"], {"generating"})
        self.assertEqual(started["progress"]["phase"], "preparing")
        self.assertEqual(self.request(accepted["poll_url"] + "/cancel", {})[0], 202)
        cancelled = self.poll(accepted["poll_url"], {"cancelled"})
        self.assertEqual(cancelled["progress"]["completed_steps"], 0)
        self.assertIsNone(cancelled["result"])
        status, accepted = self.request("/sdcpp/v1/try_on", self.raw_payload())
        self.assertEqual(status, 202, accepted)
        result = self.poll(accepted["poll_url"], {"completed"})
        self.assertEqual(result["progress"]["phase"], "done")
        decoded = base64.b64decode(result["result"]["images"][0]["b64_json"], validate=True)
        (ARGS.output / "server-raw-one-step.png").write_bytes(decoded)
        with Image.open(io.BytesIO(decoded)) as actual, Image.open(ARGS.expected) as expected:
            np.testing.assert_array_equal(np.array(actual), np.array(expected))
        REPORT["raw_exact_prepared_pixel_match"] = True
        REPORT["raw_preparation_cancellation"] = True

    def test_08_repeated_use_and_concurrent_clients(self):
        if not ARGS.soak_rounds:
            self.skipTest("--soak-rounds not supplied")
        from benchmark_fashn_vton import ProcessMemory
        memory = ProcessMemory(self.process.pid)
        samples, retained = [], 0
        retained_results = []
        REPORT["soak"] = {"samples": samples, "state": "running"}
        try:
            for index in range(ARGS.soak_rounds):
                body = self.payload()
                count = 2 if index == ARGS.soak_rounds - 1 else 1
                body["sample_count"] = count
                status, accepted = self.request("/sdcpp/v1/try_on", body)
                self.assertEqual(status, 202)
                result = self.poll(accepted["poll_url"], {"completed"})
                images = result["result"]["images"]
                self.assertEqual(len(images), count)
                self.assertEqual(result["progress"]["completed_steps"], count)
                self.assertEqual(result["progress"]["total_steps"], count)
                retained_results.append((result["completed"] + self.capabilities["limits"]["completed_job_ttl_seconds"],
                                         sum(len(image["b64_json"]) for image in images)))
                retained_results = live_retained_results(retained_results, int(time.time()))
                retained = sum(size for _, size in retained_results)
                with Image.open(io.BytesIO(base64.b64decode(images[0]["b64_json"], validate=True))) as image, Image.open(ARGS.expected) as expected:
                    np.testing.assert_array_equal(np.array(image), np.array(expected))
                    if count == 2:
                        with Image.open(io.BytesIO(base64.b64decode(images[1]["b64_json"], validate=True))) as second:
                            self.assertEqual((second.mode, second.size), (expected.mode, expected.size))
                            self.assertFalse(np.array_equal(np.array(image), np.array(second)))
                        REPORT["two_sample_batch"] = {"first_matches_single_sample": True, "distinct_second_sample": True}
                counters = memory.read()
                samples.append({"round": index, "private_bytes": counters.private_usage,
                                "known_retained_base64_bytes": retained,
                                "adjusted_private_bytes": counters.private_usage - retained})
                (ARGS.output / "soak-progress.json").write_text(json.dumps(REPORT["soak"], indent=2) + "\n", encoding="utf-8")
                print(f"Soak round {index + 1}/{ARGS.soak_rounds}: private={counters.private_usage / 1024**2:.1f} MiB", flush=True)
            warmed = [row["adjusted_private_bytes"] for row in samples[2:]]
            growth = max(warmed) - min(warmed)
            REPORT["soak"] = {"samples": samples, "post_warmup_span_bytes": growth,
                              "state": "measured",
                              "budget_bytes": ARGS.max_private_growth_mib * 1024**2,
                              "note": "Two warmup requests excluded; approximate known result storage subtracted. A short bounded-growth check, not proof of no leaks."}
            self.assertLessEqual(growth, ARGS.max_private_growth_mib * 1024**2)
            body = self.payload()
            body["steps"] = 100
            status, blocker = self.request("/sdcpp/v1/try_on", body)
            self.assertEqual(status, 202)
            self.poll(blocker["poll_url"], {"generating"})

            def queued_client(_):
                status, job = self.request("/sdcpp/v1/try_on", self.payload())
                if status != 202:
                    raise AssertionError(job)
                status, cancelled = self.request(job["poll_url"] + "/cancel", {})
                if status != 200 or cancelled["status"] != "cancelled":
                    raise AssertionError(cancelled)
                return job["id"]

            with ThreadPoolExecutor(max_workers=8) as clients:
                ids = list(clients.map(queued_client, range(16)))
            self.assertEqual(len(set(ids)), 16)
            self.assertEqual(self.request(blocker["poll_url"] + "/cancel", {})[0], 202)
            self.poll(blocker["poll_url"], {"cancelled"})
            REPORT["concurrent_clients"] = {"workers": 8, "unique_cancelled_jobs": len(ids),
                                             "post_cancel_private_bytes": memory.read().private_usage}
        finally:
            memory.close()

    def test_09_image_memory_budget(self):
        if not ARGS.inference:
            self.skipTest("inference not enabled")
        status, capabilities = self.request("/sdcpp/v1/capabilities")
        self.assertEqual(status, 200)
        limit = capabilities["limits"]["max_try_on_image_bytes"]
        baseline = capabilities["limits"]["reserved_try_on_image_bytes"]
        blocker = self.payload()
        blocker["steps"] = 1000
        status, active = self.request("/sdcpp/v1/try_on", blocker)
        self.assertEqual(status, 202, active)
        self.poll(active["poll_url"], {"generating"})
        large = self.payload()
        large["sample_count"] = 4
        queued = []
        for _ in range(capabilities["limits"]["max_queue_size"] + 1):
            status, response = self.request("/sdcpp/v1/try_on", large)
            if status == 429:
                self.assertIn("memory budget", response["error"])
                break
            self.assertEqual(status, 202, response)
            queued.append(response)
        else:
            self.fail("Image memory admission limit was not enforced")
        self.assertGreater(len(queued), 0)
        self.assertLess(len(queued) + 1, capabilities["limits"]["max_queue_size"])
        _, full = self.request("/sdcpp/v1/capabilities")
        self.assertLessEqual(full["limits"]["reserved_try_on_image_bytes"], limit)
        for job in queued:
            self.assertEqual(self.request(job["poll_url"] + "/cancel", {})[0], 200)
        _, released = self.request("/sdcpp/v1/capabilities")
        self.assertLess(released["limits"]["reserved_try_on_image_bytes"], full["limits"]["reserved_try_on_image_bytes"])
        status, replacement = self.request("/sdcpp/v1/try_on", large)
        self.assertEqual(status, 202, replacement)
        self.assertEqual(self.request(replacement["poll_url"] + "/cancel", {})[0], 200)
        self.assertEqual(self.request(active["poll_url"] + "/cancel", {})[0], 202)
        self.poll(active["poll_url"], {"cancelled"})
        _, final = self.request("/sdcpp/v1/capabilities")
        self.assertLessEqual(final["limits"]["reserved_try_on_image_bytes"], baseline + 4096)
        REPORT["image_memory_budget"] = {"limit": limit, "queued_before_rejection": len(queued),
                                         "peak_reserved": full["limits"]["reserved_try_on_image_bytes"],
                                         "final_reserved": final["limits"]["reserved_try_on_image_bytes"],
                                         "released_and_readmitted": True}

    def test_99_graceful_shutdown(self):
        if not ARGS.graceful_shutdown:
            self.skipTest("--graceful-shutdown not supplied")
        if ARGS.inference:
            body = self.payload()
            body["steps"] = 100
            status, active = self.request("/sdcpp/v1/try_on", body)
            self.assertEqual(status, 202)
            self.poll(active["poll_url"], {"generating"})
            for _ in range(2):
                self.assertEqual(self.request("/sdcpp/v1/try_on", self.payload())[0], 202)
        started = time.monotonic()
        self.process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
        self.assertEqual(self.process.wait(timeout=180), 0)
        text = (ARGS.output / "server.log").read_text(encoding="utf-8")
        self.assertIn("Try-on shutdown requested:", text)
        if ARGS.inference:
            self.assertIn("queued=2 active=1", text)
        REPORT["graceful_shutdown_seconds"] = time.monotonic() - started


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "checkpoint", "prepared", "expected", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--weight-type", choices=("f32", "f16", "bf16"), default="f32")
    for name in ("dwpose-dir", "parser-dir", "raw-person", "raw-garment"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--accept-parser-research-license", action="store_true")
    parser.add_argument("--ort-no-arena", action="store_true")
    parser.add_argument("--soak-rounds", type=int, default=0)
    parser.add_argument("--max-private-growth-mib", type=int, default=128)
    parser.add_argument("--graceful-shutdown", action="store_true")
    ARGS, remaining = parser.parse_known_args()
    if (ARGS.soak_rounds and ARGS.soak_rounds < 4) or ARGS.max_private_growth_mib < 1:
        parser.error("Soak needs at least four rounds and a positive memory budget")
    if ARGS.soak_rounds and os.name != "nt":
        parser.error("Process-memory soak measurement currently requires Windows")
    if ARGS.parser_dir and (not ARGS.accept_parser_research_license or not ARGS.dwpose_dir):
        parser.error("--parser-dir requires --dwpose-dir and explicit --accept-parser-research-license")
    for name, value in vars(ARGS).items():
        if isinstance(value, Path):
            setattr(ARGS, name, value.resolve())
    ARGS.output.mkdir(parents=True, exist_ok=False)
    result = unittest.main(argv=[__file__, *remaining], exit=False).result
    REPORT.update(passed=result.wasSuccessful(), tests_run=result.testsRun, skipped=len(result.skipped),
                  attention="flash" if ARGS.flash_attention else "manual", weight_type=ARGS.weight_type,
                  ort_no_arena=ARGS.ort_no_arena)
    (ARGS.output / "report.json").write_text(json.dumps(REPORT, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(REPORT, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
