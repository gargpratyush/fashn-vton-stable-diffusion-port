import json
from pathlib import Path
import unittest
import tempfile
import io
import re
import shutil
import subprocess

import numpy as np
from PIL import Image

from build_fashn_comparison_report import Report, compare_images, portable, script_json
from package_fashn_precision_reports import contact_sheet, install_publication
from fashn_artifacts import sha256
from fashn_report_policy import policy_labels
from unittest.mock import patch


class ComparisonReportTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for report rendering checks")
    def test_python_baseline_rendering_and_incomplete_memory_mean(self):
        root = Path(__file__).resolve().parents[1]
        published = (root / "reports/fashn-all-comparisons.html").read_text(encoding="utf-8")
        payload = json.loads(re.search(
            r'<script id="report-data" type="application/json">(.*?)</script>',
            published, re.S).group(1))
        payload["images"] = {}
        template = (root / "scripts/fashn_comparison_template.html").read_text(encoding="utf-8")
        script = template.rsplit("<script>", 1)[1].split("</script>", 1)[0]
        harness = r"""
const assert = require("node:assert/strict");
const vm = require("node:vm");
const {script, data} = JSON.parse(require("node:fs").readFileSync(0, "utf8"));
const elements = new Map();
function element(id) {
    if (!elements.has(id)) elements.set(id, {
        value: id === "case" ? "cardigan" : id === "reference" ? "q8_0" : "",
        innerHTML: "", textContent: "", style: {}, open: false,
        prepend(node) { throw new Error(node.textContent); },
    });
    return elements.get(id);
}
element("report-data").textContent = JSON.stringify(data);
const document = {getElementById: element, addEventListener() {}, createElement() { return {}; },
    documentElement: {dataset: {}}};
const context = vm.createContext({document, console});
vm.runInContext(script, context);
assert.equal(document.documentElement.dataset.reportReady, "true");
function firstCells(id) {
    const row = element(id).innerHTML.match(/<tbody><tr>(.*?)<\/tr>/s)[1];
    return [...row.matchAll(/<td>(.*?)<\/td>/gs)].map(m => m[1]);
}
for (const name of ["cardigan", "bottoms"]) {
    for (const reference of ["q8_0", "bf16", "python"]) {
        element("case").value = name;
        element("reference").value = reference;
        vm.runInContext("populate()", context);
        const cells = firstCells("current-table");
        assert.match(cells[0], /Original PyTorch F32.*historical/);
        assert.equal(cells[1], name === "cardigan" ? "1241.3" : "1263.7");
        assert.equal(cells[2], name === "cardigan" ? "1258.0" : "1325.5");
        assert.equal(cells[3], name === "cardigan" ? "N/A" : "6.573");
        assert.equal(cells[4], name === "cardigan" ? "N/A" : "9.603");
        assert.equal(cells.at(-1), "Not evaluated here");
        assert.match(element("gallery").innerHTML, /Original PyTorch F32.*historical/);
    }
}
let summary = firstCells("aggregate-table");
assert.deepEqual(summary.slice(1, 7), ["20.88", "21.53", "N/A", "N/A", "Not controlled", "Not controlled"]);
assert.equal((element("bars").innerHTML.match(/Python \[historical\]/g) || []).length, 1);
assert.match(element("python-memory-table").innerHTML, /6.573.*9.603/s);
assert.match(element("python-memory-table").innerHTML, /cardigan<\/td><td>N\/A<\/td><td>N\/A/);
// A future valid second case may enable a mean, but never reuse the single available peak.
vm.runInContext("rowBy('python-cardigan').ws=2**30; rowBy('python-cardigan').private=2**30; summaries()", context);
summary = firstCells("aggregate-table");
assert.equal(summary[3], ((1 + 7057371136 / 2**30) / 2).toFixed(3));
assert.equal(summary[4], ((1 + 10310897664 / 2**30) / 2).toFixed(3));
assert.equal((element("bars").innerHTML.match(/Python \[historical\]/g) || []).length, 2);
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", harness],
            input=json.dumps({"script": script, "data": payload}),
            text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_selection_drives_labels_and_required_rows(self):
        for selected in ("q4_K", "q5_K"):
            state = {"selection": selected, "weights": {"selected-mixed": {
                "matrix_type": selected, "policy": {"f32_matrices": ["one", "two"]}}},
                "jobs": {name: {} for name in (f"full-cardigan-{selected}-s42", f"full-cardigan-{selected}-s43",
                                               f"repeat-cardigan-{selected}-1", f"repeat-cardigan-{selected}-2")}}
            self.assertEqual(policy_labels(state)["selected-mixed"], selected.upper() + " + 2 original F32 matrices")
            state["weights"]["selected-mixed"]["matrix_type"] = "q8_0"
            with self.assertRaises(ValueError):
                policy_labels(state)

    def test_failed_publication_has_no_success_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, stage = root / "repo", root / "stage"
            for path in (repo, stage):
                (path / "reports").mkdir(parents=True)
                (path / "reports/precision-publication-manifest.json").write_text("{}")
            (stage / "reports/output.json").write_text("new")
            manifest = {"files": {"reports/output.json": {"published_sha256": sha256(stage / "reports/output.json")}}}
            with patch.object(Path, "replace", side_effect=OSError("disk full")), self.assertRaises(OSError):
                install_publication(repo, stage, manifest)
            self.assertFalse((repo / "reports/precision-publication-manifest.json").exists())

    def test_exact_and_known_pixel_errors(self):
        a = np.full((64, 64, 3), 100, dtype=np.uint8)
        exact = compare_images(a, a)
        self.assertTrue(exact["exact"])
        self.assertIsNone(exact["psnr_db"])
        result = compare_images(a, a + 20)
        self.assertEqual(result["rmse"], 20)
        self.assertAlmostEqual(result["psnr_db"], 20 * np.log10(255 / 20))
        self.assertEqual(result["pixels_above_10_percent"], 100)
        self.assertEqual(result["pixels_above_32_percent"], 0)
        self.assertEqual(result["local"]["rmse"], 20)
        self.assertEqual(result["error_population_variance"], 0)

    def test_errors_do_not_wrap_at_uint8_boundary(self):
        a = np.full((64, 64, 3), 255, dtype=np.uint8)
        b = np.zeros_like(a)
        result = compare_images(a, b)
        self.assertEqual(result["max_abs"], 255)
        self.assertEqual(result["psnr_db"], 0)
        self.assertEqual(result["pixels_above_32_percent"], 100)

    def test_report_reuses_identical_image_payloads(self):
        report = Report.__new__(Report)
        report.images, report.arrays = {}, {}
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        first = report.image(image)
        self.assertEqual(first, report.image(image.copy()))
        self.assertEqual(len(report.images), 1)
        self.assertTrue(report.images[first].startswith("data:image/png;base64,"))
        with self.assertRaises(ValueError):
            report.image(image.astype(np.float32))

    def test_embedded_json_cannot_end_script_element(self):
        value = {"caption": "</script><script>alert('test')</script>&"}
        encoded = script_json(value)
        self.assertNotIn("<", encoded)
        self.assertNotIn("&", encoded)
        self.assertEqual(json.loads(encoded), value)
        with self.assertRaises(ValueError):
            script_json({"metric": float("nan")})

    def test_portable_paths_in_keys_values_and_lists(self):
        root = Path(r"C:\example")
        result = portable({str(root): [str(root / "artifact.json")]}, [(root, "experiment")])
        self.assertEqual(list(result), ["experiment"])
        self.assertNotIn(str(root), json.dumps(result))

    def test_contact_sheet_is_a_labeled_display_derivative(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            Image.new("RGB", (576, 768), "blue").save(source)
            original = source.read_bytes()
            result = contact_sheet([source] * 10, ["Output"] * 10, "Test report")
            with Image.open(io.BytesIO(result)) as sheet:
                self.assertEqual(sheet.size, (1334, 837))
                self.assertEqual(sheet.mode, "RGB")
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(ValueError):
                contact_sheet([source], ["Only one"], "Invalid")


if __name__ == "__main__":
    unittest.main()
