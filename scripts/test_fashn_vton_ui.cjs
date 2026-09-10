const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function ui(fetch) {
    const nodes = {};
    const element = id => nodes[id] ||= {
        value: "", disabled: true, files: [], handlers: {}, children: [],
        addEventListener(name, fn) { this.handlers[name] = fn; },
        append(...items) { this.children.push(...items); },
        replaceChildren() { this.children = []; }
    };
    const context = vm.createContext({
        document: { getElementById: element, createElement: element },
        fetch: async (...args) => ({ ok: true, json: async () => fetch(...args) }),
        btoa: value => Buffer.from(value, "binary").toString("base64"),
        TextEncoder, Uint8Array, DataView, setTimeout
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, "../examples/server/try_on.js"), "utf8"), context);
    return { context, nodes, element, run: code => vm.runInContext(code, context) };
}
const capabilities = { supported_modes: ["try_on"], limits: { max_request_bytes: 12 * 1024 * 1024 + 4096 } };

function fixtures() {
    const keys = ["ca_image", "garment_image", "person_pose", "garment_pose"];
    const manifest = { schema: "fashn-vton-prepared-v1", category: "tops" };
    const files = keys.map((key, index) => {
        manifest[key] = key + ".png";
        const bytes = new Uint8Array(33);
        bytes.set([137,80,78,71,13,10,26,10]);
        const view = new DataView(bytes.buffer);
        view.setUint32(16, 576); view.setUint32(20, 864);
        bytes[24] = 8; bytes[25] = index < 2 ? 2 : 0;
        return { name: manifest[key], size: bytes.length, arrayBuffer: async () => bytes.buffer };
    });
    files.push({ name: "manifest.json", size: 400, text: async () => JSON.stringify(manifest) });
    return files;
}

test("prepared uploads preserve uint64 seed and reject missing/oversized images", async () => {
    const app = ui(() => capabilities);
    await app.run("initialize()");
    Object.entries({ steps: "20", cfg: "1.5", shift: "1.5", skip: "1", samples: "1",
        seed: "18446744073709551615" }).forEach(([id, value]) => app.element(id).value = value);
    app.element("files").files = fixtures();
    const body = await app.run("requestBody()");
    assert.ok(body.endsWith('"seed":18446744073709551615}'));
    assert.equal(JSON.parse(body).inputs.category, "tops");
    app.element("files").files[0].size = 3 * 1024 * 1024;
    await assert.rejects(app.run("requestBody()"), /limit/);
    app.element("files").files = fixtures().slice(1);
    await assert.rejects(app.run("requestBody()"), /ca_image/);
});

test("failed terminal jobs release controls and show the failure", async () => {
    const app = ui(route => route.includes("capabilities") ? capabilities : {
        id: "job_1", status: "failed", error: { message: "test failure" },
        progress: { total_steps: 20, completed_steps: 1 }
    });
    await app.run("initialize()");
    app.run('state.pollPath="/sdcpp/v1/jobs/job_1"');
    await app.run("monitor()");
    assert.equal(app.run("state.pollPath"), null);
    assert.equal(app.element("controls").disabled, false);
    assert.match(app.element("status").textContent, /test failure/);
});

test("transport errors retain the job and permit retry without duplicate submission", async () => {
    let broken = true;
    const app = ui(route => {
        if (route.includes("capabilities")) return capabilities;
        if (broken) throw new Error("offline");
        return { id: "job_1", status: "cancelled", cancellation_requested: true,
            progress: { total_steps: 20, completed_steps: 1 } };
    });
    await app.run("initialize()");
    app.run('state.pollPath="/sdcpp/v1/jobs/job_1"');
    await app.run("monitor()");
    assert.equal(app.element("controls").disabled, true);
    assert.equal(app.element("cancel").disabled, false);
    assert.equal(app.element("resume").disabled, false);
    broken = false;
    await app.element("resume").handlers.click();
    assert.equal(app.run("state.pollPath"), null);
    assert.equal(app.element("controls").disabled, false);
});

test("non-FASHN capabilities keep generation disabled", async () => {
    const app = ui(() => ({ ...capabilities, supported_modes: ["txt2img"] }));
    await app.run("initialize()");
    assert.equal(app.element("controls").disabled, true);
    assert.match(app.element("status").textContent, /FASHN model/);
});

test("expired and restarted-server jobs release controls instead of stranding the UI", async () => {
    for (const status of [404, 410]) {
        const app = ui(route => {
            if (route.includes("capabilities")) return capabilities;
            throw Object.assign(new Error("job unavailable"), { status });
        });
        await app.run("initialize()");
        app.run('state.pollPath="/sdcpp/v1/jobs/job_old"');
        await app.run("monitor()");
        assert.equal(app.run("state.pollPath"), null);
        assert.equal(app.element("controls").disabled, false);
        assert.equal(app.element("cancel").disabled, true);
        assert.match(app.element("status").textContent, /expired|restarted/);
    }
});

test("raw inputs are capability gated and preserve explicit preparation choices", async () => {
    const app = ui(() => ({ ...capabilities, features: { raw_image_preprocessing: true, raw_parser_modes: false } }));
    await app.run("initialize()");
    app.element("input-mode").value = "raw";
    app.element("input-mode").handlers.change();
    assert.equal(app.element("prepared-controls").disabled, true);
    assert.equal(app.element("raw-controls").disabled, false);
    app.element("category").value = "tops";
    app.element("photo-type").value = "flat-lay";
    app.element("mask-person").checked = false;
    app.element("raw-person").files = [fixtures()[0]];
    app.element("raw-garment").files = [fixtures()[1]];
    const raw = await app.run("rawInputs()");
    assert.equal(raw.segmentation_free, true);
    assert.equal(raw.garment_photo_type, "flat-lay");
    app.element("photo-type").value = "model";
    await assert.rejects(app.run("rawInputs()"), /research\/evaluation/);
    app.run("state.parser=true");
    assert.equal((await app.run("rawInputs()")).garment_photo_type, "model");
    app.run("state.raw=false");
    await assert.rejects(app.run("rawInputs()"), /not enabled/);
});

test("a late cancellation error cannot discard a newer job", async () => {
    let rejectCancellation;
    const app = ui(route => {
        if (route.includes("capabilities")) return capabilities;
        return new Promise((resolve, reject) => { rejectCancellation = reject; });
    });
    await app.run("initialize()");
    app.run('state.pollPath="/sdcpp/v1/jobs/job_old"; setBusy(true)');
    const pending = app.element("cancel").handlers.click();
    await new Promise(resolve => setImmediate(resolve));
    app.run('state.pollPath="/sdcpp/v1/jobs/job_new"; setBusy(true)');
    rejectCancellation(Object.assign(new Error("expired"), { status: 410 }));
    await pending;
    assert.equal(app.run("state.pollPath"), "/sdcpp/v1/jobs/job_new");
    assert.equal(app.element("controls").disabled, true);
    assert.equal(app.element("cancel").disabled, false);
});
