"use strict";

const element = id => document.getElementById(id);
const state = { supported: false, raw: false, parser: false, busy: false, pollPath: null, maxRequest: 0 };
const inputKeys = ["ca_image", "garment_image", "person_pose", "garment_pose"];

async function api(path, body) {
    const response = await fetch(path, body === undefined ? {} : {
        method: "POST", headers: { "Content-Type": "application/json" }, body
    });
    const result = await response.json();
    if (!response.ok) {
        const error = new Error(typeof result.error === "string" ? result.error : `HTTP ${response.status}`);
        error.status = response.status;
        throw error;
    }
    return result;
}

function setBusy(busy) {
    state.busy = busy;
    element("controls").disabled = busy || !state.supported || !!state.pollPath;
    element("cancel").disabled = !busy || !state.pollPath;
    element("resume").disabled = busy || !state.pollPath;
}

function integer(id, low, high) {
    const value = Number(element(id).value);
    if (!Number.isInteger(value) || value < low || value > high) throw new Error(`${id} must be an integer in [${low}, ${high}].`);
    return value;
}

async function preparedInputs(files) {
    const manifests = files.filter(file => file.name.toLowerCase().endsWith(".json"));
    if (manifests.length !== 1 || manifests[0].size > 16384) throw new Error("Select exactly one small manifest JSON.");
    const inputs = JSON.parse(await manifests[0].text());
    const allowed = new Set(["schema", "category", "crop", ...inputKeys]);
    if (!inputs || inputs.schema !== "fashn-vton-prepared-v1" ||
        !["tops", "bottoms", "one-pieces"].includes(inputs.category) ||
        Object.keys(inputs).some(key => !allowed.has(key))) throw new Error("Unsupported prepared manifest.");
    for (let index = 0; index < inputKeys.length; index++) {
        const key = inputKeys[index];
        if (typeof inputs[key] !== "string") throw new Error(`Missing ${key} filename.`);
        const name = inputs[key].split(/[\\/]/).pop();
        const matches = files.filter(file => file.name === name);
        if (matches.length !== 1) throw new Error(`Select exactly one file named ${name}.`);
        inputs[key] = await encodedPNG(matches[0], index < 2, key, false);
    }
    return inputs;
}

async function encodedPNG(file, rgb, key, raw) {
    const name = file.name;
    if (file.size > 3 * 1024 * 1024 * 3 / 4 || file.size < 33) throw new Error(`${name} exceeds the PNG limit or is truncated.`);
    const bytes = new Uint8Array(await file.arrayBuffer());
    const header = new DataView(bytes.buffer);
    const width = header.getUint32(16), height = header.getUint32(20);
    const dimensions = raw ? width > 0 && height > 0 && width <= 4096 && height <= 4096 && width * height <= 4194304 :
        width === 576 && height === 864;
    if ([137,80,78,71,13,10,26,10].some((byte, i) => bytes[i] !== byte) ||
        !dimensions || bytes[24] !== 8 || bytes[25] !== (rgb ? 2 : 0)) {
        throw new Error(`${key} must be an 8-bit ${rgb ? "RGB" : "grayscale"} PNG within ${raw ? "raw size limits" : "the 576x864 canvas"}.`);
    }
    let binary = "";
    for (let offset = 0; offset < bytes.length; offset += 8192) {
        binary += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
    }
    return btoa(binary);
}

async function rawInputs() {
    if (!state.raw) throw new Error("Native raw preparation is not enabled on this server.");
    const raw = { category: element("category").value, garment_photo_type: element("photo-type").value,
        segmentation_free: !element("mask-person").checked };
    if (!["tops", "bottoms", "one-pieces"].includes(raw.category) ||
        !["flat-lay", "model"].includes(raw.garment_photo_type)) throw new Error("Invalid raw preparation mode.");
    if (!state.parser && (!raw.segmentation_free || raw.garment_photo_type === "model")) {
        throw new Error("This mode requires an operator-consented research/evaluation parser.");
    }
    for (const [id, key] of [["raw-person", "person_image"], ["raw-garment", "garment_image"]]) {
        const files = [...element(id).files];
        if (files.length !== 1) throw new Error(`Select one ${key} RGB PNG.`);
        raw[key] = await encodedPNG(files[0], true, key, true);
    }
    return raw;
}

async function requestBody() {
    const raw = element("input-mode").value === "raw";
    const inputs = raw ? await rawInputs() : await preparedInputs([...element("files").files]);
    element("input-info").textContent = `${raw ? "Raw" : "Prepared"} ${inputs.category}; native 576 x 864 canvas.`;
    const steps = integer("steps", 1, 1000);
    const cfg = Number(element("cfg").value), shift = Number(element("shift").value);
    if (!Number.isFinite(cfg) || cfg < 0 || !Number.isFinite(shift) || Math.abs(shift) > 20) throw new Error("Invalid CFG or shift.");
    const seedText = element("seed").value;
    if (!/^[0-9]+$/.test(seedText) || seedText.length > 20 || BigInt(seedText) > 18446744073709551615n) {
        throw new Error("Seed must be an unsigned 64-bit integer.");
    }
    const body = { [raw ? "raw_inputs" : "inputs"]: inputs, steps, cfg, shift,
        skip_cfg_last_n_steps: integer("skip", 0, steps), sample_count: integer("samples", 1, 4) };
    // Preserve uint64 seeds without JavaScript Number rounding.
    const json = JSON.stringify(body).slice(0, -1) + ',"seed":' + BigInt(seedText).toString() + "}";
    if (new TextEncoder().encode(json).length > state.maxRequest) throw new Error("Request exceeds the server limit.");
    return json;
}

function showResults(job) {
    element("results").replaceChildren();
    for (const [index, result] of job.result.images.entries()) {
        if (!/^[A-Za-z0-9+/]+={0,2}$/.test(result.b64_json)) throw new Error("Invalid image received from server.");
        const figure = document.createElement("figure");
        const image = document.createElement("img"), link = document.createElement("a");
        image.src = "data:image/png;base64," + result.b64_json;
        image.alt = `Try-on sample ${index + 1}`;
        link.href = image.src;
        link.download = `try-on-${job.id}-${index + 1}.png`;
        link.textContent = `Download sample ${index + 1}`;
        figure.append(image, link);
        element("results").append(figure);
    }
}

async function poll() {
    const job = await api(state.pollPath);
    const progress = job.progress;
    if (progress) {
        element("progress").max = Math.max(1, progress.total_steps);
        element("progress").value = progress.completed_steps;
    }
    element("status").textContent = `${job.id}: ${job.status}` +
        (job.status === "queued" ? ` (queue position ${job.queue_position})` : "") +
        (progress ? `; ${progress.completed_steps}/${progress.total_steps} steps` : "") +
        (progress?.phase ? `; ${progress.phase}` : "") +
        (job.cancellation_requested ? "; cancellation requested" : "");
    if (job.status === "completed") showResults(job);
    else if (job.status === "failed") element("status").textContent += `: ${job.error.message}`;
    else if (job.status !== "cancelled") return false;
    return true;
}

element("steps").addEventListener("input", () => { element("skip").max = element("steps").value; });
function updateInputMode() {
    const raw = element("input-mode").value === "raw";
    element("prepared-controls").hidden = raw;
    element("prepared-controls").disabled = raw;
    element("raw-controls").hidden = !raw;
    element("raw-controls").disabled = !raw;
}
element("input-mode").addEventListener("change", updateInputMode);
async function monitor() {
    setBusy(true);
    try {
        while (state.pollPath && !(await poll())) await new Promise(resolve => setTimeout(resolve, 1000));
        state.pollPath = null;
    } catch (error) {
        if (error.status === 404 || error.status === 410) {
            state.pollPath = null;
            element("status").textContent = "Job no longer available on this server; it may have expired or the server restarted.";
        } else {
            element("status").textContent = `${error.message} Check job status or cancel: ${state.pollPath}.`;
        }
    } finally {
        setBusy(false);
        element("cancel").disabled = !state.pollPath;
    }
}

element("resume").addEventListener("click", monitor);
element("form").addEventListener("submit", async event => {
    event.preventDefault();
    if (state.busy || state.pollPath) return;
    setBusy(true);
    let accepted = false;
    try {
        const body = await requestBody();
        const job = await api("/sdcpp/v1/try_on", body);
        if (!/^\/sdcpp\/v1\/jobs\/[A-Za-z0-9_-]+$/.test(job.poll_url)) throw new Error("Invalid polling URL.");
        state.pollPath = job.poll_url;
        accepted = true;
        element("cancel").disabled = false;
        element("results").replaceChildren();
        element("progress").value = 0;
        await monitor();
    } catch (error) {
        element("status").textContent = error.message + (accepted ? ` Job may still be running: ${state.pollPath}.` : "");
    } finally {
        setBusy(false);
        // A transport failure must not strand an accepted job without cancellation.
        element("cancel").disabled = !state.pollPath;
    }
});

element("cancel").addEventListener("click", async () => {
    if (!state.pollPath) return;
    const pollPath = state.pollPath;
    element("cancel").disabled = true;
    try {
        const job = await api(pollPath + "/cancel", "{}");
        if (state.pollPath !== pollPath) return;
        element("status").textContent = job.status === "cancelled" ? "Job cancelled." :
            "Cancellation requested; waiting for the current model stage to finish.";
        if (!state.busy) await monitor();
    } catch (error) {
        if (state.pollPath !== pollPath) return;
        element("status").textContent = error.message;
        if (error.status === 404 || error.status === 410) {
            state.pollPath = null;
            setBusy(state.busy);
        }
        element("cancel").disabled = !state.pollPath;
    }
});

async function initialize() {
    try {
        const capabilities = await api("/sdcpp/v1/capabilities");
        state.supported = capabilities.supported_modes.includes("try_on");
        state.maxRequest = capabilities.limits.max_request_bytes;
        if (!state.supported) throw new Error("Restart the server with a FASHN model to use this page.");
        state.raw = !!capabilities.features?.raw_image_preprocessing;
        state.parser = !!capabilities.features?.raw_parser_modes;
        element("raw-option").disabled = !state.raw;
        element("photo-model").disabled = !state.parser;
        element("mask-person").disabled = !state.parser;
        updateInputMode();
        setBusy(false);
        element("status").textContent = state.raw ? "Ready. Select prepared inputs or raw RGB PNG photos." : "Ready. Select prepared inputs.";
    } catch (error) {
        element("status").textContent = error.message;
    }
}
initialize();
