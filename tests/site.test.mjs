import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);
const routeFiles = [
  "index.html",
  "research/index.html",
  "research/finite-compute-inference/index.html",
  "research/high-dimensional-pairwise-inference/index.html",
  "product/index.html",
  "health/index.html",
  "code/index.html",
  "talks/index.html",
  "cv/index.html",
];

async function read(relativePath) {
  return readFile(new URL(relativePath, root), "utf8");
}

test("all generated routes exist and contain no local paths", async () => {
  for (const route of routeFiles) {
    const html = await read(route);
    assert.match(html, /Generated from claire-wang-portfolio/);
    assert.match(html, /Xinzhu Wang/);
    assert.doesNotMatch(html, /file:\/\/|\/Users\//i);
  }
});

test("homepage leads with the unified research program", async () => {
  const html = await read("index.html");
  assert.match(html, /Reliable inference for learning from pairs and tuples/);
  assert.match(html, /143\.9M/);
  assert.match(html, /360,000 records/);
  assert.match(html, /One program, two questions/);
  assert.match(html, /The Price of Safety in Multi-Objective Optimization/);
  assert.doesNotMatch(html, /recommender|MovieLens|BPR-SGD/i);
});

test("research pages expose exact claims and explicit limits", async () => {
  const finite = await read("research/finite-compute-inference/index.html");
  const high = await read("research/high-dimensional-pairwise-inference/index.html");
  assert.match(finite, /143,928,988/);
  assert.match(finite, /standard-error agreement, not repeated-sample coverage/i);
  assert.match(high, /179,992 \/ 180,000/);
  assert.match(high, /eight failures retained/i);
  assert.match(high, /not a convergence point/i);
  assert.doesNotMatch(finite + high, /arxiv\.org|main2\.pdf|main_biometrika\.pdf/i);
});

test("static assets referenced by the HTML are present", async () => {
  const html = await read("index.html");
  const cssHref = html.match(/href="(\/_next\/static\/css\/[^"]+\.css)"/)?.[1];
  assert.ok(cssHref);
  await access(new URL(cssHref.slice(1), root));
  await access(new URL("claire-wang-portrait.jpg", root));
  await access(new URL("claire-wang-sip-2026-poster.pdf", root));
  await access(new URL(".nojekyll", root));
});

test("machine-readable research results agree with displayed headline facts", async () => {
  const finite = JSON.parse(await read("data/finite-compute-results.json"));
  const high = JSON.parse(await read("data/high-dimensional-results.json"));
  assert.equal(finite.real_data[0].possible_pairs, 143_928_988);
  assert.equal(finite.finite_budget[0].optimization_share_percent, 58.6);
  assert.equal(high.formal_study.total_records, 360_000);
  assert.equal(high.known_s.exact_recovery_records, 179_992);
  assert.equal(high.formal_study.all_failures_retained, true);
});

test("finite-compute inspection source contains the algorithm, audit, frozen records, and notice", async () => {
  const base = "research-code/finite-compute-inference/";
  await access(new URL(base + "code/alg_paper.py", root));
  await access(new URL(base + "code/audit_fixed_B.py", root));
  await access(new URL(base + "experiments_fixed_B/frozen_config.sha256", root));
  await access(new URL(base + "experiments_fixed_B/formal_runs/fixed_B_replications.parquet", root));
  const notice = await read(base + "NOTICE.md");
  assert.match(notice, /No copyright licence is granted/);
});

test("no manuscript PDF is distributed in the public repository", async () => {
  const entries = await readdir(new URL(".", root), { recursive: true });
  const forbidden = entries.filter((entry) => /(?:main2|supplement2|main_biometrika|supplement_biometrika|AAAI_price_of_safety)\.pdf$/i.test(entry));
  assert.deepEqual(forbidden, []);
});
