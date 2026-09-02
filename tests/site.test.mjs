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
  assert.match(html, /360,000 outcomes/);
  assert.match(html, /One program, two questions/);
  assert.match(html, /The Price of Safety in Multi-Objective Optimization/);
  assert.doesNotMatch(html, /recommender|MovieLens|BPR-SGD/i);
});

test("research pages expose exact claims and explicit limits", async () => {
  const finite = await read("research/finite-compute-inference/index.html");
  const high = await read("research/high-dimensional-pairwise-inference/index.html");
  assert.match(finite, /143,928,988/);
  assert.match(finite, /standard-error agreement, not repeated-sample coverage/i);
  assert.match(high, /Support Recovery and Post-Recovery Simultaneous Inference for High-Dimensional Pairwise U-Statistic M-Estimators/);
  assert.match(high, /179,992 \/ 180,000/);
  assert.match(high, /eight failures retained/i);
  assert.match(high, /65 \/ 360,000/);
  assert.match(high, /capped unknown-s route has 179,134 exact recoveries/i);
  assert.doesNotMatch(high, /(?<!capped )unknown-s/i);
  assert.doesNotMatch(finite + high, /arxiv\.org|main2\.pdf|main_biometrika\.pdf/i);
});

test("code page distinguishes exact downloads, curated source, and evidence limits", async () => {
  const html = await read("code/index.html");
  assert.match(html, /Exact Round 11 numerical package/);
  assert.match(html, /finite-compute-inference-round11-code\.zip/);
  assert.match(html, /Curated code snapshot/);
  assert.match(html, /13 GB replication-level record store is external/i);
  assert.match(html, /canonical_cell_table\.json/);
  assert.doesNotMatch(html, /August 27|not been verified|synchronization required/i);
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
  assert.equal(high.capped_unknown_s.exact_recovery_records, 179_134);
  assert.equal(high.formal_study.status_counts.ok, 359_935);
  assert.equal(high.formal_study.status_counts.numerical_failure, 16);
  assert.equal(high.formal_study.status_counts.selection_failure_empty, 49);
  assert.equal(high.source_status.raw_replication_records_included, false);
  assert.equal(finite.source_status.exact_code_archive_sha256, "40ed62ebbd7ae42eb0bf0d53545d7e582eb6ce7b74d2a93f59084dcef083244a");
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
  await access(new URL("downloads/finite-compute-inference-round11-code.zip", root));
  const checksum = await read("downloads/finite-compute-inference-round11-code.zip.sha256");
  assert.match(checksum, /40ed62ebbd7ae42eb0bf0d53545d7e582eb6ce7b74d2a93f59084dcef083244a/);
});

test("high-dimensional inspection source contains exact selected code and canonical evidence", async () => {
  const base = "research-code/high-dimensional-pairwise-inference/";
  await access(new URL(base + "source/src/htp_discovery.py", root));
  await access(new URL(base + "source/src/simultaneous_inference.py", root));
  await access(new URL(base + "source/FORMAL_GRID.json", root));
  const aggregate = JSON.parse(await read(base + "evidence/formal/canonical_cell_table.json"));
  assert.equal(aggregate.cells.length, 360);
  assert.equal(aggregate.cells.reduce((sum, row) => sum + row.n_records, 0), 360_000);
  assert.equal(aggregate.checks.n_ok_records_total, 359_935);
  assert.equal(aggregate.checks.n_numerical_failure_total, 16);
  assert.equal(aggregate.checks.n_selection_failure_empty_total, 49);
  const provenance = JSON.parse(await read(base + "PROVENANCE.json"));
  assert.equal(provenance.source_archive_sha256, "fcddebfce7992a7c02a829d2b6ebf830b21103a9a1d750aa4dc72d3d50f5597f");
  assert.equal(provenance.replication_level_records_included, false);
  const notice = await read(base + "NOTICE.md");
  assert.match(notice, /No copyright licence is granted/);
});

test("no manuscript PDF is distributed in the public repository", async () => {
  const entries = await readdir(new URL(".", root), { recursive: true });
  const forbidden = entries.filter((entry) => /(?:main2|supplement2|main_biometrika|supplement_biometrika|AAAI_price_of_safety)\.pdf$/i.test(entry));
  assert.deepEqual(forbidden, []);
});
