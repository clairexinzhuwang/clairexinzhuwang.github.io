import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);
const routeFiles = [
  "index.html",
  "research/index.html",
  "research/computation-aware-inference/index.html",
  "research/high-dimensional-inference/index.html",
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

test("homepage presents a quiet, high-level research profile", async () => {
  const html = await read("index.html");
  assert.match(html, /Reliable inference for modern statistical learning/);
  assert.match(html, /Two manuscripts in preparation/);
  assert.match(html, /Inference after high-dimensional variable selection/);
  assert.match(html, /Inference after stochastic optimization/);
  assert.match(html, /The Price of Safety in Multi-Objective Optimization/);
  assert.doesNotMatch(html, /downloads\/|results\.json/i);
});

test("doctoral research pages remain preprint-stage teasers", async () => {
  const overview = await read("research/index.html");
  const compute = await read("research/computation-aware-inference/index.html");
  const dimension = await read("research/high-dimensional-inference/index.html");
  const html = [overview, compute, dimension].join("\n");

  assert.match(overview, /technical materials will follow an approved public release/i);
  assert.match(compute, /inferential gap/i);
  assert.match(compute, /computation–precision tradeoffs/i);
  assert.match(dimension, /statistically dependent contributions/i);
  assert.match(dimension, /post-selection uncertainty/i);
  assert.match(compute, /Manuscript in preparation/i);
  assert.match(dimension, /Manuscript in preparation/i);
  assert.doesNotMatch(html, /<table\b|<figure\b|\.zip\b|\.json\b|arxiv\.org/i);
});

test("code page exposes only the cleared public case", async () => {
  const html = await read("code/index.html");
  assert.match(html, /Public code and reproducible analyses/);
  assert.match(html, /Manuscript code will follow an approved public release/);
  assert.match(html, /Randomized campaign experiment/);
  assert.match(html, /email-campaign-experiment/);
  assert.doesNotMatch(html, /download code archive|result table|\.zip\b/i);
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

test("the current public bundle contains only cleared data and documents", async () => {
  const dataFiles = await readdir(new URL("data/", root));
  const artifactFiles = await readdir(new URL("artifacts/", root));
  assert.deepEqual(dataFiles.sort(), ["email-experiment.json"]);
  assert.deepEqual(artifactFiles.sort(), ["email-campaign-decision-memo.pdf"]);

  const entries = await readdir(new URL(".", root), { recursive: true });
  const restrictedTechnicalFiles = entries.filter((entry) => /\.(?:zip|tex|parquet)$/i.test(entry));
  const publicDocuments = entries.filter((entry) => /\.pdf$/i.test(entry)).sort();
  assert.deepEqual(restrictedTechnicalFiles, []);
  assert.deepEqual(publicDocuments, [
    "artifacts/email-campaign-decision-memo.pdf",
    "claire-wang-sip-2026-poster.pdf",
  ]);
});
