import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("homepage leads with private doctoral research and keeps the public case supporting", async () => {
  const html = await readFile(new URL("index.html", root), "utf8");

  const researchPosition = html.indexOf('id="research"');
  const publicWorkPosition = html.indexOf('id="public-work"');
  const productPosition = html.indexOf('id="product"');

  assert.ok(researchPosition > -1);
  assert.ok(publicWorkPosition > researchPosition);
  assert.ok(productPosition > publicWorkPosition);
  assert.match(html, /<title>Xinzhu .*Claire.* Wang \| Statistics &amp; Data Science<\/title>/i);
  assert.match(html, /Selected materials will be linked as they become publicly available/);
  assert.match(html, /<h2 id="product-title">Product data science<\/h2>/);
  assert.match(html, /Public demonstration · Experimentation/);
  assert.match(html, /Randomized Campaign Analysis/);
  assert.match(html, /64,000 customers/);
  assert.match(html, /The Price of Safety in Multi-Objective Optimization/);
  assert.doesNotMatch(html, /recommender|MovieLens|BPR-SGD|ranking-project|recommender-benchmark/i);
  assert.doesNotMatch(html, /quantitative research roles|standard published methods|Featured project/i);
  assert.doesNotMatch(html, /interactive-lab|metric-readout|comparison-bars|lab\.js|results JSON/i);
  assert.doesNotMatch(html, /available upon request|file:\/\/|\/Users\//i);
});

test("the supporting public case remains inspectable", async () => {
  const email = JSON.parse(
    await readFile(new URL("data/email-experiment.json", root), "utf8"),
  );

  assert.equal(email.data_quality.row_count, 64_000);
  assert.equal(email.data_quality.missing_cells, 0);

  await access(new URL("artifacts/email-campaign-decision-memo.pdf", root));
  await access(new URL("projects/email-campaign-experiment/tests", root));
});
