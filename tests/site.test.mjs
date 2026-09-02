import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("homepage exposes inspectable public work and keeps private research private", async () => {
  const html = await readFile(new URL("index.html", root), "utf8");

  assert.match(html, /Email Campaign Experiment/);
  assert.match(html, /64,000 customers/);
  assert.match(html, /Training-Budget Trade-offs in a Standard Recommender/);
  assert.match(html, /100,836 ratings/);
  assert.match(html, /Technical details and results are not public/);
  assert.match(html, /The Price of Safety in Multi-Objective Optimization/);
  assert.doesNotMatch(html, /available upon request|file:\/\/|\/Users\//i);
});

test("public aggregates and decision memos are present", async () => {
  const email = JSON.parse(
    await readFile(new URL("data/email-experiment.json", root), "utf8"),
  );
  const ranking = JSON.parse(
    await readFile(new URL("data/recommender-benchmark.json", root), "utf8"),
  );

  assert.equal(email.data_quality.row_count, 64_000);
  assert.equal(email.data_quality.missing_cells, 0);
  assert.equal(ranking.dataset.rating_rows, 100_836);
  assert.equal(ranking.dataset.users, 610);
  assert.equal(ranking.settings.length, 4);

  await access(new URL("artifacts/email-campaign-decision-memo.pdf", root));
  await access(new URL("artifacts/recommender-decision-memo.pdf", root));
  await access(new URL("projects/email-campaign-experiment/tests", root));
  await access(new URL("projects/recommender-training-benchmark/tests", root));
});
