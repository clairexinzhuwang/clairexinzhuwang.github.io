import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import test from "node:test";

const root = new URL("../", import.meta.url);
const read = path => readFile(new URL(path, root), "utf8");
const manifest = JSON.parse(await read("site-manifest.json"));
const routeFile = route => route === "/" ? "index.html" : route.replace(/^\/+|\/+$/g, "") + "/index.html";
const pages = new Map(await Promise.all(manifest.routes.map(async route => [route, await read(routeFile(route))])));

test("all eleven current pages are exported from one reviewed revision", () => {
  assert.equal(pages.size, 11);
  assert.match(manifest.sourceRevision, /^[a-f0-9]{40}$/);
  for (const [route, html] of pages) {
    assert.match(html, /Generated from claire-wang-portfolio/, route);
    assert.match(html, /Xinzhu Wang/, route);
    assert.doesNotMatch(html, /file:\/\/|\/Users\/|localhost:|katex-error|chatgpt\.site|Private manuscript preview/, route);
    assert.match(html, /<script[^>]+type="module"/, route);
  }
});

test("every exported file matches the synchronized bundle", async () => {
  for (const [file, expected] of Object.entries(manifest.files)) {
    const actual = createHash("sha256").update(await readFile(new URL(file, root))).digest("hex");
    assert.equal(actual, expected, file);
  }
  await access(new URL(".nojekyll", root));
});

test("local navigation, anchors, scripts and styles resolve on static hosting", async () => {
  for (const [route, html] of pages) {
    for (const match of html.matchAll(/<(?:a|link|script|img)\b[^>]*?\b(?:href|src)="([^"]+)"/g)) {
      const value = match[1].replaceAll("&amp;", "&");
      const url = new URL(value, manifest.origin + route);
      if (url.origin !== manifest.origin) continue;
      const targetRoute = url.pathname.replace(/\/$/, "") || "/";
      const target = pages.get(targetRoute);
      if (target !== undefined) {
        if (url.hash) assert.ok(target.includes('id="' + decodeURIComponent(url.hash.slice(1)) + '"'), `${route}: ${value}`);
      } else {
        await access(new URL(url.pathname.slice(1), root)).catch(() => assert.fail(`${route}: missing ${value}`));
      }
    }
  }
  for (const file of Object.keys(manifest.files).filter(file => file.endsWith(".css"))) {
    for (const match of (await read(file)).matchAll(/url\(["']?([^\s)'"?]+)(?:\?[^)'"\s]*)?["']?\)/g)) {
      if (/^(data:|https?:)/.test(match[1])) continue;
      const url = match[1].startsWith("/") ? new URL(match[1].slice(1), root) : new URL(match[1], new URL(file, root));
      await access(url);
    }
  }
});

test("homepage includes the current work and removes retired content", () => {
  const html = pages.get("/");
  for (const text of ["Thesis work", "Bayesian optimization", "Clinical research", "Seeking full-time roles beginning June 2027",
    "Inference from stochastic gradient descent for U-statistics", "High-Dimensional Pairwise U-Statistic M-Estimation",
    "I work on safe multi-objective Bayesian optimization; the manuscript has passed AAAI Phase I and remains under review",
    "Bayesian Optimization for Dose Finding with Two Agents", "Emory COVID-19 Health Equity Dashboard", "MSPH thesis"])
    assert.ok(html.includes(text), text);
  assert.ok(html.indexOf("Inference from stochastic gradient descent") < html.indexOf("High-Dimensional Pairwise"));
  assert.doesNotMatch(html, /Randomized Campaign Analysis|replay-corrected|email-campaign-experiment/);
});

test("research pages preserve results, interactive controls and release boundaries", () => {
  assert.ok(!pages.has("/research/safe-multi-objective-optimization"));
  assert.ok(!Object.keys(manifest.files).some(file => /safe-mobo|safe-multi-objective-optimization/.test(file)));
  for (const html of pages.values()) assert.doesNotMatch(html, /safe-multi-objective-optimization|certified hypervolume/i);
  const low = pages.get("/research/computation-aware-inference");
  for (const text of ["without a prespecified likelihood model", "SGD and Full-Data Newton", "95.3%", "79.3%", "class=\"katex\"", "<math"])
    assert.ok(low.includes(text), text);
  assert.match(pages.get("/research/sma-treatment-comparisons"), /Selected manuscript findings/);
  for (const [route, html] of pages) {
    if (route.startsWith("/research/")) {
      assert.match(html, /advisor approval/, route);
      assert.doesNotMatch(html, /href="\/[^\"]*\.(?:pdf|docx|tex|zip)"|arxiv\.org/, route);
    }
  }
});

test("public bundle excludes source, unpublished documents and obsolete assets", async () => {
  async function walk(directory, prefix = "") {
    const paths = [];
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      if (entry.name === ".git") continue;
      const path = prefix + entry.name;
      if (entry.isDirectory()) paths.push(...await walk(new URL(entry.name + "/", directory), path + "/"));
      else paths.push(path);
    }
    return paths;
  }
  const paths = await walk(root);
  assert.deepEqual(paths.filter(path => /\.(?:pdf|docx?|tex|zip|parquet|csv|map)$/i.test(path)), []);
  assert.deepEqual(paths.filter(path => /^(?:app|dist|\.openai|node_modules|projects|artifacts|data|downloads|research-code)\//.test(path)), []);
  assert.ok(!paths.some(path => /sip-2026-poster|email-campaign|email-experiment/.test(path)));
});
