import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import test from "node:test";

const root = new URL("../", import.meta.url);
const read = path => readFile(new URL(path, root), "utf8");
const manifest = JSON.parse(await read("site-manifest.json"));
const routeFile = route => route === "/" ? "index.html" : route.replace(/^\/+|\/+$/g, "") + "/index.html";
const pages = new Map(await Promise.all(manifest.routes.map(async route => [route, await read(routeFile(route))])));

test("all ten current pages are exported from one reviewed revision", () => {
  assert.equal(pages.size, 10);
  assert.match(manifest.sourceRevision, /^[a-f0-9]{40}$/);
  for (const [route, html] of pages) {
    assert.match(html, /Generated from claire-wang-portfolio/, route);
    assert.match(html, /Xinzhu(?: Claire| “Claire”)? Wang/, route);
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

test("search engines receive canonical identity and discovery files", async () => {
  const origin = "https://clairexinzhuwang.github.io";
  for (const [route, html] of pages) {
    const canonical = origin + (route === "/" ? "" : route + "/");
    assert.ok(html.includes('<link rel="canonical" href="' + canonical + '"/>'), route);
    assert.match(html, /<meta name="robots" content="index, follow"\/>/, route);
  }

  const home = pages.get("/");
  const jsonLdMatch = home.match(/<script type="application\/ld\+json">([^<]+)<\/script>/);
  assert.ok(jsonLdMatch);
  const person = JSON.parse(jsonLdMatch[1]);
  assert.equal(person["@type"], "Person");
  assert.equal(person["@id"], origin + "/#person");
  assert.equal(person.name, "Xinzhu Claire Wang");
  assert.ok(person.sameAs.includes("https://github.com/clairexinzhuwang"));
  const dosePaper = person.subjectOf.find(item => item.name.includes("Bayesian Optimization for Dose Finding"));
  assert.ok(dosePaper);
  assert.ok(dosePaper.identifier.some(item => item.propertyID === "DOI" && item.value === "10.48550/arXiv.2610.09245"));
  assert.ok(dosePaper.identifier.some(item => item.propertyID === "arXiv" && item.value === "2610.09245"));
  assert.ok(dosePaper.sameAs.includes("https://doi.org/10.48550/arXiv.2610.09245"));

  const robots = await read("robots.txt");
  assert.match(robots, /User-agent: \*/);
  assert.match(robots, /Sitemap: https:\/\/clairexinzhuwang\.github\.io\/sitemap\.xml/);

  const verification = await read("googlec1172d0f4f61b72c.html");
  assert.equal(verification.trim(), "google-site-verification: googlec1172d0f4f61b72c.html");

  const sitemap = await read("sitemap.xml");
  const listed = [...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map(match => match[1]).sort();
  const expected = manifest.routes.map(route => origin + (route === "/" ? "/" : route + "/")).sort();
  assert.deepEqual(listed, expected);
  assert.match(sitemap, /research\/bayesian-dose-finding\/<\/loc><lastmod>2026-10-08<\/lastmod>/);
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
    "I work on safe multi-objective Bayesian optimization.",
    "Bayesian Optimization for Dose Finding with Two Agents", "Emory COVID-19 Health Equity Dashboard", "MSPH thesis"])
    assert.ok(html.includes(text), text);
  assert.ok(html.indexOf("Inference from stochastic gradient descent") < html.indexOf("High-Dimensional Pairwise"));
  assert.doesNotMatch(html, /Randomized Campaign Analysis|replay-corrected|email-campaign-experiment/);
  for (const page of pages.values()) assert.doesNotMatch(page, /AAAI/);
});

test("research pages preserve results, interactive controls and release boundaries", () => {
  assert.ok(!pages.has("/research/safe-multi-objective-optimization"));
  assert.ok(!Object.keys(manifest.files).some(file => /safe-mobo|safe-multi-objective-optimization/.test(file)));
  for (const html of pages.values()) assert.doesNotMatch(html, /safe-multi-objective-optimization|certified hypervolume/i);
  const low = pages.get("/research/computation-aware-inference");
  for (const text of ["without a prespecified likelihood model", "58.8% of estimated total variance", "6.6% at BT/n = 20",
    "95.05%", "94.25%", "analytic Monte Carlo standard errors", "Small dots: individual coefficients",
    "Absolute standardized bias has no error bars", "23,571 policies with exactly one recorded claim and 31 coefficients",
    "41,984 training patients and 40 ranking-score coefficients", "These comparisons measure numerical agreement, not population coverage",
    "main fit is 275 Newton SEs away", "0.6362 (95% DeLong interval: 0.6248–0.6476)", "class=\"katex\"", "<math"])
    assert.ok(low.includes(text), text);
  for (const family of ["rank", "auc", "triplet"]) {
    for (const metric of ["coverage", "standardized-bias", "variance-ratio"]) {
      assert.ok(low.includes(`src="/research/v51-figure1-${family}-${metric}.svg"`));
    }
  }
  const rankFocus = low.match(/<figure\b[^>]*data-inference-focus="rank"[^>]*>[\s\S]*?<\/figure>/)?.[0];
  assert.ok(rankFocus, 'default rank view explains the inferential workflow');
  for (const phrase of ['Last-iterate inference', 'Sampled comparisons', 'Last SGD iterate', 'Marginal Wald intervals',
    'Data uncertainty', 'Sampling policies', 'Algorithmic uncertainty', 'Sampling comparisons, dataset fixed.',
    'Population loss coefficients · conceptual workflow.']) assert.ok(rankFocus.includes(phrase), phrase);
  assert.doesNotMatch(rankFocus, /<svg\b|<path\b|<math\b|class="katex"|rank-gap|type="range"|<output\b|\bbandwidth\b/);
  const rankFocusText = rankFocus.replace(/<[^>]+>/g, ' ').trim();
  assert.doesNotMatch(rankFocusText, /\d/, 'conceptual workflow shows no invented numerical values');
  assert.ok(rankFocusText.split(/\s+/).length <= 45, 'default rank workflow stays concise');
  const applicationSelect = low.match(/<select\b[^>]*id="inference-application"[^>]*>[\s\S]*?<\/select>/)?.[0];
  assert.ok(applicationSelect, 'application selection uses a native select');
  assert.match(low, /<label\b[^>]*for="inference-application"[^>]*>Application/);
  for (const [value, name, structure] of [['rank', 'Insurance claim severity', 'One sample · order 2'],
    ['auc', 'Thirty-day readmission', 'Two samples · order (1,1)'], ['triplet', 'Metric learning', 'One sample · order 3']]) {
    const option = applicationSelect.match(new RegExp(`<option\\b[^>]*value="${value}"[^>]*>[\\s\\S]*?<\\/option>`))?.[0];
    assert.ok(option, value);
    assert.ok(option.includes(`aria-label="${name}, ${structure}"`));
    assert.ok(option.endsWith(`>${name}</option>`), 'visible option contains only the application name');
  }
  assert.doesNotMatch(low, /class="inference-family-buttons"/);
  const rankTechnical = low.match(/<details\b[^>]*data-rank-technical="true"[^>]*>[\s\S]*?<\/details>/)?.[0];
  assert.ok(rankTechnical, 'rank-loss mechanics remain available on demand');
  assert.doesNotMatch(rankTechnical, /^<details\b[^>]*\bopen(?:[\s=>])/);
  assert.equal((rankTechnical.match(/<details\b/g) ?? []).length, 1, 'rank disclosure has no nested disclosure');
  assert.match(rankTechnical, /<summary>Application details<\/summary>/);
  assert.match(rankTechnical, /Data-sampling uncertainty/);
  assert.match(rankTechnical, /observed dataset held fixed/);
  assert.match(rankTechnical, /Both covariance components are estimated during the same SGD run/);
  assert.match(rankTechnical, /accounting for shared policies/);
  assert.match(rankTechnical, /23,571 single-claim policies and 31 rating variables/);
  assert.match(rankTechnical, /id="rank-gap"[^>]*type="range"/);
  assert.match(rankTechnical, /Normalized loss/);
  assert.match(rankTechnical, /aria-label="Normalized loss shown in the rank-regression graph"/);
  assert.match(rankTechnical, /<mi>h<\/mi>/);
  const lowSummary = low.match(/<p\b[^>]*data-research-summary="low-dimensional"[^>]*>([\s\S]*?)<\/p>/)?.[1];
  assert.equal(lowSummary, 'We develop Wald intervals for the population minimizer of a U-statistic loss from the last SGD iterate. One pass estimates both data-sampling and algorithmic uncertainty, without evaluating the complete criterion or revisiting sampled terms.');
  assert.doesNotMatch(low, /r72-figure1-|r72-budget-coverage|79\.3%|Data-only coverage|500 bootstrap refits|7\.169 s|0\.0027 s|Mean of five isolated single-thread runs|covariance pass through stored tuples/);
  const earlierBenchmark = low.match(/<details\b[^>]*><summary>SGD and Full-Data Newton · earlier proposal benchmark<\/summary>[\s\S]*?<\/details>/)?.[0];
  assert.ok(earlierBenchmark, 'historical benchmark stays in a labeled details disclosure');
  assert.doesNotMatch(earlierBenchmark, /^<details\b[^>]*\bopen(?:[\s=>])/);
  assert.match(earlierBenchmark, /3\.32 s/);
  assert.match(earlierBenchmark, /These times do not benchmark the V51 one-pass method/);
  assert.match(earlierBenchmark, /Hardware, timing aggregation and included stages are unspecified/);
  assert.match(low, /The V51 archive does not isolate a timing benchmark for fitting plus one-pass covariance estimation/);
  assert.match(pages.get("/research/sma-treatment-comparisons"), /Selected manuscript findings/);
  for (const route of [
    "/research/computation-aware-inference",
    "/research/high-dimensional-inference",
    "/research/sma-treatment-comparisons",
  ]) {
    const html = pages.get(route);
    assert.match(html, /advisor approval/, route);
    assert.doesNotMatch(html, /href="\/[^\"]*\.(?:pdf|docx|tex|zip)"|arxiv\.org/, route);
  }

  for (const route of ["/", "/research", "/health", "/research/bayesian-dose-finding"]) {
    assert.match(pages.get(route), /href="https:\/\/doi\.org\/10\.48550\/arXiv\.2610\.09245"/, route);
    assert.match(pages.get(route), /href="https:\/\/arxiv\.org\/abs\/2610\.09245"/, route);
    assert.match(pages.get(route), /href="https:\/\/github\.com\/clairexinzhuwang\/dose-combination-bo"/, route);
  }
  const dose = pages.get("/research/bayesian-dose-finding");
  assert.match(dose, /<meta name="citation_doi" content="10\.48550\/arXiv\.2610\.09245"\/>/);
  assert.match(dose, /href="https:\/\/arxiv\.org\/pdf\/2610\.09245"/);
  assert.match(dose, /"@type":"ScholarlyArticle"/);
  assert.match(dose, /"propertyID":"DOI","value":"10\.48550\/arXiv\.2610\.09245"/);
  assert.match(dose, /"propertyID":"arXiv","value":"2610\.09245"/);
  assert.match(dose, /"codeRepository":"https:\/\/github\.com\/clairexinzhuwang\/dose-combination-bo"/);
  assert.doesNotMatch(dose, /Unpublished manuscript|data-private-research-preview="true"/i);
  assert.match(pages.get("/code"), /dose-combination-bo/);
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
  assert.deepEqual(paths.filter(path => /\.(?:pdf|docx?|tex|zip|parquet|csv|map)$/i.test(path)), ["research/SIP2026_Wang_Love.pdf"]);
  assert.equal(createHash("sha256").update(await readFile(new URL("research/SIP2026_Wang_Love.pdf", root))).digest("hex"), "053115403e2915581a894365b68221452b745f9bfab71f9e4ee853ea38475436");
  assert.deepEqual(paths.filter(path => /^(?:app|dist|\.openai|node_modules|projects|artifacts|data|downloads|research-code)\//.test(path)), []);
  assert.ok(!paths.some(path => /sip-2026-poster|email-campaign|email-experiment/.test(path)));
});
