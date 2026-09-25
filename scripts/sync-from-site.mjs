// Export an explicitly reviewed source revision, leaving the source Site's
// default private-preview export guard in place.
import { cp, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const [sourceArgument, approvedRevision] = process.argv.slice(2);
if (!sourceArgument || !/^[a-f0-9]{40}$/.test(approvedRevision ?? "")) {
  throw new Error("Usage: node scripts/sync-from-site.mjs /path/to/reviewed/site APPROVED_FULL_COMMIT_SHA. Publication authorization is required for that revision.");
}
const source = resolve(sourceArgument);
const output = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const git = (...args) => execFileSync("git", ["-C", source, ...args], { encoding: "utf8" }).trim();
if (git("rev-parse", "HEAD") !== approvedRevision || git("status", "--porcelain")) {
  throw new Error("The source must be clean and match the explicitly approved revision.");
}

const origin = "https://clairexinzhuwang.github.io";
const sourceOrigin = "https://xinzhu-claire-wang.nhz2d6wgt7.chatgpt.site";
const routes = ["/", "/research", "/research/computation-aware-inference",
  "/research/high-dimensional-inference", "/research/bayesian-dose-finding",
  "/research/sma-treatment-comparisons",
  "/product", "/health", "/code", "/talks"];
const stage = await mkdtemp(join(tmpdir(), "reviewed-portfolio-"));
const digest = data => createHash("sha256").update(data).digest("hex");
async function files(directory, prefix = "") {
  const result = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (entry.isSymbolicLink()) throw new Error("Symbolic links are not public assets.");
    const relative = join(prefix, entry.name);
    if (entry.isDirectory()) result.push(...await files(join(directory, entry.name), relative));
    else result.push(relative);
  }
  return result.sort();
}

try {
  const { default: worker } = await import(pathToFileURL(join(source, "dist/server/index.js")));
  for (const route of routes) {
    const response = await worker.fetch(new Request(origin + route, { headers: { accept: "text/html" } }),
      { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
      { waitUntil() {}, passThroughOnException() {} });
    if (!response.ok) throw new Error(`Cannot render ${route}: ${response.status}`);
    const html = (await response.text())
      .replaceAll(sourceOrigin, origin)
      .replaceAll("Private manuscript preview", "Selected manuscript findings")
      .replaceAll("data-private-research-preview", "data-reviewed-research-summary")
      .replace("<!DOCTYPE html>", "<!DOCTYPE html><!-- Generated from claire-wang-portfolio. -->");
    if (/file:\/\/|\/Users\/|katex-error/.test(html)) throw new Error(`Invalid rendered page: ${route}`);
    const destination = join(stage, route === "/" ? "index.html" : route.slice(1) + "/index.html");
    await mkdir(dirname(destination), { recursive: true });
    await writeFile(destination, html);
  }
  await cp(join(source, "dist/client/_next"), join(stage, "_next"), { recursive: true });
  await cp(join(source, "public"), stage, { recursive: true });
  await writeFile(join(stage, ".nojekyll"), "");

  const exported = await files(stage);
  const hashes = {};
  for (const file of exported) {
    if (/\.(pdf|docx?|zip|tex|parquet|csv|map)$/i.test(file)) throw new Error(`Document or raw-data asset requires separate review: ${file}`);
    hashes[file] = digest(await readFile(join(stage, file)));
  }
  const manifest = {
    sourceRevision: approvedRevision,
    exportedAt: new Date().toISOString(),
    origin,
    routes,
    workerSha256: digest(await readFile(join(source, "dist/server/index.js"))),
    transformations: ["Use the GitHub Pages origin", "Label approved research summaries for public display"],
    files: hashes,
  };
  await writeFile(join(stage, "site-manifest.json"), JSON.stringify(manifest, null, 2) + "\n");

  // Replace generated output as a unit; retain repository tooling and history.
  const managed = ["_next", "research", "product", "health", "code", "talks", "cv", "index.html",
    "og.png", "claire-wang-portrait.jpg", ".nojekyll", "site-manifest.json",
    "projects", "data", "artifacts", "downloads", "research-code", "claire-wang-sip-2026-poster.pdf"];
  for (const name of managed) await rm(join(output, name), { recursive: true, force: true });
  await cp(stage, output, { recursive: true });
  console.log(`Exported ${routes.length} routes and ${exported.length} files from ${approvedRevision}.`);
} finally {
  await rm(stage, { recursive: true, force: true });
}
