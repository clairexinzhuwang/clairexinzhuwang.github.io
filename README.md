# Xinzhu Wang (Claire)

[Personal website](https://clairexinzhuwang.github.io/)

Research in statistical inference, U-statistic and pairwise learning methods,
Bayesian optimization, and clinical statistics, alongside applied data work.

The site includes research summaries, selected findings, interactive illustrations,
clinical projects, professional background, and a CV. Manuscripts, patient-level data,
and unreleased research implementations are not distributed in this repository.
Examples and reported findings retain their qualifications and source captions.

## Synchronization

The complete website is exported from a reviewed revision of its companion Site.
`site-manifest.json` records that revision and the hashes of all generated files.
Navigation uses ordinary links; the interactive figures run in the browser without
a server. GitHub Pages publishes the root of `main`.

With explicit authorization to publish the chosen source revision:

```sh
node scripts/sync-from-site.mjs /path/to/site FULL_REVIEWED_COMMIT_SHA
node --test tests/site.test.mjs
```

The export requires a clean source checkout and its existing production build.
It replaces pages and assets together, removes retired website files, and adjusts
the website origin and preview labels for public display. It does not publish
automatically when the companion Site changes. Review and test each export before
pushing it to GitHub.
