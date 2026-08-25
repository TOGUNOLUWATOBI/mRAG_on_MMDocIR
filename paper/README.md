# Paper draft

`main.tex` + `references.bib` — a first full draft, using ACM's `acmart` class in
`anonymous,review` mode (strips author names automatically, adds line numbers —
matches an anonymised-PDF submission requirement; swap the `\documentclass` line
if the actual programme template differs).

**I have no LaTeX installed in this environment and could not compile this.**
The fastest way to check it: create a new Overleaf project, upload both files
(Overleaf bundles `acmart` already), and compile. If compiling locally instead,
you need `acmart.cls` (TeX Live: `texlive-publishers` or similar; MiKTeX: install
on first use).

## What's solid

Every number in the Results section is pulled directly from `KG_RESEARCH_FINDINGS.md`
and the committed files under `src/results/`, cross-checked while writing. The
Systems 1–3 numbers are from the project's submitted course report (`DAT560_Group6_report.pdf`).

## What needs your attention before this is submission-ready

1. **Citations are best-effort, not verified.** Every entry in `references.bib` was
   written from memory, not checked against a live database — double-check title,
   venue, and year for each (`lewis2020rag`, `gao2023hyde`, `zheng2024stepback`,
   `press2022selfask`, `mmdocir`, `edge2024graphrag`, `gutierrez2024hipporag`)
   before trusting them. The `mmdocir` entry especially needs the real citation.
2. **The Related Work section is thin on Systems 1–3's own techniques.** The
   original report has a proper related-work treatment of the eight query-transformation
   techniques and four prompting strategies (with its own citations, numbered
   `[14]`–`[16]` and others in that PDF) — this draft doesn't reproduce that
   bibliography since I didn't have the report's `.bib`/reference list, only
   extracted text. Pull it from there rather than rewriting it.
3. **Per-technique ablation tables** (the 8 query techniques, 5 chunking strategies,
   4 prompting strategies from the original report) are summarized in one sentence
   in §5.1 rather than tabulated — add back a table if you have page budget.
4. **Anonymization**: `acmart`'s `anonymous` option handles author/affiliation
   fields automatically, but re-read the whole PDF yourself before submitting —
   don't rely on the class option alone (e.g., check figure/table captions,
   acknowledgments, and any file metadata).
5. **Length**: this draft is written for substance first: figure/table placement,
   exact section balance, and trimming to the actual page limit are not done.
6. **Fill in real author names/affiliations** and remove the `anonymous,review`
   options only for your own working copy — keep them for the actual submission.
