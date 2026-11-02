# Technical report

`main.tex` is a short technical report on the project: motivation, related work, data processing, models, results, post-hoc analysis, and ethics. It uses `report.sty`, a NeurIPS-derived layout. Figures are pulled from [`../assets/`](../assets) so the report and README always show the same plots.

## Editor setup on Windows

1. Install MiKTeX and enable automatic installation of missing packages.
2. In MiKTeX Console, check for and apply package updates.
3. Install the **LaTeX Workshop** extension by James Yu in VS Code.
4. Restart VS Code so the TeX executables are added to its PATH.

Verify the installation in a new PowerShell terminal:

```powershell
pdflatex --version
bibtex --version
```

Build from this directory with:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The final two passes resolve citations, page numbers, and cross-references. Inspect `main.log` for undefined references and overfull boxes. Build products (including `main.pdf`) are git-ignored. On Linux/macOS, any TeX Live installation works with the same commands.
