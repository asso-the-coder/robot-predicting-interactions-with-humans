# Final report

The main text must fit within four pages; references may continue afterward. The report uses the supplied APS360 LaTeX style.

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

The final two passes resolve citations, page numbers, and cross-references. Inspect `main.log` for undefined references, overfull boxes, and other warnings, then confirm that references begin after no more than four pages of main text.
