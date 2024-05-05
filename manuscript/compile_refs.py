#!/usr/bin/python

import subprocess, sys

commands = [
    ['pdflatex', sys.argv[1] + '.tex'],
    ['bibtex', sys.argv[1] + '.aux'],
    ['pdflatex', sys.argv[1] + '.tex'],
    ['pdflatex', sys.argv[1] + '.tex']
]

for c in commands:
    subprocess.call(c)

# how to compile the diffs of two files
# save the previous draft in a tex file `paper_prev_draft.tex`
# let the current draft be tex file `paper.tex`
# latexdiff paper_prev_draft.tex paper.tex > diff.tex
# python compile_refs.py diff
