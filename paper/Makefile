PAPER = entropic_unification
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean figures

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex figures
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

figures:
	cd .. && python paper/make_figures.py

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.synctex.gz

clean-all: clean
	rm -f $(PAPER).pdf
