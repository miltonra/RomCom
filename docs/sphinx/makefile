# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source/
BUILDDIR      = ../

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

.PHONY: help makefile

clean:
    python -m source/utilities.py clean
    @$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)"

tidy:
    python -m source/utilities.py clean
    @$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)"
    python -m source/utilities.py tidy

%: makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)"
