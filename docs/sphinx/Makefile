# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source/
BUILDDIR      = ../

help:
	@python -m utilities

.PHONY: help

clean:
	@python -m utilities clean
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)"

tidy:
	@python -m utilities clean
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)"
	@python -m utilities tidy

%:
	@python -m utilities
