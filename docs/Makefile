# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

mxnet:
	@echo "##################################################################"
	@echo "#                                                                #"
	@echo "#                Step 1: Building MXNet tutorials                #"
	@echo "#                                                                #"
	@echo "##################################################################"
	@DGLBACKEND=mxnet $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

pytorch:
	@echo "##################################################################"
	@echo "#                                                                #"
	@echo "#                Step 2: Building PyTorch tutorials              #"
	@echo "#                                                                #"
	@echo "##################################################################"
	@DGLBACKEND=pytorch $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) 

tensorflow:
	@echo "##################################################################"
	@echo "#                                                                #"
	@echo "#                Step 3: Building Tensorflow tutorials           #"
	@echo "#                                                                #"
	@echo "##################################################################"
	@DGLBACKEND=tensorflow $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) 

html-noexec:
	$(SPHINXBUILD) -D plot_gallery=0 -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" 
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

html: Makefile mxnet pytorch tensorflow

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) 
