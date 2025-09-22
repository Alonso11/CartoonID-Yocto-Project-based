# Configuration
TEXFILE = main.tex
JOBNAME = $(basename $(TEXFILE))
OUTPUT_DIR = build

# LaTeX commands
LATEX = pdflatex
LATEX_OPTS = -interaction=nonstopmode -output-directory=$(OUTPUT_DIR)

# Main targets
.PHONY: all clean view continuous refresh cleanup

# Default target: build the PDF and clean up
all: $(OUTPUT_DIR)/$(JOBNAME).pdf cleanup

# Create output directory if it doesn't exist
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Compile the LaTeX document to PDF
$(OUTPUT_DIR)/$(JOBNAME).pdf: $(TEXFILE) | $(OUTPUT_DIR)
	$(LATEX) $(LATEX_OPTS) $(TEXFILE)
	$(LATEX) $(LATEX_OPTS) $(TEXFILE)

# Keep only the PDF in the build directory, remove all other files
cleanup:
	@echo "Cleaning up: Keeping only the PDF file in build directory..."
	@find $(OUTPUT_DIR) -type f -not -name "$(JOBNAME).pdf" -delete

# Clean up all files including the PDF
clean:
	rm -rf $(OUTPUT_DIR)/*

# View the PDF (requires a PDF viewer; customize as needed)
view: $(OUTPUT_DIR)/$(JOBNAME).pdf
	@if command -v xdg-open > /dev/null; then \
		xdg-open $<; \
	elif command -v open > /dev/null; then \
		open $<; \
	else \
		echo "Could not find a suitable PDF viewer command."; \
	fi

# Useful for continuous compilation (requires latexmk)
continuous:
	latexmk -pdf -pvc -output-directory=$(OUTPUT_DIR) $(TEXFILE)

# Force a full refresh by cleaning and rebuilding
refresh: clean all