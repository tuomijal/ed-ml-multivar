.PHONY: all clean print_vars clear truedata diagnostics clear_results

#################################################################################
# GLOBALS                                                                       #
#################################################################################

SHELL = /bin/zsh

mordir :=  /Users/jalmarituominen/fun/ed-mor/data/interim
rawdir := data/raw
interimdir := data/interim
preddir := data/processed/prediction_matrices
truedir := data/processed/true_matrices

MORDATA := $(wildcard $(mordir)/*suht.csv)
RAWDATAUP := $(patsubst $(mordir)/%.csv, $(rawdir)/%.csv, $(MORDATA))
RAWDATA = $(shell echo $(RAWDATAUP) | tr A-Z a-z)

SN = $(preddir)/50/occ-sn-u-1.csv\

DEEPAR = $(preddir)/50/occ-deepar-u-0.csv\
	$(preddir)/50/occ-deepar-a-0.csv\
	$(preddir)/50/occ-deepar-u-1.csv\
	$(preddir)/50/occ-deepar-a-1.csv\

LGBM = $(preddir)/50/occ-lgbm-u-0.csv\
	$(preddir)/50/occ-lgbm-a-0.csv\
	$(preddir)/50/occ-lgbm-u-1.csv\
	$(preddir)/50/occ-lgbm-a-1.csv\

TFT = $(preddir)/50/occ-tft-u-0.csv\
	$(preddir)/50/occ-tft-a-0.csv\
	$(preddir)/50/occ-tft-u-1.csv\
	$(preddir)/50/occ-tft-a-1.csv\

ARIMAX = $(preddir)/50/occ-arimax-u-1.csv\

NBEATS = $(preddir)/50/occ-nbeats-u-0.csv\
	$(preddir)/50/occ-nbeats-u-1.csv\

ETS = $(preddir)/50/occ-hwam-u-1.csv\
	$(preddir)/50/occ-hwdm-u-1.csv

SCRIPTS = scripts/train.py\
	scripts/models/arimax.py\
	scripts/models/deepar.py\
	scripts/models/hwam.py\
	scripts/models/hwdm.py\
	scripts/models/lgbm.py\
	scripts/models/nbeats.py\
	scripts/models/tft.py\
	scripts/models/utils.py

PLOT1 = output/plots/beds.jpg\

PLOT2 = output/plots/horizon_mae-a-1.jpg\
	output/plots/horizon_mae-u-1.jpg\
	output/plots/performance_monthly-u-1.jpg\
	output/plots/performance_monthly-a-1.jpg\

NTB2 = notebooks/plot_horizon_mae.ipynb\
	notebooks/plot_performance_monthly.ipynb\

PLOT3 = output/plots/importance-1.jpg\
	output/plots/importance-24.jpg

TABLES = output/tables/msis.tex\
	output/tables/performance.tex\
	output/tables/studies.tex\
	output/tables/distance_hospitals.tex\
	output/tables/distance_traffic.tex\
	output/tables/monthly_performance.tex\
	output/tables/seasonality.tex\
	output/tables/runtime.tex\

TEXFILES = report/manuscript.tex\
	report/sections/abstract.tex\
	report/sections/introduction.tex\
	report/sections/materials_and_methods.tex\
	report/sections/results.tex\
	report/sections/discussion.tex\
	report/sections/appendix_a.tex\
	report/sections/appendix_b.tex\
	report/sections/appendix_c.tex\
	report/sections/appendix_d.tex\
	report/sections/declarations.tex\
	report/sections/conclusions.tex\

PREDDATA = $(SN) $(DEEPAR) $(NBEATS) $(LGBM) $(TFT) $(ARIMAX) $(ETS)

TRUEDATA = data/processed/true_matrices/occ.csv

#################################################################################
# FUNCTIONS                                                                     #
#################################################################################

# $(call get_word, position_in_list, list) 
# Get nth element from {w1}-{w2}-{w3}-..-{wn}
define get_word
$(word $(1),$(subst -, ,$(basename $(notdir $(2)))))
endef

#################################################################################
# COMMANDS                                                                      #
#################################################################################

all: output/manuscript.pdf 

manuscript: output/manuscript.pdf

blind: output/manuscript_blind.pdf output/declarations.pdf

title: output/title.pdf

response: output/response.pdf

truedata: $(TRUEDATA)

preddata: $(PREDDATA)

figures: $(FIGURES)

tft: $(TFT)

lgbm: $(LGBM)

output/declarations.pdf: report/declarations.tex report/sections/declarations.tex
	cd report/\
	&& pdflatex declarations\
	&& mv declarations.pdf ../output/\
	&& open ../output/declarations.pdf

output/title.pdf: report/title.tex report/sections/abstract.tex
	cd report/\
	&& pdflatex title\
	&& mv title.pdf ../output/\
	&& open ../output/title.pdf

output/manuscript_blind.pdf: $(TEXFILES) $(TABLES) $(PLOT1) $(PLOT2) $(PLOT3) report/references.bib report/manuscript_blind.tex
	cd report/\
	&& pdflatex manuscript_blind\
	&& bibtex manuscript_blind\
	&& pdflatex manuscript_blind\
	&& pdflatex manuscript_blind\
	&& mv manuscript_blind.pdf ../output/\
	&& open ../output/manuscript_blind.pdf

output/manuscript.pdf: $(TEXFILES) $(TABLES) $(PLOT1) $(PLOT2) $(PLOT3) report/references.bib
	cd report/\
	&& pdflatex manuscript\
	&& bibtex manuscript\
	&& pdflatex manuscript\
	&& pdflatex manuscript\
	&& cp manuscript.pdf ~/Desktop/ed-tft-draft.pdf\
	&& mv manuscript.pdf ../output/\
	&& open ../output/manuscript.pdf

output/presentation.pdf:
	cd presentation/\
	&& pdflatex presentation\
	&& open presentation.pdf

$(PLOT1): output/plots/%.jpg: notebooks/plot_%.ipynb notebooks/nutils.py $(TRUEDATA)
	cd notebooks\
	&& papermill $(notdir $<) /dev/null

$(PLOT2): $(NTB2) notebooks/nutils.py $(TRUEDATA)
	cd notebooks\
	&& papermill plot_$(call get_word,1,$@).ipynb /dev/null -p FS $(call get_word,2,$@) -p HPO $(call get_word,3,$@)

$(PLOT3): notebooks/plot_importance.ipynb notebooks/nutils.py $(TRUEDATA)
	cd notebooks\
	&& papermill plot_$(call get_word,1,$@).ipynb /dev/null -p H $(call get_word,2,$@)

$(TABLES): output/tables/%.tex: notebooks/tab_%.ipynb $(TRUEDATA)
	cd notebooks\
	&& papermill $(notdir $<) /dev/null

$(PREDDATA): $(preddir)%.csv : $(SCRIPTS) data/interim/data.csv
	if command -v sbatch &> /dev/null;\
	then;\
		export TARGET=$(call get_word,1,$@)\
		&& export MODEL=$(call get_word,2,$@)\
		&& export FEATURESET=$(call get_word,3,$@)\
		&& export HPO=$(call get_word,4,$@)\
		&& sbatch\
		--output logs/slurm/$(call get_word,1,$@)-$(call get_word,2,$@)-$(call get_word,3,$@)-$(call get_word,4,$@).log\
		--job-name $(call get_word,1,$@)-$(call get_word,2,$@)-$(call get_word,3,$@)-$(call get_word,4,$@)\
		--account $(ACCOUNT)\
		scripts/batch_$(call get_word,2,$@).sh;\
	else;\
		cd scripts\
		&& python train.py $(call get_word,1,$@) $(call get_word,2,$@) $(call get_word,3,$@) $(call get_word,4,$@);\
	fi

# FIXME: For I am probably broken
$(TRUEDATA): scripts/create_true_matrix.py
	python scripts/create_true_matrix.py $(call get_word,1,$@)

data/interim/data.csv: data/raw/data.csv notebooks/preprocess_data.ipynb
	cd notebooks\
	&& papermill preprocess_data.ipynb /dev/null

#################################################################################
# Helper Commands                                                     #
#################################################################################

clean:
	rm -rf data/processed/*
	rm -r output/*

clear_report:
	rm report/manuscript.bbl
	rm report/manuscript.blg
	rm report/manuscript.log
	rm report/manuscript.out
	rm report/manuscript.aux

clear_notebooks:
	jupyter nbconvert notebooks/*.ipynb --clear-output --inplace

clear_results:
	rm -rf data/processed/prediction_matrices
	rm -rf data/processed/models
	rm -rf data/processed/studies
	rm -rf darts_logs
	rm logs/slurm/*
	rm logs/logger/*

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

print_vars:
	@echo "TARGETS"
	@echo $(TARGETS)

	@echo "INTERIMDATA"
	@echo $(INTERIMDATA)

	@echo "RAWDATA"
	@echo $(RAWDATA)

	@echo "PREDDATA"
	@echo $(PREDDATA)

	@echo "ACPLOTS"
	@echo $(ACPLOTS)
