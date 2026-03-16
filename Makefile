PYTHON ?= python
VENV_PYTHON ?= .venv/bin/python

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) -m llm_project.experiments.train_char_gpt --input data/sample.txt

generate:
	$(PYTHON) -m llm_project.experiments.generate --checkpoint checkpoints/last.pt --prompt "$(PROMPT)"

clean:
	rm -rf checkpoints/*.pt __pycache__ llm_project/**/__pycache__
