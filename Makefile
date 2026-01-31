SPECIFIED-PYTHON=3.10
TARGET_FILE=src/main.py


vpath %.py src
# Declare phony targets to avoid conflicts with files of the same name
.PHONY: all check-tools setup init shell run clean purge

all: check-tools setup init

check-tools:
	@which poetry > /dev/null || (echo "Poetry is not installed. Please install Poetry to proceed." && exit 1)
	@which pyenv > /dev/null || (echo "pyenv is not installed. Please install pyenv to proceed." && exit 1)

setup: check-tools
	@pyenv install $(SPECIFIED-PYTHON) --skip-existing && (echo "Install of Python $(SPECIFIED-PYTHON) complete.")
	@pyenv local $(SPECIFIED-PYTHON) && (echo "Set local Python version to $(SPECIFIED-PYTHON).")
	@poetry env use $(SPECIFIED-PYTHON) && (echo "Poetry is now using Python $(SPECIFIED-PYTHON).")

init: setup
	@poetry install && (echo "Project dependencies installed.")

shell:
	@echo "Using zsh, so we can't get into poetry shell directly."
	@echo "Use poetry run python <python_file.py> to run scripts within the virtual environment."

run: 
	@poetry run python $(TARGET_FILE) && (echo "Executed $(TARGET_FILE) in virtual environment successfully.")

clean:
	@rm -rf .venv && (echo "Removed virtual environment.")

purge:
	@poetry cache clear pypi --all && (echo "Cleared Poetry cache.")
	@poetry run pip cache purge && (echo "Cleared pip cache.")

