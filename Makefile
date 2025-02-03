training:
	uv run python model_training.py

check-turbo-ml-installation:
	@echo "Checking import turboml_installer works"
	uv run python -c "import turboml_installer"

	@echo "Checking import turboml works"
	uv run python -c "import turboml"