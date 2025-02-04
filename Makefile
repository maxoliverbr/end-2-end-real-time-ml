feature-pipeline:
	python setup_feature_pipeline.py

model:
	python setup_model.py
	
check-turbo-ml-installation:
	@echo "Checking import turboml_installer works"
	uv run python -c "import turboml_installer"

	@echo "Checking import turboml works"
	uv run python -c "import turboml"