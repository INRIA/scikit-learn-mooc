render:
	rm -rf rendered_notebooks
	mkdir rendered_notebooks
	cp notebooks/* rendered_notebooks/
	jupyter nbconvert --to notebook --execute rendered_notebooks/*.ipynb --ExecutePreprocessor.timeout=None

format:
	jupytext --set-formats notebooks//ipynb,python_scripts//py:percent notebooks/*.ipynb

sync:
	jupytext --sync notebooks/*.ipynb
