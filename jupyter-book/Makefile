# This assumes that the folder mooc-scikit-learn-coordination and
# scikit-learn-mooc are siblings, e.g. the repos are in the
# ~/dev/mooc-scikit-learn-coordination and ~/dev/scikit-learn-mooc. This should
# be the case in most development setups. If not then you can pass the
# GITLAB_REPO_JUPYTERBOOK_DIR variable with
# make -e GITLAB_REPO_JUPYTERBOOK_DIR=your/gitlab/repo/jupyter-book-dir/goes-here
GITLAB_REPO_JUPYTERBOOK_DIR = ../../mooc-scikit-learn-coordination/jupyter-book

all: html

html:
	jupyter-book build .
	rm -rf _build/html/slides _build/html/figures && cp -r ../slides ../figures _build/html

clean:
	# keep jupyter-cache cache folder
	jupyter-book clean .

full-clean:
	rm -rf _build

quizzes:
	python convert_exercises.py $(GITLAB_REPO_JUPYTERBOOK_DIR) .
