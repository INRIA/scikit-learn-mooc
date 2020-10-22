
all: html

html:
	jupyter-book build .
	rm -rf _build/html/slides _build/html/figures && cp -r ../slides ../figures _build/html
