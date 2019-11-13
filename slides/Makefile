# Compilation is done via remarker and htmlark, both pip instalable

all: ml_concepts.html overfit.html

%.html: %.md custom.css
	# HTMLArk is to embed images and css
	remarker $< -c custom.css | htmlark --ignore-js > $@

