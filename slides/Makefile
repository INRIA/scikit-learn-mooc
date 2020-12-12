# Compilation is done via remarker and htmlark, both pip instalable

all: ml_concepts.html overfitting_vs_underfitting.html \
     learning_validation_curves.html bias_vs_variance.html \
	 linear_model.html trees.html

%.html: %.md custom.css
	# HTMLArk is to embed images and css
	remarker $< -c custom.css > $@

