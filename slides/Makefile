# Compilation is done via remarker and htmlark, both pip instalable

all: ml_concepts.html overfitting_vs_underfitting.html \
     learning_validation_curves.html bias_vs_variance.html \
     linear_models.html regularized_linear_models.html trees.html \
     ensemble.html concluding_remarks.html


%.html: %.md custom.css
	# HTMLArk is to embed images and css
	remarker $< -c custom.css > $@

