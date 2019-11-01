# Compilation is done via remarker and htmlark, both pip instalable

all: slides.html

slides.html: slides.md custom.css
	# HTMLArk is to embed images and css
	remarker slides.md -c custom.css | htmlark --ignore-js > slides.html

