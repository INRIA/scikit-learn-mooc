(function() {
    function inIframe() {
        try {
            return window.self !== window.top;
        } catch (e) {
            return true;
        }
    }

    function contentOnly() {
        var urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('content_only') !== null;
    }

    function removeIfExists(el) {
        if (el) {
            el.remove();
        };
    }

    function adjustBinderLink() {
        // Binder links to .py instead of .ipynb. In an ideal world, there
        // would be a way to do it in _config.yml or you could tell Jupyter to
        // use the Notebook interface to open the .py but ?factory=Notebook
        // does not work on the mybinder.org URL only on the
        // hub.2i2c.mybinder.org URL
        var elements = document.querySelectorAll('.dropdown-launch-buttons a');
        elements.forEach(
            function(el) {
                el.href = el.href.replace(/python_scripts\/(.+)\.py/, "notebooks/$1.ipynb");
            }
        );
    }

    function displayContentOnly() {
        removeIfExists(document.querySelector('#site-navigation'));
        removeIfExists(document.querySelector('.topbar'));
        removeIfExists(document.querySelector('.footer'));
        // the prev/next buttons at the bottom of the page may have a different
        // class (depending on the theme version maybe?), removing both to be
        // safe.
        removeIfExists(document.querySelector('.prev-next-bottom'));
        removeIfExists(document.querySelector('.prev-next-area'));
        var elementsToRemove = document.querySelectorAll('.remove-from-content-only');
        elementsToRemove.forEach(
            function(el) {
                removeIfExists(el);
            }
        );
        document.querySelector('#main-content').querySelector('.col-md-9').className = 'col-12';

        var style = document.createElement('style');
        style.appendChild(
            document.createTextNode(
                'hypothesis-sidebar, hypothesis-notebook, hypothesis-adder{display:none!important;}'));
        document.getElementsByTagName('head')[0].appendChild(style);
    }

    document.addEventListener("DOMContentLoaded", function() {
        if (inIframe() || contentOnly()) {
            displayContentOnly();
        }
        adjustBinderLink();
    });
}());
