(function(){
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

  function displayContentOnly() {
    document.querySelector('#site-navigation').remove();
    document.querySelector('.topbar').remove();
    document.querySelector('.prev-next-bottom').remove();
    document.querySelector('.footer').remove();
    var elementsToRemove = document.querySelectorAll('.remove-from-content-only');
    elementsToRemove.forEach(
      function(el) { el.remove(); }
    );
    document.querySelector('#main-content').querySelector('.col-md-9').className = 'col-12';

    var style=document.createElement('style');
    style.appendChild(document.createTextNode('hypothesis-sidebar, hypothesis-notebook, hypothesis-adder{display:none!important;}'));
    document.getElementsByTagName('head')[0].appendChild(style);
  }

  document.addEventListener("DOMContentLoaded", function() {
    if (inIframe() || contentOnly()) {
      displayContentOnly();
    }
  });
}());
