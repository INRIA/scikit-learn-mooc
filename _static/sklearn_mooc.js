(function(){
  function inIframe() {
    try {
      return window.self !== window.top;
    } catch (e) {
      return true;
    }
  }

  function displayContentOnly() {
    document.querySelector('#site-navigation').remove();
    document.querySelector('.topbar').remove();
    document.querySelector('.prev-next-bottom').remove();
    document.querySelector('.footer').remove();
    document.querySelector('#main-content').querySelector('.col-md-9').className = 'col-12';
  }

  if (inIframe()) {
    displayContentOnly();
  }
}());
