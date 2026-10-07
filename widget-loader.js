(function() {
    // Get the API key from the URL
    const urlParams = new URLSearchParams(window.location.search);
    const API_KEY = urlParams.get('api_key');

    if (!API_KEY) {
        console.error('AI Web Agent: No API key provided.');
        return;
    }

    // Find the backend domain (same as where this script is hosted)
    const scriptSrc = document.currentScript ? document.currentScript.src : '';
    const backendUrl = scriptSrc.split('/widget-loader.js')[0];

    // Create the iframe
    const iframe = document.createElement('iframe');
    iframe.src = backendUrl + '/widget.html?api_key=' + API_KEY;
    iframe.style.cssText = 'position:fixed;bottom:20px;right:20px;width:350px;height:500px;border:none;border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,0.2);z-index:999999;background:transparent;';
    iframe.setAttribute('allow', 'clipboard-read; clipboard-write');

    // Add it to the page
    document.body.appendChild(iframe);

    console.log('✅ AI Web Agent loaded for client:', API_KEY.substring(0, 10) + '...');
})();