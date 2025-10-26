const API_ENDPOINT = "http://127.0.0.1:5000/analyze";

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  
  if (changeInfo.status === 'complete' && tab && tab.url && (tab.url.startsWith('http://') || tab.url.startsWith('https://'))) {
    
    // Don't re-check our own warning page
    if (tab.url.includes('warning.html')) {
      return;
    }

    console.log(`Checking URL: ${tab.url}`);
    
    checkUrlWithAI(tab.url, tabId);
  }
});

async function checkUrlWithAI(url, tabId) {
  try {
    const response = await fetch(API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url: url }),
    });

    if (!response.ok) {
      console.error("Error from AI server:", response.status, response.statusText);
      
      try {
        const errorData = await response.json();
        console.error("Server error detail:", errorData);
      } catch (e) {
          
          console.error("Server response was not JSON.");
      }
      return; 
    }

    const data = await response.json();
    console.log(`AI Prediction for ${url}:`, data);

    // If the AI says it's malicious, block the page!
    if (data.is_malicious) {
      console.warn(`MALICIOUS URL DETECTED: ${url}`);
      
      // Save the analysis data so the warning page can read it
      // save the original URL the user was trying to go to.
      const analysisData = {
        blockedUrl: url,
        analysis: data 
      };
      //  chrome.storage.local which is preferred for extensions
      await chrome.storage.local.set({ 'lastBlockedAnalysis': analysisData });

      // Redirect to  warning page
      const warningPageUrl = chrome.runtime.getURL('warning.html');
      
      // Ensure the tab still exists before trying to update it
      chrome.tabs.get(tabId, (existingTab) => {
        if (chrome.runtime.lastError) {
          console.error("Tab does not exist:", chrome.runtime.lastError.message);
        } else if (existingTab) {
          chrome.tabs.update(tabId, { url: warningPageUrl });
        }
      });
    }

  } catch (error) {
    console.error("Cannot connect to SafeLink AI server or network error:", error);
  }
}