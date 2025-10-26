document.addEventListener('DOMContentLoaded', async () => {
    try {
        // Retrieve the full analysis from storage
        const result = await chrome.storage.local.get('lastBlockedAnalysis');
        
        if (result.lastBlockedAnalysis) {
            const data = result.lastBlockedAnalysis;
            
            // 1. Display the blocked URL
            document.getElementById('blocked-url').textContent = data.blockedUrl || 'No URL specified';

            // 2. Populate the threat report
            const reportList = document.getElementById('threat-report-list');
            reportList.innerHTML = ''; // Clear previous entries if any
            
            if (data.analysis && data.analysis.threat_report && data.analysis.threat_report.length > 0) {
                data.analysis.threat_report.forEach(reason => {
                    const li = document.createElement('li');
                    li.textContent = reason;
                    reportList.appendChild(li);
                });
            } else {
                // Fallback if the report is empty for some reason
                const li = document.createElement('li');
                li.textContent = "Matches a general malicious URL pattern.";
                reportList.appendChild(li);
            }
            
            // 3. Clear the data from storage so it's not shown again on reload
            await chrome.storage.local.remove('lastBlockedAnalysis');

        } else {
            document.getElementById('blocked-url').textContent = 'No specific URL data found.';
            document.getElementById('threat-report-list').innerHTML = '<li>No analysis data found.</li>';
        }
    } catch (error) {
         console.error("Error retrieving or processing analysis data:", error);
         document.getElementById('blocked-url').textContent = 'Error loading data.';
         document.getElementById('threat-report-list').innerHTML = '<li>Error loading threat details.</li>';
    }
});

// Make the "Go Back" button work
document.getElementById('go-back').addEventListener('click', () => {
    // Check if history is available to go back
    if (window.history.length > 1) {
         window.history.back();
    } else {
         // If no history, maybe close the tab or provide a safe link
         alert("No previous page in history to go back to.");
    }
});

