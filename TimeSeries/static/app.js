let dataColumns = [];

// Base URL for backend server
const baseUrl = 'http://127.0.0.1:8000';

// Helper function for creating elements
function createOptionElement(value, text) {
    const option = document.createElement('option');
    option.value = value;
    option.text = text;
    return option;
}

// Helper for showing a loading state
function showLoading(elementId, message = 'Loading...') {
    document.getElementById(elementId).innerHTML = `<div class="loading">${message}</div>`;
}

// Check if data is available
function checkDataAvailability() {
    return dataColumns.length > 0;
}

// Update UI based on data availability
function updateUIForDataAvailability() {
    const edaButton = document.getElementById('edaButton');
    if (edaButton) {
        edaButton.disabled = !checkDataAvailability();
    }
}

async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput.files.length === 0) {
        console.warn('No file selected for upload');
        document.getElementById('uploadMessage').innerText = 'Please select a file to upload';
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        showLoading('uploadMessage');
        const response = await fetch(`${baseUrl}/upload_file/`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Upload Response:', data);
        document.getElementById('uploadMessage').innerText = data.message;
        updateColumns(data.columns);
        updateUIForDataAvailability();
    } catch (error) {
        console.error('Upload Error:', error);
        document.getElementById('uploadMessage').innerText = `Error uploading file: ${error.message}`;
    }
}

async function fetchFinanceData() {
    const ticker = document.getElementById('predefinedStocks').value;
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;

    const params = new URLSearchParams({ ticker, start_date: startDate, end_date: endDate });

    try {
        showLoading('fetchMessage');
        console.log('Sending fetch request to:', `${baseUrl}/fetch_finance_data/`);
        console.log('With parameters:', params.toString());
        
        const response = await fetch(`${baseUrl}/fetch_finance_data/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: params.toString()
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log('Fetch Finance Data Response:', result);

        if (response.ok) {
            document.getElementById('fetchMessage').innerText = result.message;
            await fetchAndDisplayPlot(ticker);
            updateColumns(result.columns);
            updateUIForDataAvailability();
        } else {
            document.getElementById('fetchMessage').innerText = `Error: ${result.detail || 'Unknown error'}`;
        }
    } catch (error) {
        console.error('Fetch or plot error:', error);
        document.getElementById('fetchMessage').innerText = `Network error: ${error.message}`;
    }
}

async function fetchAndDisplayPlot(ticker) {
    try {
        showLoading('plotArea', 'Fetching plot...');
        const response = await fetch(`${baseUrl}/eda/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({ option: 'line_plot', column: 'Close' })
        });

        const plotResult = await response.json();
        console.log('Plot Data Response:', plotResult);

        if (response.ok && plotResult.image) {
            document.getElementById('plotArea').innerHTML = `<img src="data:image/png;base64,${plotResult.image}" alt="Plot for ${ticker}">`;
        } else {
            document.getElementById('plotArea').innerText = 'Error in plotting data';
        }
    } catch (error) {
        console.error('Plotting error:', error);
        document.getElementById('plotArea').innerText = 'Failed to fetch plot data';
    }
}

function updateColumns(columns) {
    dataColumns = columns;
    ['columnName', 'targetColumn'].forEach(id => {
        const select = document.getElementById(id);
        if (select) {
            select.innerHTML = '';
            columns.forEach(column => select.add(createOptionElement(column, column)));
        } else {
            console.error(`Element with id ${id} not found in DOM`);
        }
    });
    updateUIForDataAvailability();
}

async function performEDA() {
    const option = document.getElementById('edaOption').value;
    const column = document.getElementById('columnName').value;
    const data = new URLSearchParams({ option, column });

    try {
        showLoading('edaResult');
        const response = await fetch(`${baseUrl}/eda/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: data.toString()
        });

        const result = await response.json();
        console.log('EDA Response:', result);

        if (response.ok) {
            if (result.image) {
                document.getElementById('edaResult').innerHTML = `<img src="data:image/png;base64,${result.image}" alt="EDA Result">`;
            } else if (result.summary) {
                document.getElementById('edaResult').innerText = JSON.stringify(result.summary, null, 2);
            } else {
                document.getElementById('edaResult').innerText = 'No data available for EDA';
            }
        } else {
            // Handle specific error messages from the server
            if (result.detail) {
                document.getElementById('edaResult').innerText = result.detail;
            } else {
                document.getElementById('edaResult').innerText = 'An error occurred during EDA';
            }
        }
    } catch (error) {
        console.error('EDA Error:', error);
        document.getElementById('edaResult').innerText = 'Error performing EDA';
    }
}

async function selectModel() {
    const model = document.getElementById('modelSelect').value;
    const targetColumn = document.getElementById('targetColumn').value;
    const params = {}; // Placeholder for dynamic parameters

    const data = new URLSearchParams({ model, params: JSON.stringify(params), target_column: targetColumn });

    try {
        showLoading('modelResult');
        const response = await fetch(`${baseUrl}/model_selection/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: data.toString()
        });
        const result = await response.json();
        console.log('Model Selection Response:', result);

        let resultHTML = '';
        for (let [key, value] of Object.entries(result)) {
            if (key === 'image') {
                resultHTML += `<img src="data:image/png;base64,${value}" alt="Model Result">`;
            } else if (key === 'summary') {
                resultHTML += `<pre>${value}</pre>`;
            } else {
                resultHTML += `<p>${key}: ${typeof value === 'object' ? JSON.stringify(value) : value}</p>`;
            }
        }
        document.getElementById('modelResult').innerHTML = resultHTML;
    } catch (error) {
        console.error('Model Selection Error:', error);
        document.getElementById('modelResult').innerText = 'Error selecting model';
    }
}