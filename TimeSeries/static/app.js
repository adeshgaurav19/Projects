let dataColumns = [];

// Base URL for backend server
const baseUrl = 'http://127.0.0.1:8000'; 

async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${baseUrl}/upload_file/`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('File upload failed');
            }

            const data = await response.json();
            console.log('Upload Response:', data);
            document.getElementById('uploadMessage').innerText = data.message;
            updateColumns(data.columns);  // Assuming 'columns' is returned with file
        } catch (error) {
            console.error('Upload Error:', error);
            document.getElementById('uploadMessage').innerText = `Error uploading file: ${error.message}`;
        }
    } else {
        console.warn('No file selected for upload');
        document.getElementById('uploadMessage').innerText = 'Please select a file to upload';
    }
}


async function fetchFinanceData() {
    const ticker = document.getElementById('predefinedStocks').value;
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;

    const data = new URLSearchParams({
        ticker: ticker,
        start_date: startDate,
        end_date: endDate
    });

    try {
        const response = await fetch("http://127.0.0.1:8000/fetch_finance_data/", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',  // Ensure the correct content type
            },
            body: data
        });

        const result = await response.json();
        console.log('Fetch Finance Data Response:', result);

        if (response.ok) {
            document.getElementById('fetchMessage').innerText = result.message;
            await fetchAndDisplayPlot(ticker);
        } else {
            document.getElementById('fetchMessage').innerText = `Error: ${result.detail}`;
        }
        updateColumns(result.columns);
    } catch (error) {
        console.error('Fetch or plot error:', error);
        document.getElementById('fetchMessage').innerText = `Network error: ${error.message}`;
    }
}


async function fetchAndDisplayPlot(ticker) {
    try {
        const plotData = await fetch(`${baseUrl}/eda/?option=line_plot&column=Close`);
        const plotResult = await plotData.json();
        console.log('Plot Data Response:', plotResult);

        if (plotData.ok) {
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
    const columnSelect = document.getElementById('columnName');
    const targetSelect = document.getElementById('targetColumn');
    
    if (columnSelect && targetSelect) {
        columnSelect.innerHTML = '';
        targetSelect.innerHTML = '';
        
        columns.forEach(column => {
            const option = document.createElement('option');
            option.text = column;
            option.value = column;
            columnSelect.add(option);

            const targetOption = option.cloneNode(true);
            targetSelect.add(targetOption);
        });
    } else {
        console.error('Column or target select elements not found in DOM');
    }
}

async function performEDA() {
    const option = document.getElementById('edaOption').value;
    const column = document.getElementById('columnName').value;
    const data = new URLSearchParams({ option, column });

    try {
        const response = await fetch(`${baseUrl}/eda/`, {
            method: 'POST',
            body: data
        });
        const result = await response.json();
        console.log('EDA Response:', result);

        if (result.image) {
            document.getElementById('edaResult').innerHTML = `<img src="data:image/png;base64,${result.image}" alt="EDA Result">`;
        } else if (result.summary) {
            document.getElementById('edaResult').innerText = JSON.stringify(result.summary, null, 2);
        }
    } catch (error) {
        console.error('EDA Error:', error);
    }
}


async function selectModel() {
    const model = document.getElementById('modelSelect').value;
    const targetColumn = document.getElementById('targetColumn').value;
    const params = {}; // You can dynamically set model parameters based on user input

    const data = new URLSearchParams({ model, params: JSON.stringify(params), target_column: targetColumn });

    try {
        const response = await fetch(`${baseUrl}/model_selection/`, {
            method: 'POST',
            body: data
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
    }
}
