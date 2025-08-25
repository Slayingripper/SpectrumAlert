// Additional functions for dataset and model selection GUI

async function analyzeSelectedData() {
    try {
        const selectedFiles = document.querySelectorAll('.file-item.selected');
        if (selectedFiles.length === 0) {
            showNotification('Please select at least one data file to analyze', 'warning');
            return;
        }
        
        const filenames = Array.from(selectedFiles).map(item => 
            item.querySelector('.file-name').textContent
        );
        
        showNotification(`Analyzing ${filenames.length} selected file(s)...`, 'info');
        
        // For now, just analyze the first selected file
        const filename = filenames[0];
        await analyzeDataFile(filename);
        
    } catch (error) {
        console.error('Error analyzing selected data:', error);
        showNotification('Error analyzing data: ' + error.message, 'error');
    }
}

// Enhanced loadDataFiles function to support selection
async function loadDataFilesWithSelection() {
    try {
        const response = await fetch('/api/data/files');
        if (response.ok) {
            const result = await response.json();
            if (result.status === 'ok' && result.data) {
                displayDataFilesWithSelection(result.data.files);
                window.dataFilesLoaded = true;
            }
        }
    } catch (error) {
        console.error('Error loading data files:', error);
        showNotification('Error loading data files', 'error');
    }
}

function displayDataFilesWithSelection(files) {
    const filesList = document.getElementById('data-file-list');
    if (!filesList) {
        console.error('data-file-list element not found');
        return;
    }
    
    if (files.length === 0) {
        filesList.innerHTML = '<div class="no-files" style="text-align: center; color: #888; padding: 20px;">No data files found. Start monitoring to collect data.</div>';
        return;
    }
    
    let filesHTML = '<div class="files-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 10px;">';
    
    files.slice(0, 10).forEach(file => {
        const fileTypeIcon = file.type === 'csv' ? 'fa-file-csv' : 'fa-file-code';
        const freqInfo = file.frequency_range ? 
            `${(file.frequency_range.min_hz / 1e6).toFixed(1)}-${(file.frequency_range.max_hz / 1e6).toFixed(1)} MHz` : 
            'Unknown freq';
        
        filesHTML += `
            <div class="file-item" onclick="toggleFileSelection(this)" style="
                background: #2a2a2a; border: 1px solid #555; border-radius: 5px; 
                padding: 10px; cursor: pointer; transition: all 0.3s;
            " onmouseover="this.style.borderColor='#00ff88'" onmouseout="this.style.borderColor='#555'">
                <div class="file-icon" style="text-align: center; font-size: 24px; color: #00ff88; margin-bottom: 5px;">
                    <i class="fas ${fileTypeIcon}"></i>
                </div>
                <div class="file-info">
                    <div class="file-name" style="color: #fff; font-weight: bold; margin-bottom: 5px;">${file.filename}</div>
                    <div class="file-details" style="color: #888; font-size: 12px;">
                        <div>Size: ${file.size_mb} MB</div>
                        <div>Samples: ${file.samples.toLocaleString()}</div>
                        <div>Freq: ${freqInfo}</div>
                    </div>
                    <div class="file-date" style="color: #666; font-size: 11px; margin-top: 5px;">
                        ${new Date(file.created_at).toLocaleDateString()}
                    </div>
                </div>
            </div>
        `;
    });
    
    filesHTML += '</div>';
    
    if (files.length > 10) {
        filesHTML += `<div style="text-align: center; margin-top: 10px; color: #888;">
            Showing 10 of ${files.length} files
        </div>`;
    }
    
    filesList.innerHTML = filesHTML;
}

function toggleFileSelection(element) {
    element.classList.toggle('selected');
    if (element.classList.contains('selected')) {
        element.style.borderColor = '#ffaa00';
        element.style.backgroundColor = '#3a3a2a';
    } else {
        element.style.borderColor = '#555';
        element.style.backgroundColor = '#2a2a2a';
    }
}

// Update the loadDataFiles function to call the new implementation
if (typeof window !== 'undefined') {
    window.originalLoadDataFiles = window.loadDataFiles || loadDataFiles;
    window.loadDataFiles = loadDataFilesWithSelection;
}
