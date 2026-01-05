// ============================================
// GLOBAL VARIABLES
// ============================================
let isDetecting = false;
let statsInterval = null;

// ============================================
// INITIALIZATION
// ============================================
document.addEventListener('DOMContentLoaded', function() {
    updateTime();
    setInterval(updateTime, 1000);
});

// ============================================
// TIME UPDATE
// ============================================
function updateTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit' 
    });
    const timeElement = document.getElementById('currentTime');
    if (timeElement) {
        timeElement.textContent = timeString;
    }
}

// ============================================
// START DETECTION
// ============================================
function startDetection() {
    // Disable button immediately to prevent double-clicks
    document.getElementById('startBtn').disabled = true;
    
    fetch('/start_detection')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'started') {
                isDetecting = true;
                
                // Update UI
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('systemStatus').innerHTML = 
                    '<i class="fas fa-circle pulse" style="color: #28a745;"></i> Detection Active';
                
                // Show video feed with a small delay to ensure backend is ready
                setTimeout(() => {
                    const videoFeed = document.getElementById('videoFeed');
                    const videoOverlay = document.getElementById('videoOverlay');
                    // Add timestamp to prevent caching
                    videoFeed.src = '/video_feed?t=' + new Date().getTime();
                    videoFeed.classList.add('active');
                    videoOverlay.classList.add('hidden');
                }, 500);
                
                // Start stats polling
                statsInterval = setInterval(updateStats, 1000);
                
                // Add activity
                addActivity('Detection started', 'success');
                
                // Show notification
                showNotification('Detection Started', 'Camera is now active', 'success');
            }
        })
        .catch(error => {
            console.error('Error starting detection:', error);
            showNotification('Error', 'Failed to start detection', 'error');
            document.getElementById('startBtn').disabled = false;
        });
}

// ============================================
// STOP DETECTION
// ============================================
function stopDetection() {
    // Disable button immediately
    document.getElementById('stopBtn').disabled = true;
    
    fetch('/stop_detection')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'stopped') {
                isDetecting = false;
                
                // Update UI
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('systemStatus').innerHTML = 
                    '<i class="fas fa-circle"></i> System Ready';
                
                // Hide video feed
                const videoFeed = document.getElementById('videoFeed');
                const videoOverlay = document.getElementById('videoOverlay');
                videoFeed.src = '';
                videoFeed.classList.remove('active');
                videoOverlay.classList.remove('hidden');
                
                // Stop stats polling
                if (statsInterval) {
                    clearInterval(statsInterval);
                    statsInterval = null;
                }
                
                // Add activity
                addActivity('Detection stopped', 'info');
                
                // Show notification
                showNotification('Detection Stopped', 'Camera has been deactivated', 'info');
            }
        })
        .catch(error => {
            console.error('Error stopping detection:', error);
            showNotification('Error', 'Failed to stop detection', 'error');
        });
}

// ============================================
// UPDATE STATISTICS
// ============================================
function updateStats() {
    fetch('/get_stats')
        .then(response => response.json())
        .then(data => {
            // Update stat numbers with animation
            animateNumber('totalDetections', data.total_detections);
            animateNumber('maskCount', data.mask_count);
            animateNumber('noMaskCount', data.no_mask_count);
            
            // Update current status
            const statusDisplay = document.getElementById('currentStatus');
            const status = data.current_status;
            
            if (status.includes('Mask Detected')) {
                statusDisplay.innerHTML = '<i class="fas fa-check-circle"></i><span>Mask Detected</span>';
                statusDisplay.className = 'status-display active-mask';
            } else if (status.includes('No Mask')) {
                statusDisplay.innerHTML = '<i class="fas fa-exclamation-triangle"></i><span>No Mask Warning</span>';
                statusDisplay.className = 'status-display active-nomask';
            } else {
                statusDisplay.innerHTML = '<i class="fas fa-pause-circle"></i><span>Inactive</span>';
                statusDisplay.className = 'status-display';
            }
        })
        .catch(error => {
            console.error('Error updating stats:', error);
        });
}

// ============================================
// ANIMATE NUMBER
// ============================================
function animateNumber(elementId, targetValue) {
    const element = document.getElementById(elementId);
    const currentValue = parseInt(element.textContent) || 0;
    
    if (currentValue !== targetValue) {
        element.textContent = targetValue;
        element.style.transform = 'scale(1.2)';
        setTimeout(() => {
            element.style.transform = 'scale(1)';
        }, 200);
    }
}

// ============================================
// RESET STATISTICS
// ============================================
function resetStats() {
    if (confirm('Are you sure you want to reset all statistics?')) {
        fetch('/reset_stats')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'reset') {
                    // Update UI immediately
                    document.getElementById('totalDetections').textContent = '0';
                    document.getElementById('maskCount').textContent = '0';
                    document.getElementById('noMaskCount').textContent = '0';
                    
                    // Add activity
                    addActivity('Statistics reset', 'warning');
                    
                    // Show notification
                    showNotification('Stats Reset', 'All statistics have been cleared', 'success');
                }
            })
            .catch(error => {
                console.error('Error resetting stats:', error);
                showNotification('Error', 'Failed to reset statistics', 'error');
            });
    }
}

// ============================================
// ADD ACTIVITY TO LIST
// ============================================
function addActivity(message, type) {
    const activityList = document.getElementById('activityList');
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
    
    const activityItem = document.createElement('li');
    activityItem.className = 'activity-item';
    
    let iconColor = '#667eea';
    if (type === 'success') iconColor = '#28a745';
    else if (type === 'warning') iconColor = '#ffc107';
    else if (type === 'error') iconColor = '#dc3545';
    
    activityItem.innerHTML = `
        <i class="fas fa-circle activity-dot" style="color: ${iconColor};"></i>
        <span>${message}</span>
        <span class="activity-time">${timeString}</span>
    `;
    
    // Add to top of list
    activityList.insertBefore(activityItem, activityList.firstChild);
    
    // Keep only last 10 items
    while (activityList.children.length > 10) {
        activityList.removeChild(activityList.lastChild);
    }
}

// ============================================
// SHOW NOTIFICATION (SIMPLE VERSION)
// ============================================
function showNotification(title, message, type) {
    // Simple console notification - can be enhanced with a toast library
    console.log(`[${type.toUpperCase()}] ${title}: ${message}`);
    
    // You can add a toast notification library here like:
    // - Toastify
    // - SweetAlert2
    // - Bootstrap Toast
}

// ============================================
// SMOOTH TRANSITIONS
// ============================================
document.querySelectorAll('.stat-number').forEach(element => {
    element.style.transition = 'transform 0.2s ease';
});
