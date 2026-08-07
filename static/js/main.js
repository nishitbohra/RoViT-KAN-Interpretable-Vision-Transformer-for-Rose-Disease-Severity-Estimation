/**
 * Main JavaScript for RoViT-KAN Web Interface
 * Handles UI interactions and HTMX events
 */

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('🌹 RoViT-KAN Web Interface loaded');
    
    // Initialize all components
    initDragAndDrop();
    initHTMXEvents();
    initTooltips();
});

/**
 * Initialize drag and drop functionality
 */
function initDragAndDrop() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    
    if (!dropZone || !fileInput) return;
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop zone
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });
    
    // Handle drop
    dropZone.addEventListener('drop', handleDrop, false);
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight(e) {
        dropZone.classList.add('dragover');
    }
    
    function unhighlight(e) {
        dropZone.classList.remove('dragover');
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            previewImage(fileInput);
        }
    }
}

/**
 * Preview uploaded image
 */
function previewImage(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        const previewImg = document.getElementById('preview-img');
        const imagePreview = document.getElementById('image-preview');
        const dropZone = document.getElementById('drop-zone');
        const submitBtn = document.getElementById('submit-btn');
        
        reader.onload = function(e) {
            if (previewImg) previewImg.src = e.target.result;
            if (imagePreview) imagePreview.classList.remove('hidden');
            if (dropZone) dropZone.classList.add('hidden');
            if (submitBtn) submitBtn.disabled = false;
        };
        
        reader.readAsDataURL(input.files[0]);
    }
}

/**
 * Clear image preview
 */
function clearPreview() {
    const fileInput = document.getElementById('file-input');
    const previewImg = document.getElementById('preview-img');
    const imagePreview = document.getElementById('image-preview');
    const dropZone = document.getElementById('drop-zone');
    const submitBtn = document.getElementById('submit-btn');
    
    if (fileInput) fileInput.value = '';
    if (previewImg) previewImg.src = '';
    if (imagePreview) imagePreview.classList.add('hidden');
    if (dropZone) dropZone.classList.remove('hidden');
    if (submitBtn) submitBtn.disabled = true;
}

/**
 * Initialize HTMX event handlers
 */
function initHTMXEvents() {
    // Before request - show loading state
    document.body.addEventListener('htmx:beforeRequest', function(evt) {
        console.log('🔄 Sending request...');
        const submitBtn = document.getElementById('submit-btn');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.textContent = 'Analyzing...';
        }
    });
    
    // After request - reset button
    document.body.addEventListener('htmx:afterRequest', function(evt) {
        console.log('✅ Request completed');
        const submitBtn = document.getElementById('submit-btn');
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Analyze Image';
        }
        
        // Scroll to results on mobile
        if (window.innerWidth < 1024) {
            const resultsContainer = document.getElementById('results-container');
            if (resultsContainer) {
                resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }
    });
    
    // On error
    document.body.addEventListener('htmx:responseError', function(evt) {
        console.error('❌ Request failed:', evt.detail);
        const submitBtn = document.getElementById('submit-btn');
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Analyze Image';
        }
    });
    
    // After swap - initialize charts if needed
    document.body.addEventListener('htmx:afterSwap', function(evt) {
        if (evt.detail.target.id === 'results-container') {
            console.log('🎨 Results loaded, initializing visualizations...');
            initializeCharts();
        }
    });
}

/**
 * Initialize tooltips
 */
function initTooltips() {
    // Simple tooltip implementation
    const tooltips = document.querySelectorAll('[data-tooltip]');
    
    tooltips.forEach(el => {
        el.classList.add('tooltip');
    });
}

/**
 * Initialize charts (if Chart.js is available)
 */
function initializeCharts() {
    // Example: Initialize probability chart if element exists
    const ctx = document.getElementById('probability-chart');
    if (ctx && typeof Chart !== 'undefined') {
        // Charts would be initialized here
        // This is a placeholder for future enhancements
    }
}

/**
 * Toggle raw heatmaps visibility
 */
function toggleHeatmaps() {
    const heatmaps = document.getElementById('raw-heatmaps');
    const button = event.target;
    
    if (heatmaps) {
        heatmaps.classList.toggle('hidden');
        button.textContent = heatmaps.classList.contains('hidden') 
            ? 'Show Raw Heatmaps ▼' 
            : 'Hide Raw Heatmaps ▲';
    }
}

/**
 * Utility: Format percentage
 */
function formatPercent(value) {
    return (value * 100).toFixed(2) + '%';
}

/**
 * Utility: Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Utility: Throttle function
 */
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Export functions for global access
window.previewImage = previewImage;
window.clearPreview = clearPreview;
window.toggleHeatmaps = toggleHeatmaps;
window.formatPercent = formatPercent;
