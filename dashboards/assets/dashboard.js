/**
 * JavaScript for EntropicUnification Dashboard
 */

// Prevent conflicts with React's event system
window.dashboardInitialized = false;

// Wait for the document to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Avoid duplicate initialization
    if (window.dashboardInitialized) return;
    window.dashboardInitialized = true;
    
    // Wait for React to finish rendering
    setTimeout(function() {
        // Initialize theme from localStorage
        initializeTheme();
        
        // Initialize settings panel
        initializeSettingsPanel();
        
        // Add event listeners for theme switching
        setupThemeSwitching();
        
        // Add event listeners for plot download buttons
        setupPlotDownloadButtons();
        
        // Add tooltips
        setupTooltips();
    }, 500); // Give React time to render
});

/**
 * Initialize theme from localStorage
 */
function initializeTheme() {
    // Check if dark mode is enabled in localStorage
    const darkMode = localStorage.getItem('darkMode') === 'true';
    
    // Apply dark mode if enabled
    if (darkMode) {
        document.body.classList.add('dark-mode');
        
        // Update dark mode switch if it exists
        const darkModeSwitch = document.getElementById('dark-mode-switch');
        if (darkModeSwitch) {
            darkModeSwitch.checked = true;
        }
    }
    
    // Check if color theme is set in localStorage
    const colorTheme = localStorage.getItem('colorTheme');
    if (colorTheme) {
        // Update color theme dropdown if it exists
        const colorThemeDropdown = document.getElementById('color-theme-dropdown');
        if (colorThemeDropdown) {
            colorThemeDropdown.value = colorTheme;
        }
    }
}

/**
 * Initialize settings panel
 */
function initializeSettingsPanel() {
    // Get settings toggle button
    const settingsToggle = document.getElementById('settings-toggle');
    if (!settingsToggle) return;
    
    // Get settings panel
    const settingsPanel = document.getElementById('settings-panel');
    if (!settingsPanel) return;
    
    // Add click event listener to toggle settings panel
    settingsToggle.addEventListener('click', function() {
        settingsPanel.classList.toggle('open');
    });
    
    // Close settings panel when clicking outside
    document.addEventListener('click', function(event) {
        if (!settingsPanel.contains(event.target) && event.target !== settingsToggle) {
            settingsPanel.classList.remove('open');
        }
    });
}

/**
 * Set up theme switching
 */
function setupThemeSwitching() {
    // Get dark mode switch
    const darkModeSwitch = document.getElementById('dark-mode-switch');
    if (darkModeSwitch) {
        // Add change event listener
        darkModeSwitch.addEventListener('change', function() {
            // Toggle dark mode class on body
            document.body.classList.toggle('dark-mode', this.checked);
            
            // Save preference to localStorage
            localStorage.setItem('darkMode', this.checked);
            
            // Update plot styles
            updatePlotStyles(this.checked);
        });
    }
    
    // Get color theme dropdown
    const colorThemeDropdown = document.getElementById('color-theme-dropdown');
    if (colorThemeDropdown) {
        // Add change event listener
        colorThemeDropdown.addEventListener('change', function() {
            // Save preference to localStorage
            localStorage.setItem('colorTheme', this.value);
            
            // Reload page to apply new theme
            // In a real implementation, you would use Dash callbacks to update the theme
            // without reloading the page
            location.reload();
        });
    }
}

/**
 * Update plot styles based on dark mode
 */
function updatePlotStyles(darkMode) {
    // Get all plot style dropdowns
    const plotStyleDropdown = document.getElementById('plot-style-dropdown');
    if (plotStyleDropdown) {
        // Set appropriate plot style based on dark mode
        plotStyleDropdown.value = darkMode ? 'plotly_dark' : 'plotly_white';
        
        // Trigger change event to update plots
        const event = new Event('change');
        plotStyleDropdown.dispatchEvent(event);
    }
}

/**
 * Set up plot download buttons
 */
function setupPlotDownloadButtons() {
    // Get all plot download buttons
    const downloadButtons = document.querySelectorAll('[id^="download-plot-"]');
    
    // Add click event listener to each button
    downloadButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Get the plot ID from the button ID
            const plotId = this.id.replace('download-', '');
            
            // Get the plot
            const plot = document.getElementById(plotId);
            if (!plot) return;
            
            // Get the download format
            const formatDropdown = document.getElementById('download-format-dropdown');
            const format = formatDropdown ? formatDropdown.value : 'png';
            
            // Trigger download
            downloadPlot(plot, format);
        });
    });
}

/**
 * Download a plot
 */
function downloadPlot(plot, format) {
    // In a real implementation, you would use Plotly's toImage function
    // For now, we'll just log a message
    console.log(`Downloading plot ${plot.id} as ${format}`);
    
    // Trigger Plotly's download functionality
    if (plot && plot.data && plot._context) {
        const downloadButton = plot.querySelector('.modebar-btn[data-title="Download plot as a png"]');
        if (downloadButton) {
            downloadButton.click();
        }
    }
}

/**
 * Set up tooltips
 */
function setupTooltips() {
    // Get help tooltips switch
    const helpTooltipsSwitch = document.getElementById('help-tooltips-switch');
    if (helpTooltipsSwitch) {
        // Add change event listener
        helpTooltipsSwitch.addEventListener('change', function() {
            // Get all help icons
            const helpIcons = document.querySelectorAll('.help-icon');
            
            // Show or hide help icons based on switch state
            helpIcons.forEach(icon => {
                icon.style.display = this.checked ? 'inline' : 'none';
            });
            
            // Save preference to localStorage
            localStorage.setItem('helpTooltips', this.checked);
        });
        
        // Initialize from localStorage
        const helpTooltips = localStorage.getItem('helpTooltips') !== 'false';
        helpTooltipsSwitch.checked = helpTooltips;
        
        // Trigger change event
        const event = new Event('change');
        helpTooltipsSwitch.dispatchEvent(event);
    }
}
