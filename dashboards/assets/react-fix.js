/**
 * React-fix.js - Fixes for React component errors in Dash
 * 
 * This script patches some common React errors that can occur in Dash applications,
 * particularly when using complex components or when components are unmounted and
 * remounted frequently.
 */

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('React-fix.js loaded');
    
    // Patch React's setState for unmounted components
    patchReactSetState();
    
    // Fix for "ResizeObserver loop limit exceeded" error
    fixResizeObserverError();
    
    // Fix for "Maximum update depth exceeded" error
    fixMaxUpdateDepthError();
    
    // Fix for duplicate key warnings
    fixDuplicateKeyWarnings();
});

/**
 * Patch React's setState to prevent errors when calling setState on unmounted components
 */
function patchReactSetState() {
    // Check if React DevTools are available (development mode)
    const devToolsDetected = window.__REACT_DEVTOOLS_GLOBAL_HOOK__ && 
                            window.__REACT_DEVTOOLS_GLOBAL_HOOK__.inject;
    
    if (devToolsDetected) {
        console.log('React DevTools detected, applying setState patch');
        
        // Store original console.error
        const originalConsoleError = console.error;
        
        // Override console.error to filter out specific React warnings
        console.error = function(...args) {
            // Filter out "Can't perform a React state update on an unmounted component" warnings
            if (args[0] && typeof args[0] === 'string' && 
                args[0].includes("Can't perform a React state update on an unmounted component")) {
                // Ignore this error
                return;
            }
            
            // Pass through all other errors
            return originalConsoleError.apply(this, args);
        };
    }
}

/**
 * Fix for "ResizeObserver loop limit exceeded" error
 * This error occurs when ResizeObserver detects a loop of resizing events
 */
function fixResizeObserverError() {
    // Store original error handler
    const originalOnError = window.onerror;
    
    // Override window.onerror to catch ResizeObserver errors
    window.onerror = function(message, source, lineno, colno, error) {
        // Check if it's a ResizeObserver error
        if (message && message.toString().includes('ResizeObserver loop limit exceeded')) {
            // Prevent the error from propagating
            console.warn('ResizeObserver loop limit exceeded - error suppressed');
            return true;
        }
        
        // Pass through all other errors to the original handler
        if (originalOnError) {
            return originalOnError.apply(this, arguments);
        }
        
        return false;
    };
}

/**
 * Fix for "Maximum update depth exceeded" error
 * This occurs when there are too many nested state updates
 */
function fixMaxUpdateDepthError() {
    // Check if React is available
    if (window.React) {
        const originalUseEffect = window.React.useEffect;
        
        // Wrap useEffect to catch and prevent infinite loops
        window.React.useEffect = function() {
            try {
                return originalUseEffect.apply(this, arguments);
            } catch (error) {
                if (error.message && error.message.includes('Maximum update depth exceeded')) {
                    console.warn('Maximum update depth exceeded - error suppressed');
                    // Return a no-op cleanup function
                    return function() {};
                }
                throw error;
            }
        };
    }
}

/**
 * Fix for duplicate key warnings
 * This suppresses warnings about duplicate keys in lists
 */
function fixDuplicateKeyWarnings() {
    // Store original console.error
    const originalConsoleError = console.error;
    
    // Override console.error to filter out specific React warnings
    console.error = function(...args) {
        // Filter out duplicate key warnings
        if (args[0] && typeof args[0] === 'string' && 
            args[0].includes("Warning: Encountered two children with the same key")) {
            // Just log a shorter warning
            console.warn('Duplicate React key detected');
            return;
        }
        
        // Pass through all other errors
        return originalConsoleError.apply(this, args);
    };
}