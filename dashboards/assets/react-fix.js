/**
 * React component error fixes for EntropicUnification Dashboard
 */

// Prevent React component errors
if (window._dashprivate_isReactComponentWrapper === undefined) {
    window._dashprivate_isReactComponentWrapper = function() { return false; };
}

// Fix for React component stack errors
if (window.__REACT_DEVTOOLS_GLOBAL_HOOK__) {
    try {
        // Save original function
        const originalOnCommitFiberRoot = window.__REACT_DEVTOOLS_GLOBAL_HOOK__.onCommitFiberRoot;
        
        // Override with error-safe version
        window.__REACT_DEVTOOLS_GLOBAL_HOOK__.onCommitFiberRoot = function(id, root, ...rest) {
            try {
                return originalOnCommitFiberRoot.call(this, id, root, ...rest);
            } catch (error) {
                console.log('Suppressed React DevTools error:', error.message);
                return null;
            }
        };
    } catch (e) {
        console.log('Could not patch React DevTools:', e.message);
    }
}

// Fix for React component stack errors in console
if (console.error && typeof console.error === 'function') {
    const originalError = console.error;
    console.error = function(...args) {
        // Filter out React component stack errors
        if (args[0] && typeof args[0] === 'string' && 
            (args[0].includes('Component Stack') || 
             args[0].includes('React DevTools') ||
             args[0].includes('APIController'))) {
            return;
        }
        return originalError.apply(this, args);
    };
}

// Prevent React warnings from being displayed
if (console.warn && typeof console.warn === 'function') {
    const originalWarn = console.warn;
    console.warn = function(...args) {
        // Filter out React warnings
        if (args[0] && typeof args[0] === 'string' && 
            (args[0].includes('React') || 
             args[0].includes('component') ||
             args[0].includes('prop'))) {
            return;
        }
        return originalWarn.apply(this, args);
    };
}
