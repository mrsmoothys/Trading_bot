/**
 * Professional Trading Dashboard - JavaScript Helpers
 * Utility functions, event handlers, and interactive features
 * Based on TradingView/Binance/Coinbase Pro patterns
 */

(function() {
    'use strict';

    // ==========================================================================
    // Configuration & Constants
    // ==========================================================================

    const CONFIG = {
        ANIMATION_DURATION: 300,
        RETRY_ATTEMPTS: 3,
        RETRY_DELAY: 1000,
        UPDATE_INTERVAL: 30000, // 30 seconds
        STORAGE_KEYS: {
            SYMBOL: 'trading_dashboard_symbol',
            TIMEFRAME: 'trading_dashboard_timeframe',
            OVERLAYS: 'trading_dashboard_overlays',
            CHAT_HISTORY: 'trading_dashboard_chat_history'
        }
    };

    // ==========================================================================
    // DOM Utilities
    // ==========================================================================

    const DOM = {
        /**
         * Query single element
         */
        qs: (selector, parent = document) => parent.querySelector(selector),

        /**
         * Query multiple elements
         */
        qsa: (selector, parent = document) => Array.from(parent.querySelectorAll(selector)),

        /**
         * Check if element exists
         */
        exists: (selector) => DOM.qs(selector) !== null,

        /**
         * Create element with attributes
         */
        create: (tag, attrs = {}, children = []) => {
            const el = document.createElement(tag);

            Object.entries(attrs).forEach(([key, value]) => {
                if (key === 'className') {
                    el.className = value;
                } else if (key === 'text') {
                    el.textContent = value;
                } else if (key === 'html') {
                    el.innerHTML = value;
                } else {
                    el.setAttribute(key, value);
                }
            });

            children.forEach(child => {
                if (typeof child === 'string') {
                    el.appendChild(document.createTextNode(child));
                } else {
                    el.appendChild(child);
                }
            });

            return el;
        },

        /**
         * Add event listener with automatic cleanup
         */
        on: (element, event, handler, options) => {
            element.addEventListener(event, handler, options);
            return () => element.removeEventListener(event, handler, options);
        },

        /**
         * Toggle class with animation
         */
        toggleClass: (el, className, condition) => {
            if (condition === undefined) {
                condition = !el.classList.contains(className);
            }

            if (condition) {
                el.classList.add(className);
            } else {
                el.classList.remove(className);
            }

            return condition;
        }
    };

    // ==========================================================================
    // Local Storage Utilities
    // ==========================================================================

    const Storage = {
        /**
         * Get item from localStorage with JSON parsing
         */
        get: (key, defaultValue = null) => {
            try {
                const item = localStorage.getItem(key);
                return item ? JSON.parse(item) : defaultValue;
            } catch (e) {
                console.error('Storage get error:', e);
                return defaultValue;
            }
        },

        /**
         * Set item in localStorage with JSON stringification
         */
        set: (key, value) => {
            try {
                localStorage.setItem(key, JSON.stringify(value));
                return true;
            } catch (e) {
                console.error('Storage set error:', e);
                return false;
            }
        },

        /**
         * Remove item from localStorage
         */
        remove: (key) => {
            try {
                localStorage.removeItem(key);
                return true;
            } catch (e) {
                console.error('Storage remove error:', e);
                return false;
            }
        },

        /**
         * Clear all dashboard storage
         */
        clear: () => {
            Object.values(CONFIG.STORAGE_KEYS).forEach(key => {
                Storage.remove(key);
            });
        }
    };

    // ==========================================================================
    // Format Utilities
    // ==========================================================================

    const Format = {
        /**
         * Format price with appropriate decimal places
         */
        price: (value, decimals = 2) => {
            if (value >= 100000) {
                return new Intl.NumberFormat('en-US', {
                    style: 'currency',
                    currency: 'USD',
                    minimumFractionDigits: 0,
                    maximumFractionDigits: 0
                }).format(value);
            } else if (value >= 1) {
                return new Intl.NumberFormat('en-US', {
                    style: 'currency',
                    currency: 'USD',
                    minimumFractionDigits: decimals,
                    maximumFractionDigits: decimals
                }).format(value);
            } else {
                return value.toFixed(6);
            }
        },

        /**
         * Format percentage change
         */
        percentage: (value, decimals = 2) => {
            const sign = value >= 0 ? '+' : '';
            return `${sign}${value.toFixed(decimals)}%`;
        },

        /**
         * Format volume
         */
        volume: (value) => {
            if (value >= 1e9) {
                return `${(value / 1e9).toFixed(2)}B`;
            } else if (value >= 1e6) {
                return `${(value / 1e6).toFixed(2)}M`;
            } else if (value >= 1e3) {
                return `${(value / 1e3).toFixed(2)}K`;
            }
            return value.toFixed(2);
        },

        /**
         * Format timestamp
         */
        timestamp: (date = new Date()) => {
            return date.toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
        },

        /**
         * Format relative time (e.g., "5 minutes ago")
         */
        relativeTime: (timestamp) => {
            const now = Date.now();
            const diff = now - timestamp;
            const seconds = Math.floor(diff / 1000);
            const minutes = Math.floor(seconds / 60);
            const hours = Math.floor(minutes / 60);
            const days = Math.floor(hours / 24);

            if (days > 0) return `${days}d ago`;
            if (hours > 0) return `${hours}h ago`;
            if (minutes > 0) return `${minutes}m ago`;
            return 'just now';
        }
    };

    // ==========================================================================
    // Notification System
    // ==========================================================================

    const Toast = {
        /**
         * Show toast notification
         */
        show: (message, type = 'info', duration = 3000) => {
            const toast = DOM.create('div', {
                className: `toast toast-${type}`,
                html: `
                    <div class="toast-icon">
                        ${type === 'success' ? '✓' : type === 'error' ? '✗' : 'ℹ'}
                    </div>
                    <div class="toast-message">${message}</div>
                    <button class="toast-close">×</button>
                `
            });

            document.body.appendChild(toast);

            // Animate in
            setTimeout(() => toast.classList.add('show'), 10);

            // Auto remove
            const timeout = setTimeout(() => {
                Toast.hide(toast);
            }, duration);

            // Close button
            DOM.on(DOM.qs('.toast-close', toast), 'click', () => {
                clearTimeout(timeout);
                Toast.hide(toast);
            });

            return toast;
        },

        /**
         * Hide toast notification
         */
        hide: (toast) => {
            toast.classList.remove('show');
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        },

        /**
         * Show success message
         */
        success: (message, duration) => {
            return Toast.show(message, 'success', duration);
        },

        /**
         * Show error message
         */
        error: (message, duration) => {
            return Toast.show(message, 'error', duration);
        },

        /**
         * Show info message
         */
        info: (message, duration) => {
            return Toast.show(message, 'info', duration);
        }
    };

    // ==========================================================================
    // Chart Interaction Helpers
    // ==========================================================================

    const Chart = {
        /**
         * Update chart overlays
         */
        updateOverlays: (overlays) => {
            console.log('Updating overlays:', overlays);

            // Persist to storage
            Storage.set(CONFIG.STORAGE_KEYS.OVERLAYS, overlays);

            // Dispatch custom event for Dash callbacks
            const event = new CustomEvent('chart:overlaysUpdated', {
                detail: overlays
            });
            document.dispatchEvent(event);
        },

        /**
         * Switch timeframe
         */
        switchTimeframe: (timeframe) => {
            console.log('Switching timeframe:', timeframe);

            // Update UI
            DOM.qsa('.timeframe-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.tf === timeframe);
            });

            // Persist to storage
            Storage.set(CONFIG.STORAGE_KEYS.TIMEFRAME, timeframe);

            // Dispatch custom event
            const event = new CustomEvent('chart:timeframeChanged', {
                detail: { timeframe }
            });
            document.dispatchEvent(event);
        },

        /**
         * Switch symbol
         */
        switchSymbol: (symbol) => {
            console.log('Switching symbol:', symbol);

            // Persist to storage
            Storage.set(CONFIG.STORAGE_KEYS.SYMBOL, symbol);

            // Dispatch custom event
            const event = new CustomEvent('chart:symbolChanged', {
                detail: { symbol }
            });
            document.dispatchEvent(event);
        },

        /**
         * Get current chart state
         */
        getState: () => {
            return {
                symbol: Storage.get(CONFIG.STORAGE_KEYS.SYMBOL, 'BTCUSDT'),
                timeframe: Storage.get(CONFIG.STORAGE_KEYS.TIMEFRAME, '15m'),
                overlays: Storage.get(CONFIG.STORAGE_KEYS.OVERLAYS, {})
            };
        }
    };

    // ==========================================================================
    // Chat Interface Helpers
    // ==========================================================================

    const Chat = {
        /**
         * Send message
         */
        send: async (message) => {
            if (!message.trim()) return;

            // Add to history
            const history = Storage.get(CONFIG.STORAGE_KEYS.CHAT_HISTORY, []);
            history.push({
                sender: 'user',
                message,
                timestamp: Date.now()
            });
            Storage.set(CONFIG.STORAGE_KEYS.CHAT_HISTORY, history);

            // In a real implementation, this would call the DeepSeek API
            // For now, simulate response
            setTimeout(() => {
                const response = {
                    sender: 'ai',
                    message: `I analyzed your request: "${message}". Here's what I found...`,
                    timestamp: Date.now()
                };

                history.push(response);
                Storage.set(CONFIG.STORAGE_KEYS.CHAT_HISTORY, history);

                // Add to UI
                Chat.addMessage(response);
            }, 1000);
        },

        /**
         * Add message to chat
         */
        addMessage: (msg) => {
            const chatHistory = DOM.qs('#chat-history');
            if (!chatHistory) return;

            const messageEl = DOM.create('div', {
                className: `chat-message ${msg.sender}-message`
            }, [
                DOM.create('div', {
                    className: 'message-avatar'
                }, [msg.sender === 'ai' ? '🤖' : '👤']),
                DOM.create('div', {
                    className: 'message-content'
                }, [
                    DOM.create('div', {
                        className: 'message-header'
                    }, [
                        DOM.create('span', {
                            className: 'message-author',
                            text: msg.sender === 'ai' ? 'DeepSeek' : 'You'
                        }),
                        DOM.create('span', {
                            className: 'message-time',
                            text: Format.relativeTime(msg.timestamp)
                        })
                    ]),
                    DOM.create('div', {
                        className: 'message-text',
                        html: msg.message
                    })
                ])
            ]);

            chatHistory.appendChild(messageEl);
            chatHistory.scrollTop = chatHistory.scrollHeight;
        },

        /**
         * Load chat history
         */
        loadHistory: () => {
            const history = Storage.get(CONFIG.STORAGE_KEYS.CHAT_HISTORY, []);
            const chatHistory = DOM.qs('#chat-history');

            if (!chatHistory) return;

            chatHistory.innerHTML = '';

            history.forEach(msg => {
                Chat.addMessage(msg);
            });
        },

        /**
         * Clear chat history
         */
        clear: () => {
            Storage.remove(CONFIG.STORAGE_KEYS.CHAT_HISTORY);
            const chatHistory = DOM.qs('#chat-history');
            if (chatHistory) {
                chatHistory.innerHTML = '';
            }
        }
    };

    // ==========================================================================
    // Keyboard Shortcuts
    // ==========================================================================

    const Keyboard = {
        shortcuts: {},

        /**
         * Register keyboard shortcut
         */
        register: (key, callback) => {
            Keyboard.shortcuts[key] = callback;
        },

        /**
         * Handle keydown events
         */
        handle: (e) => {
            const key = [];

            if (e.ctrlKey) key.push('ctrl');
            if (e.shiftKey) key.push('shift');
            if (e.altKey) key.push('alt');

            // Add the main key
            let mainKey = e.key.toLowerCase();
            if (mainKey === ' ') mainKey = 'space';
            key.push(mainKey);

            const shortcut = key.join('+');

            if (Keyboard.shortcuts[shortcut]) {
                e.preventDefault();
                Keyboard.shortcuts[shortcut]();
            }
        },

        /**
         * Initialize default shortcuts
         */
        init: () => {
            // Ctrl+R - Refresh chart
            Keyboard.register('ctrl+r', () => {
                const event = new CustomEvent('chart:refresh');
                document.dispatchEvent(event);
                Toast.info('Chart refreshed');
            });

            // Ctrl+1-6 - Switch timeframes
            Keyboard.register('ctrl+1', () => Chart.switchTimeframe('1m'));
            Keyboard.register('ctrl+2', () => Chart.switchTimeframe('5m'));
            Keyboard.register('ctrl+3', () => Chart.switchTimeframe('15m'));
            Keyboard.register('ctrl+4', () => Chart.switchTimeframe('1h'));
            Keyboard.register('ctrl+5', () => Chart.switchTimeframe('4h'));
            Keyboard.register('ctrl+6', () => Chart.switchTimeframe('1d'));

            // ESC - Clear selection
            Keyboard.register('escape', () => {
                const event = new CustomEvent('chart:clearSelection');
                document.dispatchEvent(event);
            });

            document.addEventListener('keydown', Keyboard.handle);
        }
    };

    // ==========================================================================
    // Animation Helpers
    // ==========================================================================

    const Animate = {
        /**
         * Fade in element
         */
        fadeIn: (el, duration = CONFIG.ANIMATION_DURATION) => {
            el.style.opacity = '0';
            el.style.display = 'block';

            let start = null;

            function step(timestamp) {
                if (!start) start = timestamp;
                const progress = timestamp - start;

                el.style.opacity = Math.min(progress / duration, 1).toString();

                if (progress < duration) {
                    requestAnimationFrame(step);
                }
            }

            requestAnimationFrame(step);
        },

        /**
         * Fade out element
         */
        fadeOut: (el, duration = CONFIG.ANIMATION_DURATION) => {
            let start = null;

            function step(timestamp) {
                if (!start) start = timestamp;
                const progress = timestamp - start;

                el.style.opacity = Math.max(1 - (progress / duration), 0).toString();

                if (progress < duration) {
                    requestAnimationFrame(step);
                } else {
                    el.style.display = 'none';
                }
            }

            requestAnimationFrame(step);
        },

        /**
         * Slide toggle
         */
        slideToggle: (el, duration = CONFIG.ANIMATION_DURATION) => {
            if (el.style.display === 'none' || el.style.height === '0px') {
                Animate.slideDown(el, duration);
            } else {
                Animate.slideUp(el, duration);
            }
        },

        /**
         * Slide down
         */
        slideDown: (el, duration = CONFIG.ANIMATION_DURATION) => {
            el.style.height = '0px';
            el.style.overflow = 'hidden';
            el.style.display = 'block';

            const height = el.scrollHeight;
            let start = null;

            function step(timestamp) {
                if (!start) start = timestamp;
                const progress = timestamp - start;

                el.style.height = Math.min((progress / duration) * height, height) + 'px';

                if (progress < duration) {
                    requestAnimationFrame(step);
                } else {
                    el.style.height = '';
                    el.style.overflow = '';
                }
            }

            requestAnimationFrame(step);
        },

        /**
         * Slide up
         */
        slideUp: (el, duration = CONFIG.ANIMATION_DURATION) => {
            const height = el.scrollHeight;
            el.style.height = height + 'px';
            el.style.overflow = 'hidden';

            let start = null;

            function step(timestamp) {
                if (!start) start = timestamp;
                const progress = timestamp - start;

                el.style.height = Math.max(height - (progress / duration) * height, 0) + 'px';

                if (progress < duration) {
                    requestAnimationFrame(step);
                } else {
                    el.style.display = 'none';
                    el.style.height = '';
                    el.style.overflow = '';
                }
            }

            requestAnimationFrame(step);
        }
    };

    // ==========================================================================
    // API Utilities
    // ==========================================================================

    const API = {
        /**
         * Fetch with retry
         */
        fetch: async (url, options = {}, attempts = CONFIG.RETRY_ATTEMPTS) => {
            for (let i = 0; i < attempts; i++) {
                try {
                    const response = await fetch(url, options);

                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }

                    return await response.json();
                } catch (error) {
                    console.error(`Attempt ${i + 1} failed:`, error);

                    if (i === attempts - 1) {
                        throw error;
                    }

                    await new Promise(resolve => setTimeout(resolve, CONFIG.RETRY_DELAY * (i + 1)));
                }
            }
        },

        /**
         * POST with JSON
         */
        post: async (url, data) => {
            return API.fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
        },

        /**
         * GET request
         */
        get: async (url) => {
            return API.fetch(url);
        }
    };

    // ==========================================================================
    // Event System
    // ==========================================================================

    const Events = {
        /**
         * Emit custom event
         */
        emit: (name, detail = {}) => {
            const event = new CustomEvent(name, { detail });
            document.dispatchEvent(event);
        },

        /**
         * Listen to event once
         */
        once: (name, callback) => {
            const handler = (e) => {
                callback(e.detail);
                document.removeEventListener(name, handler);
            };
            document.addEventListener(name, handler);
        }
    };

    // ==========================================================================
    // Initialization
    // ==========================================================================

    function init() {
        console.log('Initializing Trading Dashboard JavaScript...');

        // Load saved state
        const state = Chart.getState();
        console.log('Loaded state:', state);

        // Initialize keyboard shortcuts
        Keyboard.init();

        // Load chat history
        Chat.loadHistory();

        // Set up overlay toggles
        DOM.qsa('.toggle-input').forEach(toggle => {
            DOM.on(toggle, 'change', function() {
                const overlayName = this.id.replace('toggle-', '');
                const overlays = Storage.get(CONFIG.STORAGE_KEYS.OVERLAYS, {});
                overlays[overlayName] = this.checked;
                Chart.updateOverlays(overlays);
            });

            // Load saved state
            const overlays = Storage.get(CONFIG.STORAGE_KEYS.OVERLAYS, {});
            const overlayName = toggle.id.replace('toggle-', '');
            if (overlays.hasOwnProperty(overlayName)) {
                toggle.checked = overlays[overlayName];
            }
        });

        // Set up timeframe buttons
        DOM.qsa('.timeframe-btn').forEach(btn => {
            DOM.on(btn, 'click', function() {
                Chart.switchTimeframe(this.dataset.tf);
            });
        });

        // Set up symbol dropdown
        const symbolDropdown = DOM.qs('#symbol-select');
        if (symbolDropdown) {
            DOM.on(symbolDropdown, 'change', function() {
                Chart.switchSymbol(this.value);
            });
        }

        // Set up chat input
        const chatInput = DOM.qs('#chat-input');
        const chatSend = DOM.qs('#chat-send');

        if (chatInput && chatSend) {
            DOM.on(chatSend, 'click', () => {
                Chat.send(chatInput.value);
                chatInput.value = '';
                chatInput.focus();
            });

            DOM.on(chatInput, 'keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    Chat.send(this.value);
                    this.value = '';
                }
            });
        }

        // Set up chat clear button
        const clearChatBtn = DOM.qs('.chat-action-btn[title="Clear Chat"]');
        if (clearChatBtn) {
            DOM.on(clearChatBtn, 'click', () => {
                if (confirm('Clear chat history?')) {
                    Chat.clear();
                }
            });
        }

        // Listen for chart events
        document.addEventListener('chart:overlaysUpdated', (e) => {
            console.log('Overlays updated:', e.detail);
        });

        document.addEventListener('chart:timeframeChanged', (e) => {
            console.log('Timeframe changed:', e.detail);
        });

        document.addEventListener('chart:symbolChanged', (e) => {
            console.log('Symbol changed:', e.detail);
        });

        console.log('✅ Trading Dashboard JavaScript initialized');
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Export to global scope for Dash callbacks
    window.TradingDashboard = {
        DOM,
        Storage,
        Format,
        Toast,
        Chart,
        Chat,
        Keyboard,
        Animate,
        API,
        Events,
        CONFIG
    };

})();