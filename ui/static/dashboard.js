/**
 * DeepSeek Trading Dashboard - Professional JavaScript
 * Enhanced interactivity and animations
 */

class DashboardEnhancer {
  constructor() {
    this.chatHistoryObserver = null;
    this.chatHistoryContainerObserver = null;
    this.chatHistoryElement = null;
    this.chatInputObserver = null;
    this.chartAxisObserver = null;
    this.boundPriceAxisScroll = this.handlePriceAxisScroll.bind(this);
    this.init();
  }

  init() {
    this.setupEventListeners();
    this.animateElements();
    this.startPeriodicUpdates();
    this.setupTooltips();
    this.setupKeyboardShortcuts();
    this.setupResponsiveFeatures();
    this.setupChartAxisControls();
  }

  /**
   * Setup enhanced event listeners
   */
  setupEventListeners() {
    // Animate buttons on click
    document.querySelectorAll('.btn').forEach(btn => {
      btn.addEventListener('click', function(e) {
        // Create ripple effect
        const ripple = document.createElement('span');
        ripple.classList.add('ripple');
        this.appendChild(ripple);

        setTimeout(() => {
          ripple.remove();
        }, 600);
      });
    });

    // Chat UX helpers
    this.observeChatScroll();
    this.setupChatInputInteractions();

    // Add hover effects to metric cards
    document.querySelectorAll('.metric-card').forEach(card => {
      card.addEventListener('mouseenter', () => {
        card.style.transform = 'translateY(-5px) scale(1.02)';
      });

      card.addEventListener('mouseleave', () => {
        card.style.transform = 'translateY(0) scale(1)';
      });
    });

    // Handle price updates with animation
    this.observePriceChanges();
  }

  /**
   * Animate elements on page load
   */
  animateElements() {
    // Fade in sections
    const sections = document.querySelectorAll('.section');
    sections.forEach((section, index) => {
      section.style.opacity = '0';
      section.style.transform = 'translateY(20px)';
      section.style.transition = 'all 0.5s ease';

      setTimeout(() => {
        section.style.opacity = '1';
        section.style.transform = 'translateY(0)';
      }, index * 100);
    });

    // Stagger metric cards
    const metricCards = document.querySelectorAll('.metric-card');
    metricCards.forEach((card, index) => {
      card.style.opacity = '0';
      card.style.transform = 'scale(0.9)';
      card.style.transition = 'all 0.4s ease';

      setTimeout(() => {
        card.style.opacity = '1';
        card.style.transform = 'scale(1)';
      }, 200 + (index * 50));
    });
  }

  /**
   * Observe price changes and animate them
   */
  observePriceChanges() {
    const priceElement = document.querySelector('#live-price');
    if (!priceElement) return;

    let lastPrice = null;

    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'childList') {
          const newPrice = priceElement.textContent;
          if (newPrice !== lastPrice) {
            this.animatePriceChange(newPrice, lastPrice);
            lastPrice = newPrice;
          }
        }
      });
    });

    observer.observe(priceElement, {
      childList: true,
      subtree: true
    });
  }

  /**
   * Animate price changes
   */
  animatePriceChange(newPrice, oldPrice) {
    const priceElement = document.querySelector('#live-price');
    if (!priceElement) return;

    priceElement.style.transition = 'all 0.3s ease';
    priceElement.style.transform = 'scale(1.1)';

    setTimeout(() => {
      priceElement.style.transform = 'scale(1)';
    }, 300);
  }

  /**
   * Observe chat scrolling
   */
  observeChatScroll() {
    const attachObserver = () => {
      const chatHistory = document.querySelector('#chat-history');
      if (!chatHistory || this.chatHistoryElement === chatHistory) {
        return;
      }

      if (this.chatHistoryObserver) {
        this.chatHistoryObserver.disconnect();
      }

      this.chatHistoryElement = chatHistory;

      const scrollToBottom = () => {
        window.requestAnimationFrame(() => {
          chatHistory.scrollTop = chatHistory.scrollHeight;
        });
      };

      scrollToBottom();

      this.chatHistoryObserver = new MutationObserver(() => {
        scrollToBottom();
      });

      this.chatHistoryObserver.observe(chatHistory, {
        childList: true,
        subtree: true
      });
    };

    attachObserver();

    if (this.chatHistoryContainerObserver) {
      this.chatHistoryContainerObserver.disconnect();
    }

    this.chatHistoryContainerObserver = new MutationObserver(() => {
      attachObserver();
    });

    this.chatHistoryContainerObserver.observe(document.body, {
      childList: true,
      subtree: true
    });
  }

  /**
   * Ensure Enter submits chat messages and keeps focus synced
   */
  setupChatInputInteractions() {
    const attachHandlers = () => {
      const chatInput = document.querySelector('#chat-input');
      if (!chatInput || chatInput.dataset.listenersAttached === 'true') {
        return;
      }

      chatInput.dataset.listenersAttached = 'true';

      chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          const sendBtn = document.querySelector('#chat-send-button');
          const hasValue = chatInput.value && chatInput.value.trim().length > 0;
          if (sendBtn && !sendBtn.disabled && hasValue) {
            e.preventDefault();
            sendBtn.click();
          }
        }
      });

      chatInput.addEventListener('focus', () => {
        const chatHistory = this.chatHistoryElement || document.querySelector('#chat-history');
        if (chatHistory) {
          chatHistory.scrollTop = chatHistory.scrollHeight;
        }
      });
    };

    attachHandlers();

    if (this.chatInputObserver) {
      this.chatInputObserver.disconnect();
    }

    this.chatInputObserver = new MutationObserver(() => {
      attachHandlers();
    });

    this.chatInputObserver.observe(document.body, {
      childList: true,
      subtree: true
    });
  }

  /**
   * Attach scroll/drag helpers for TradingView-style price axis control
   */
  setupChartAxisControls() {
    const attach = () => {
      const chart = document.getElementById('price-chart');
      if (!chart || chart.dataset.axisControlsAttached === 'true') {
        return;
      }

      chart.dataset.axisControlsAttached = 'true';
      chart.addEventListener('wheel', this.boundPriceAxisScroll, { passive: false });
    };

    attach();

    if (this.chartAxisObserver) {
      this.chartAxisObserver.disconnect();
    }

    this.chartAxisObserver = new MutationObserver(() => {
      attach();
    });

    this.chartAxisObserver.observe(document.body, {
      childList: true,
      subtree: true
    });
  }

  /**
   * Handle mouse wheel interactions near the price axis
   */
  handlePriceAxisScroll(event) {
    const chart = document.getElementById('price-chart');
    if (!chart || !window.Plotly || !chart._fullLayout) {
      return;
    }

    const rect = chart.getBoundingClientRect();
    const offsetX = event.clientX - rect.left;

    // Only intercept when user is close to the right edge (price axis area)
    const axisZoneWidth = rect.width * 0.15;
    const nearRightAxis = offsetX >= rect.width - axisZoneWidth;

    if (!nearRightAxis) {
      return;
    }

    event.preventDefault();

    const layout = chart._fullLayout;
    const priceAxis = layout.yaxis;
    if (!priceAxis || !priceAxis.range) {
      return;
    }

    const [currentMin, currentMax] = priceAxis.range;
    const span = currentMax - currentMin;
    if (!isFinite(span) || span <= 0) {
      return;
    }

    let newMin;
    let newMax;

    if (event.shiftKey) {
      // Shift + scroll translates the axis up/down
      const moveRatio = 0.05;
      const direction = event.deltaY > 0 ? 1 : -1;
      const delta = span * moveRatio * direction;
      newMin = currentMin + delta;
      newMax = currentMax + delta;
    } else {
      // Default scroll zooms price axis in/out
      const zoomFactor = event.deltaY > 0 ? 1.1 : 0.9;
      const center = (currentMin + currentMax) / 2;
      const newHalf = (span * zoomFactor) / 2;
      newMin = center - newHalf;
      newMax = center + newHalf;
    }

    window.Plotly.relayout(chart, {
      'yaxis.range': [newMin, newMax]
    });
  }

  /**
   * Setup tooltips for elements
   */
  setupTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');

    tooltipElements.forEach(element => {
      let tooltip = null;

      element.addEventListener('mouseenter', (e) => {
        tooltip = document.createElement('div');
        tooltip.className = 'custom-tooltip';
        tooltip.textContent = element.getAttribute('data-tooltip');
        document.body.appendChild(tooltip);

        const rect = element.getBoundingClientRect();
        tooltip.style.left = `${rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2)}px`;
        tooltip.style.top = `${rect.top - tooltip.offsetHeight - 10}px`;
        tooltip.style.opacity = '0';
        tooltip.style.transition = 'opacity 0.2s ease';

        requestAnimationFrame(() => {
          tooltip.style.opacity = '1';
        });
      });

      element.addEventListener('mouseleave', () => {
        if (tooltip) {
          tooltip.style.opacity = '0';
          setTimeout(() => {
            tooltip.remove();
          }, 200);
        }
      });
    });
  }

  /**
   * Setup keyboard shortcuts
   */
  setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      // Ctrl/Cmd + R: Refresh data
      if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
        e.preventDefault();
        const refreshBtn = document.querySelector('#refresh-btn');
        if (refreshBtn) {
          refreshBtn.click();
          this.showNotification('Data refreshed', 'success');
        }
      }

      // Ctrl/Cmd + 1-6: Quick timeframe selection
      if ((e.ctrlKey || e.metaKey) && e.key >= '1' && e.key <= '6') {
        e.preventDefault();
        const timeframes = ['tf-1m', 'tf-5m', 'tf-15m', 'tf-1h', 'tf-4h', 'tf-1d'];
        const btn = document.querySelector(`#${timeframes[parseInt(e.key) - 1]}`);
        if (btn) {
          btn.click();
        }
      }

      // Escape: Clear chat input
      if (e.key === 'Escape') {
        const chatInput = document.querySelector('#chat-input');
        if (chatInput) {
          chatInput.blur();
        }
      }
    });
  }

  /**
   * Setup responsive features
   */
  setupResponsiveFeatures() {
    // Handle window resize
    window.addEventListener('resize', () => {
      this.handleResize();
    });

    // Initial resize handling
    this.handleResize();
  }

  /**
   * Handle window resize
   */
  handleResize() {
    const width = window.innerWidth;

    // Adjust layout based on screen size
    if (width < 768) {
      document.body.classList.add('mobile-view');
    } else {
      document.body.classList.remove('mobile-view');
    }
  }

  /**
   * Start periodic updates
   */
  startPeriodicUpdates() {
    // Update system time every second
    setInterval(() => {
      this.updateSystemTime();
    }, 1000);

    // Check for connection status
    setInterval(() => {
      this.checkConnectionStatus();
    }, 5000);
  }

  /**
   * Update system time display
   */
  updateSystemTime() {
    const now = new Date();
    const timeString = now.toLocaleString();
    const timeElements = document.querySelectorAll('.system-time');

    timeElements.forEach(element => {
      element.textContent = timeString;
    });
  }

  /**
   * Check connection status
   */
  checkConnectionStatus() {
    // Simulate connection check
    const isOnline = navigator.onLine;
    const statusElement = document.querySelector('.connection-status');

    if (statusElement) {
      statusElement.textContent = isOnline ? 'Online' : 'Offline';
      statusElement.className = `connection-status ${isOnline ? 'online' : 'offline'}`;
    }
  }

  /**
   * Show notification
   */
  showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.padding = '15px 20px';
    notification.style.backgroundColor = type === 'success' ? '#00ff88' :
                                        type === 'error' ? '#ff4444' :
                                        type === 'warning' ? '#ff9900' : '#4488ff';
    notification.style.color = type === 'success' || type === 'error' || type === 'warning' ? '#000' : '#fff';
    notification.style.borderRadius = '8px';
    notification.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.5)';
    notification.style.zIndex = '10000';
    notification.style.animation = 'slideIn 0.3s ease';

    document.body.appendChild(notification);

    setTimeout(() => {
      notification.style.animation = 'slideOut 0.3s ease';
      setTimeout(() => {
        notification.remove();
      }, 300);
    }, 3000);
  }

  /**
   * Format numbers with commas
   */
  formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  /**
   * Format currency
   */
  formatCurrency(amount, decimals = 2) {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    }).format(amount);
  }

  /**
   * Format percentage
   */
  formatPercentage(value, decimals = 2) {
    return `${value.toFixed(decimals)}%`;
  }
}

/**
 * Chart Enhancer
 */
class ChartEnhancer {
  constructor() {
    this.init();
  }

  init() {
    this.setupChartInteractions();
  }

  /**
   * Setup chart interactions
   */
  setupChartInteractions() {
    // Listen for chart updates
    const chartContainer = document.querySelector('#price-chart');
    if (!chartContainer) return;

    const observer = new MutationObserver(() => {
      this.enhanceChartElements();
    });

    observer.observe(chartContainer, {
      childList: true,
      subtree: true
    });
  }

  /**
   * Enhance chart elements
   */
  enhanceChartElements() {
    // Add hover effects to chart traces
    const traces = document.querySelectorAll('.trace');
    traces.forEach(trace => {
      trace.addEventListener('mouseenter', () => {
        trace.style.opacity = '0.8';
        trace.style.transition = 'opacity 0.2s ease';
      });

      trace.addEventListener('mouseleave', () => {
        trace.style.opacity = '1';
      });
    });
  }
}

/**
 * Table Enhancer
 */
class TableEnhancer {
  constructor() {
    this.init();
  }

  init() {
    this.enhanceTables();
  }

  /**
   * Enhance tables with sorting and highlighting
   */
  enhanceTables() {
    const tables = document.querySelectorAll('.positions-table');
    tables.forEach(table => {
      const headers = table.querySelectorAll('th');
      headers.forEach((header, index) => {
        header.style.cursor = 'pointer';
        header.addEventListener('click', () => {
          this.sortTable(table, index);
        });
      });
    });
  }

  /**
   * Sort table by column
   */
  sortTable(table, columnIndex) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));

    const isNumeric = !isNaN(rows[0].children[columnIndex].textContent.replace(/[^0-9.-]/g, ''));

    rows.sort((a, b) => {
      const aVal = a.children[columnIndex].textContent.trim();
      const bVal = b.children[columnIndex].textContent.trim();

      if (isNumeric) {
        return parseFloat(aVal) - parseFloat(bVal);
      } else {
        return aVal.localeCompare(bVal);
      }
    });

    // Clear tbody
    tbody.innerHTML = '';

    // Add sorted rows
    rows.forEach(row => {
      tbody.appendChild(row);
    });

    // Add animation
    rows.forEach((row, index) => {
      row.style.opacity = '0';
      row.style.transform = 'translateY(10px)';
      setTimeout(() => {
        row.style.transition = 'all 0.3s ease';
        row.style.opacity = '1';
        row.style.transform = 'translateY(0)';
      }, index * 50);
    });
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  // Add CSS animations
  const style = document.createElement('style');
  style.textContent = `
    @keyframes slideIn {
      from {
        transform: translateX(100%);
        opacity: 0;
      }
      to {
        transform: translateX(0);
        opacity: 1;
      }
    }

    @keyframes slideOut {
      from {
        transform: translateX(0);
        opacity: 1;
      }
      to {
        transform: translateX(100%);
        opacity: 0;
      }
    }

    .ripple {
      position: absolute;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.3);
      pointer-events: none;
      transform: scale(0);
      animation: ripple-animation 0.6s linear;
    }

    @keyframes ripple-animation {
      to {
        transform: scale(4);
        opacity: 0;
      }
    }

    .custom-tooltip {
      position: absolute;
      background-color: rgba(0, 0, 0, 0.9);
      color: white;
      padding: 8px 12px;
      border-radius: 4px;
      font-size: 12px;
      pointer-events: none;
      z-index: 10000;
      white-space: nowrap;
    }

    .custom-tooltip::after {
      content: '';
      position: absolute;
      top: 100%;
      left: 50%;
      margin-left: -5px;
      border-width: 5px;
      border-style: solid;
      border-color: rgba(0, 0, 0, 0.9) transparent transparent transparent;
    }

    .notification {
      font-weight: 600;
      animation: slideIn 0.3s ease;
    }

    @media (max-width: 768px) {
      .mobile-view .metric-card:hover {
        transform: none !important;
      }
    }
  `;
  document.head.appendChild(style);

  // Initialize enhancers
  window.dashboardEnhancer = new DashboardEnhancer();
  window.chartEnhancer = new ChartEnhancer();
  window.tableEnhancer = new TableEnhancer();

  console.log('DeepSeek Trading Dashboard - Professional Mode Enabled ✓');
});
