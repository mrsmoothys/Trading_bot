# DeepSeek Trading Dashboard - QA Checklist

## Overview
This checklist ensures all features and components of the DeepSeek Trading Dashboard are working correctly.

## Pre-Test Setup

### System Requirements
- [ ] Python 3.12+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Dashboard accessible at http://127.0.0.1:8050
- [ ] Browser console shows no critical errors

### Environment Variables
- [ ] DEEPSEEK_API_KEY configured (if using AI features)
- [ ] BINANCE_API_KEY configured (if using live data)
- [ ] All config files present and valid

---

## 1. Dashboard Loading & Layout

### Initial Load
- [ ] Dashboard loads within 3 seconds
- [ ] No 404 errors in browser console
- [ ] All static assets (CSS, JS) load successfully
- [ ] Dark theme applied correctly
- [ ] Header displays "DeepSeek Trading Dashboard"

### Layout Components
- [ ] Header section visible with title and subtitle
- [ ] Control panel section present
- [ ] Metrics cards row displays correctly
- [ ] Price chart renders with proper dimensions
- [ ] Chat interface panel visible
- [ ] Positions table section present
- [ ] System health panel visible

---

## 2. Chart Functionality

### Price Chart
- [ ] Candlestick chart displays OHLCV data
- [ ] Price axis (left) labeled correctly
- [ ] Volume subplot (bottom) displays bars
- [ ] Time axis shows timestamps
- [ ] Zoom controls work (mouse wheel, select box)
- [ ] Pan controls work (click and drag)
- [ ] Reset view button works

### Timeframe Selection
- [ ] 1m button changes chart data
- [ ] 5m button changes chart data
- [ ] 15m button changes chart data (default active)
- [ ] 1h button changes chart data
- [ ] 4h button changes chart data
- [ ] 1d button changes chart data
- [ ] Selected timeframe highlighted in green
- [ ] Chart updates within 2 seconds of selection

### Range Selector/Slider
- [ ] Range selector buttons (50, 100, 200, All) work
- [ ] Range slider at bottom functional
- [ ] Can select different data ranges
- [ ] View updates correctly
- [ ] Visible on all timeframe selections

### Signal Annotations
- [ ] LONG signals annotated with green markers
- [ ] SHORT signals annotated with red markers
- [ ] Arrows point to correct price levels
- [ ] Labels clearly visible
- [ ] Annotations don't obstruct price data

---

## 3. Chart Overlays

### Feature Toggle Panel
- [ ] All 6 checkboxes present and clickable:
  - [ ] 🔵 Liquidity Zones
  - [ ] ⚡ Supertrend
  - [ ] 📊 Chandelier Exit
  - [ ] 📈 Order Flow
  - [ ] 🎯 Market Regime
  - [ ] 🔄 Alignment
- [ ] Order Flow checked by default
- [ ] Clicking checkbox updates chart immediately

### Overlay: Liquidity Zones
- [ ] Toggle ON: Yellow vertical zones appear
- [ ] Zones are semi-transparent
- [ ] Zones span entire chart height
- [ ] Toggle OFF: Zones disappear
- [ ] No error when insufficient data

### Overlay: Supertrend
- [ ] Toggle ON: Blue line overlays price
- [ ] Line follows price action
- [ ] Line changes direction at trend reversals
- [ ] Toggle OFF: Line disappears
- [ ] Compatible with other overlays

### Overlay: Chandelier Exit
- [ ] Toggle ON: Two dashed lines appear (long/short exit)
- [ ] Orange colored lines
- [ ] Lines above/below current price
- [ ] Toggle OFF: Lines disappear
- [ ] Compatible with other overlays

### Overlay: Order Flow
- [ ] Toggle ON: Bottom subplot shows bars
- [ ] Green bars for buying pressure
- [ ] Red bars for selling pressure
- [ ] Toggle OFF: Subplot hides
- [ ] Always enabled by default

### Overlay: Market Regime
- [ ] Toggle ON: Background shading appears
- [ ] Different colors for different regimes:
  - Green for trending up
  - Red for trending down
  - Orange/yellow for ranging
- [ ] Regime label displays in top-left
- [ ] Toggle OFF: Shading disappears

### Overlay: Timeframe Alignment
- [ ] Toggle ON: Triangle markers appear on price
- [ ] Green triangles for bullish alignment
- [ ] Red triangles for bearish alignment
- [ ] Markers don't obstruct candlesticks
- [ ] Toggle OFF: Markers disappear

### Overlay Combinations
- [ ] All 6 overlays can be enabled simultaneously
- [ ] Performance acceptable (updates < 3 seconds)
- [ ] No visual conflicts between overlays
- [ ] Chart remains readable
- [ ] Disabling individual overlays works correctly

---

## 4. Chat Interface

### Chat Panel Layout
- [ ] Chat panel visible in dashboard
- [ ] "DeepSeek AI Chat" header present
- [ ] Chat history area displays messages
- [ ] Text input area present
- [ ] "Send Message" button visible
- [ ] "Clear Chat" button visible

### Quick Action Buttons
- [ ] 8 quick action buttons present:
  - [ ] 📊 Analyze Performance
  - [ ] 🎯 Market Analysis
  - [ ] ⚠️ Risk Assessment
  - [ ] 🔧 System Optimization
  - [ ] 📈 Feature Performance
  - [ ] 💰 Close All Positions
  - [ ] ⏸️ Pause Trading
  - [ ] ▶️ Resume Trading
- [ ] Buttons clickable
- [ ] Generate appropriate messages

### Message Functionality
- [ ] Type message in text area
- [ ] Click "Send Message"
- [ ] Message appears in chat history
- [ ] AI response appears (or demo message)
- [ ] Messages scroll automatically
- [ ] "Clear Chat" button clears history

### Chat-to-Chart Interaction
- [ ] Chat messages can reference overlays
- [ ] Keywords recognized: "liquidity", "supertrend", "trend", "chandelier", etc.
- [ ] Highlight store updates when overlays mentioned
- [ ] No errors when chat commands fail

---

## 5. System Controls

### Refresh Button
- [ ] "🔄 Refresh Data" button present
- [ ] Clicking button triggers data refresh
- [ ] Loading indicator appears briefly
- [ ] Chart updates with fresh data

### Emergency Stop
- [ ] "⛔ Emergency Stop" button present
- [ ] Button colored red
- [ ] Clicking button changes color to orange
- [ ] Clicking again returns to red
- [ ] No errors when clicked

---

## 6. Metrics & Positions

### Metrics Cards
- [ ] 4 metrics cards display correctly:
  - [ ] Total P&L
  - [ ] Win Rate
  - [ ] Active Positions
  - [ ] Current Drawdown
- [ ] Values update when positions change
- [ ] Colors appropriate (green for positive, red for negative)

### Active Positions Table
- [ ] Table displays when positions exist
- [ ] Table empty when no positions
- [ ] Columns present: Symbol, Side, Entry Price, Current Price, Size, P&L, P&L %, Actions
- [ ] "Close" button on each row
- [ ] Button styled correctly (red)
- [ ] Clicking close button works

---

## 7. System Health

### Health Panel
- [ ] Panel displays at bottom of dashboard
- [ ] "System Health & Performance" header
- [ ] Shows Memory Usage (MB)
- [ ] Shows CPU Usage (%)
- [ ] Shows Last Update timestamp
- [ ] Shows Data Connection status

### Memory & CPU
- [ ] Memory usage displays realistic values
- [ ] CPU usage displays realistic values
- [ ] Values update periodically
- [ ] No memory leaks visible

---

## 8. Data & Performance

### Data Sources
- [ ] Sample data displays correctly
- [ ] Real-time data (if configured) updates
- [ ] No data errors in console
- [ ] Cache working (subsequent loads faster)

### Chart Performance
- [ ] Initial chart load < 3 seconds
- [ ] Timeframe switch < 2 seconds
- [ ] Feature toggle < 1 second
- [ ] Smooth scrolling and zooming
- [ ] No lag with 200+ data points

### Browser Compatibility
- [ ] Chrome: All features work
- [ ] Firefox: All features work
- [ ] Safari: All features work
- [ ] Edge: All features work

---

## 9. Responsive Design

### Screen Sizes
- [ ] Desktop (1920x1080): Full layout
- [ ] Laptop (1366x768): Layout adapts
- [ ] Tablet (768x1024): Columns stack
- [ ] Mobile (375x667): Usable interface

### UI Elements
- [ ] Buttons remain clickable at all sizes
- [ ] Text remains readable
- [ ] Chart scales appropriately
- [ ] Chat panel accessible on mobile
- [ ] No horizontal scroll bars (except chart)

---

## 10. Edge Cases & Error Handling

### Empty States
- [ ] No data: Chart shows placeholder
- [ ] No positions: "No active positions" message
- [ ] No connection: Graceful fallback to sample data

### Error States
- [ ] Invalid timeframe: Default to 15m
- [ ] Invalid symbol: Fallback to BTCUSDT
- [ ] API failure: Display error message
- [ ] JavaScript error: Log to console, don't crash

### Loading States
- [ ] Data loading: Show visual indicator
- [ ] Chart updating: Disable controls briefly
- [ ] Server offline: Show connection error

---

## 11. DeepSeek AI Integration (Optional)

### AI Chat
- [ ] DeepSeek AI responses (when API key configured)
- [ ] Context-aware responses
- [ ] Response time < 5 seconds
- [ ] Error handling for API failures

### System Context
- [ ] AI receives system context
- [ ] Overlay state shared with AI
- [ ] Market regime information available
- [ ] Feature performance metrics accessible

---

## 12. Database & Logging

### Signal Logging
- [ ] Signals save to database (when configured)
- [ ] Overlay metadata included
- [ ] Audit logging works
- [ ] Chat history saved

### Performance Monitoring
- [ ] System health tracked
- [ ] Memory usage logged
- [ ] Error tracking active
- [ ] Performance metrics collected

---

## Test Execution Notes

### Test Environment
- **URL**: http://127.0.0.1:8050
- **Browser**: Chrome (latest)
- **Screen Resolution**: 1920x1080
- **Date**: ___________

### Test Results Summary
- **Total Checks**: 150+
- **Passed**: _______
- **Failed**: _______
- **Not Tested**: _______
- **Tester Signature**: ___________

### Known Issues
List any issues discovered during testing:
1. ________________________________
2. ________________________________
3. ________________________________

### Recommendations
1. ________________________________
2. ________________________________
3. ________________________________

---

## Automated Test Results

### Unit Tests
```bash
# Run unit tests
python -m pytest tests/test_plotly_overlays.py -v
```
- [ ] All 24 Plotly overlay tests pass

### Integration Tests
```bash
# Run all tests
python -m pytest tests/ -v
```
- [ ] All tests pass
- [ ] No critical failures
- [ ] Coverage acceptable (>80%)

### Performance Tests
```bash
# Run performance tests
python -m pytest tests/ -m "not slow" --tb=short
```
- [ ] Tests complete in < 60 seconds
- [ ] No timeout errors
- [ ] Memory usage stable

---

## Final Sign-off

### Development Team
- [ ] Code review completed
- [ ] All tests passing
- [ ] Documentation updated

### QA Team
- [ ] Manual testing completed
- [ ] All critical issues resolved
- [ ] Performance acceptable
- [ ] Ready for release

### Product Owner
- [ ] Features match requirements
- [ ] User experience acceptable
- [ ] Approved for deployment

**Release Decision**: [ ] APPROVED [ ] REJECTED

**Comments**: _________________________________________

---

## Appendix

### Keyboard Shortcuts
- Ctrl+R: Refresh chart
- Ctrl+1: Switch to 1m timeframe
- Ctrl+2: Switch to 5m timeframe
- Ctrl+3: Switch to 15m timeframe
- Ctrl+4: Switch to 1h timeframe
- Ctrl+5: Switch to 4h timeframe
- Ctrl+6: Switch to 1d timeframe
- ESC: Clear selection

### Browser Console Commands
```javascript
// Check chart overlays state
console.log(window.dash_clientside);

// Test chat interaction
console.log(document.querySelector('[data-dash-is-loading="true"]'));

// Check for errors
console.log(performance.getEntriesByType('navigation'));
```

### Common Issues & Solutions

**Issue**: Chart doesn't update when switching timeframes
**Solution**: Check browser console for JavaScript errors; verify data source

**Issue**: Overlays not showing
**Solution**: Ensure checkboxes are checked; verify feature calculation functions

**Issue**: Chat interface not responding
**Solution**: Check if DeepSeek API key is configured; verify backend connection

**Issue**: Performance issues
**Solution**: Reduce data points; enable caching; check system resources

---

*Last Updated: 2025-11-15*
*Version: 1.0*
*Owner: QA Team*
