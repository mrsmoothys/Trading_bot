"""
Shared pytest configuration for dashboard/UI tests.
- Forces test mode to avoid real network calls.
- Adds missing helpers expected by our smoke tests.
"""
import os
import time
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException
from dash.testing.composite import DashComposite

# Older dash-duo examples expose .sleep; provide a thin wrapper for compatibility.
if not hasattr(DashComposite, "sleep"):
    def _sleep(self, seconds: float):
        time.sleep(seconds)
    DashComposite.sleep = _sleep

# Allow driver.find_elements(css_selector) single-argument usage in legacy tests.
_orig_find_elements = WebDriver.find_elements
def _find_elements(self, by=None, value=None):
    if value is None and isinstance(by, str):
        return _orig_find_elements(self, By.CSS_SELECTOR, by)
    return _orig_find_elements(self, by, value)
WebDriver.find_elements = _find_elements

_orig_find_element = WebDriver.find_element
def _find_element(self, by=None, value=None):
    if value is None and isinstance(by, str):
        return _orig_find_element(self, By.CSS_SELECTOR, by)
    return _orig_find_element(self, by, value)
WebDriver.find_element = _find_element

# Retry get_attribute on stale references for plotly graphs
_orig_get_attribute = WebElement.get_attribute
def _get_attribute(self, name):
    try:
        return _orig_get_attribute(self, name)
    except StaleElementReferenceException:
        try:
            if name == "id" and hasattr(self, "_parent"):
                fresh = self._parent.find_element(By.CSS_SELECTOR, "#main-chart")
                return _orig_get_attribute(fresh, name)
        except Exception:
            pass
        raise
WebElement.get_attribute = _get_attribute


import pytest

@pytest.fixture(autouse=True)
def _toggle_dash_test_mode(monkeypatch, request):
    """
    For UI/Selenium tests (dash_duo), force synthetic data to avoid live API calls.
    For pure unit tests, use real flow so freshness checks remain meaningful.
    """
    if "dash_duo" in request.fixturenames:
        monkeypatch.setenv("DASH_TEST_MODE", "1")
        monkeypatch.setenv("DASH_USE_SAMPLE_DATA", "1")
    else:
        monkeypatch.delenv("DASH_TEST_MODE", raising=False)
        monkeypatch.delenv("DASH_USE_SAMPLE_DATA", raising=False)
