#!/usr/bin/env python3
"""
Environment Check Script
Comprehensive validation of all required environment variables and dependencies.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dotenv import load_dotenv


class Colors:
    """Terminal colors for better output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_success(msg: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")


def print_error(msg: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {msg}{Colors.END}")


def print_warning(msg: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")


def print_info(msg: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")


def print_section(title: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}\n")


def check_env_file() -> Tuple[bool, List[str]]:
    """Check if .env file exists and has required variables."""
    env_file = Path(__file__).parent.parent / ".env"
    env_example = Path(__file__).parent.parent / ".env.example"

    issues = []

    # Load .env file if it exists
    if env_file.exists():
        load_dotenv(env_file)
        print_success(f"Loaded environment from {env_file}")
    else:
        msg = f".env file not found! Please copy {env_example} to {env_file} and configure."
        print_error(msg)
        issues.append(msg)
        return False, issues

    return True, issues


def validate_required_vars() -> Tuple[bool, List[str]]:
    """Validate required DeepSeek environment variables."""
    print_section("REQUIRED: DeepSeek AI Configuration")

    required_vars = {
        "DEEPSEEK_API_KEY": {
            "description": "DeepSeek API key for AI integration",
            "validate": lambda v: v.startswith("sk-"),
            "error": "API key should start with 'sk-'"
        },
        "DEEPSEEK_API_URL": {
            "description": "DeepSeek API endpoint URL",
            "validate": lambda v: v.startswith(("http://", "https://")),
            "error": "URL must be a valid HTTP/HTTPS URL"
        },
        "DEEPSEEK_MODEL": {
            "description": "DeepSeek model name (e.g., deepseek-chat)",
            "validate": lambda v: len(v) > 0,
            "error": "Model name cannot be empty"
        },
        "DEEPSEEK_MAX_TOKENS": {
            "description": "Maximum tokens for API responses",
            "validate": lambda v: 1 <= int(v) <= 32000,
            "error": "Must be between 1 and 32000"
        },
        "DEEPSEEK_TEMPERATURE": {
            "description": "AI temperature (creativity level)",
            "validate": lambda v: 0.0 <= float(v) <= 2.0,
            "error": "Must be between 0.0 and 2.0"
        },
    }

    all_valid = True
    issues = []

    for var, config in required_vars.items():
        value = os.getenv(var)

        # Check if missing
        if not value:
            msg = f"{var} is missing: {config['description']}"
            print_error(msg)
            issues.append(msg)
            all_valid = False
            continue

        # Check if placeholder
        if value.startswith("your_") or value.startswith("replace_with_"):
            msg = f"{var} has placeholder value: {config['description']}"
            print_error(msg)
            issues.append(msg)
            all_valid = False
            continue

        # Validate value
        try:
            if not config["validate"](value):
                msg = f"{var} validation failed: {config['error']}"
                print_error(msg)
                issues.append(msg)
                all_valid = False
            else:
                print_success(f"{var}: {value[:20]}{'...' if len(value) > 20 else ''}")
        except (ValueError, TypeError) as e:
            msg = f"{var} type conversion error: {str(e)}"
            print_error(msg)
            issues.append(msg)
            all_valid = False

    if all_valid:
        print_success("All required DeepSeek variables are properly configured")

    return all_valid, issues


def validate_optional_apis() -> Tuple[bool, List[str]]:
    """Validate optional API configurations."""
    print_section("OPTIONAL: Exchange & External APIs")

    all_valid = True
    issues = []

    # Check Binance API
    binance_vars = {
        "BINANCE_API_KEY": "Binance API key for trading",
        "BINANCE_API_SECRET": "Binance API secret",
        "BINANCE_TESTNET": "Use testnet (recommended: true)"
    }

    binance_missing = []
    for var, desc in binance_vars.items():
        value = os.getenv(var)
        if not value or value.startswith("your_"):
            binance_missing.append(var)
        else:
            # Mask the value for security
            display_value = value[:8] + "..." if len(value) > 8 else value
            print_info(f"{var}: {display_value}")

    if binance_missing:
        msg = f"Binance API keys not configured ({len(binance_missing)} missing)"
        print_warning(f"{msg} - Needed for live trading")
    else:
        testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
        if testnet:
            print_success("Binance testnet configured - SAFE for testing")
        else:
            print_warning("Binance live trading mode - REAL MONEY at risk!")

    # Check Minimax API
    minimax_vars = ["MINIMAX_API_KEY", "MINIMAX_GROUP_ID"]
    minimax_missing = []
    for var in minimax_vars:
        value = os.getenv(var)
        if not value or value.startswith("your_"):
            minimax_missing.append(var)
        else:
            display_value = value[:8] + "..." if len(value) > 8 else value
            print_info(f"{var}: {display_value}")

    if minimax_missing:
        msg = f"Minimax API keys not configured ({len(minimax_missing)} missing)"
        print_warning(f"{msg} - Needed for Claude Code integration")
    else:
        print_success("Minimax API configured")

    # Note: Missing optional APIs do not fail the validation
    # They are just warnings

    return all_valid, issues


def validate_trading_config() -> Tuple[bool, List[str]]:
    """Validate trading configuration."""
    print_section("CONFIGURATION: Trading Parameters")

    all_valid = True
    issues = []

    # Trading symbols
    symbols = os.getenv("TRADING_SYMBOLS")
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
        print_info(f"Trading symbols: {len(symbol_list)} configured")
        print_info(f"  {', '.join(symbol_list[:5])}{'...' if len(symbol_list) > 5 else ''}")
    else:
        print_warning("TRADING_SYMBOLS not set - using defaults")

    # Risk parameters
    risk_params = {
        "MAX_POSITION_SIZE": {
            "default": 0.05,
            "validate": lambda v: 0.0 < float(v) < 1.0,
            "desc": "Max position size (default: 5%)"
        },
        "MAX_TOTAL_EXPOSURE": {
            "default": 0.20,
            "validate": lambda v: 0.0 < float(v) < 1.0,
            "desc": "Max total exposure (default: 20%)"
        },
        "MAX_DRAWDOWN_LIMIT": {
            "default": 0.10,
            "validate": lambda v: 0.0 < float(v) < 1.0,
            "desc": "Max drawdown limit (default: 10%)"
        }
    }

    for param, config in risk_params.items():
        value = os.getenv(param)
        if value:
            try:
                if config["validate"](value):
                    pct = float(value) * 100
                    print_info(f"{param}: {pct:.1f}%")
                else:
                    print_warning(f"{param} should be between 0 and 1")
                    all_valid = False
            except ValueError:
                print_error(f"{param} must be a number")
                all_valid = False
        else:
            pct = config["default"] * 100
            print_info(f"{param}: {pct:.1f}% (default)")

    return all_valid, issues


def validate_system_config() -> Tuple[bool, List[str]]:
    """Validate system performance configuration."""
    print_section("CONFIGURATION: System & Performance")

    all_valid = True
    issues = []

    # System parameters
    sys_params = {
        "MEMORY_LIMIT_MB": {
            "default": 4000,
            "validate": lambda v: 1000 <= int(v) <= 16000,
            "desc": "Memory limit in MB (default: 4000)"
        },
        "DATA_CACHE_TTL": {
            "default": 3600,
            "validate": lambda v: 60 <= int(v) <= 86400,
            "desc": "Cache TTL in seconds (default: 3600 = 1h)"
        },
        "LOG_LEVEL": {
            "default": "INFO",
            "validate": lambda v: v.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "desc": "Logging level (default: INFO)"
        }
    }

    for param, config in sys_params.items():
        value = os.getenv(param)
        if value:
            try:
                if config["validate"](value):
                    print_info(f"{param}: {value}")
                else:
                    print_warning(f"{param} value may be invalid")
                    all_valid = False
            except ValueError:
                print_error(f"{param} must be a number")
                all_valid = False
        else:
            print_info(f"{param}: {config['default']} (default)")

    # DeepSeek memory config
    memory_vars = {
        "DEEPSEEK_CONTEXT_DAYS": "Context history (days)",
        "DEEPSEEK_MAX_MEMORY_TRADES": "Max trades in memory",
        "DEEPSEEK_LEARNING_ENABLED": "AI learning mode"
    }

    print_info("\nDeepSeek Memory Configuration:")
    for var, desc in memory_vars.items():
        value = os.getenv(var, "not set")
        print_info(f"  {var}: {value} ({desc})")

    return all_valid, issues


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if all required packages can be imported."""
    print_section("DEPENDENCIES: Required Packages")

    required_packages = {
        "pandas": "pandas",
        "numpy": "numpy",
        "plotly": "plotly",
        "dash": "dash",
        "loguru": "loguru",
        "requests": "requests",
        "dotenv": "python-dotenv",
        "openai": "openai",
        "sqlalchemy": "sqlalchemy",
    }

    optional_packages = {
        "ccxt": "ccxt (optional, for exchanges)",
        "torch": "pytorch (optional, for ML)",
        "websockets": "websockets (optional, for real-time)",
    }

    all_valid = True
    issues = []

    print_info("Required packages:")
    for package, name in required_packages.items():
        try:
            __import__(package)
            print_success(f"  {name} ✓")
        except ImportError:
            msg = f"  {name} NOT installed"
            print_error(msg)
            issues.append(msg)
            all_valid = False

    print_info("\nOptional packages:")
    for package, name in optional_packages.items():
        try:
            __import__(package)
            print_success(f"  {name} ✓")
        except ImportError:
            print_warning(f"  {name} not installed (optional)")

    if not all_valid:
        print_error("\nMissing packages. Install with:")
        print_info(f"  pip install {' '.join([v for k, v in required_packages.items() if k not in ['pandas', 'numpy']])}")

    return all_valid, issues


def check_config_file() -> Tuple[bool, List[str]]:
    """Check if configuration files exist."""
    print_section("CONFIGURATION: File System")

    issues = []

    # Check for config directory
    config_dir = Path(__file__).parent.parent / "config"
    if config_dir.exists():
        print_success(f"Config directory exists: {config_dir}")
    else:
        print_warning(f"Config directory not found: {config_dir}")

    # Check for data directory
    data_dir = Path(__file__).parent.parent / "data"
    if data_dir.exists():
        print_success(f"Data directory exists: {data_dir}")
    else:
        print_warning(f"Data directory not found: {data_dir}")
        print_info("  Will be created automatically when needed")

    # Check for logs directory
    logs_dir = Path(__file__).parent.parent / "logs"
    if logs_dir.exists():
        print_success(f"Logs directory exists: {logs_dir}")
    else:
        print_warning(f"Logs directory not found: {logs_dir}")
        print_info("  Will be created automatically when needed")

    return True, issues


def generate_report(all_checks_passed: bool, all_issues: List[str]) -> str:
    """Generate a summary report."""
    report = []

    report.append("\n" + "=" * 70)
    report.append("ENVIRONMENT VALIDATION REPORT")
    report.append("=" * 70)

    if all_checks_passed:
        report.append(f"{Colors.GREEN}{Colors.BOLD}✅ ALL CHECKS PASSED{Colors.END}")
        report.append("\nYour environment is properly configured and ready to use!")
    else:
        report.append(f"{Colors.RED}{Colors.BOLD}❌ SOME CHECKS FAILED{Colors.END}")
        report.append(f"\nTotal issues found: {len(all_issues)}")
        report.append("\nIssues to fix:")
        for i, issue in enumerate(all_issues, 1):
            report.append(f"  {i}. {issue}")

    report.append("\n" + "=" * 70)
    report.append("NEXT STEPS")
    report.append("=" * 70)

    if all_checks_passed:
        report.append("\n1. Run the test suite:")
        report.append("   python -m pytest tests/ -v")
        report.append("\n2. Start the dashboard:")
        report.append("   python ui/dashboard.py")
        report.append("\n3. Try the signal generator:")
        report.append("   python examples/signal_generator_demo.py")
    else:
        report.append("\n1. Fix the issues listed above")
        report.append("2. Run this check again:")
        report.append("   python ops/check_env.py")
        report.append("3. Consult the documentation:")
        report.append("   - INTEGRATION_GUIDE.md")
        report.append("   - QUICKSTART.md")

    report.append("\n" + "=" * 70)

    return "\n".join(report)


def main():
    """Run all environment checks."""
    print()
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}DeepSeek Trading System - Environment Validation{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}")

    all_checks_passed = True
    all_issues = []

    # Run all checks
    checks = [
        ("Environment File", check_env_file),
        ("Required Variables", validate_required_vars),
        ("Optional APIs", validate_optional_apis),
        ("Trading Config", validate_trading_config),
        ("System Config", validate_system_config),
        ("Dependencies", check_dependencies),
        ("Config Files", check_config_file),
    ]

    for name, check_func in checks:
        try:
            result, issues = check_func()
            if not result:
                all_checks_passed = False
                all_issues.extend(issues)
        except Exception as e:
            print_error(f"Check '{name}' failed with exception: {str(e)}")
            all_checks_passed = False
            all_issues.append(f"Check '{name}' failed: {str(e)}")

    # Generate and print report
    report = generate_report(all_checks_passed, all_issues)
    print(report)

    # Return exit code
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {str(e)}{Colors.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
