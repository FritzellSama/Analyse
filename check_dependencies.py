#!/usr/bin/env python3
"""
Dependency Checker - Quantum Trader Pro
Validates that all required dependencies are installed and compatible.
"""

import sys
import importlib
import subprocess
from typing import List, Tuple, Dict


def check_python_version() -> Tuple[bool, str]:
    """Check Python version compatibility"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major < 3:
        return False, f"Python 3.9+ required, found: {version_str}"

    if version.minor < 9:
        return False, f"Python 3.9+ required, found: {version_str}"

    if version.minor >= 12:
        return False, f"Python 3.11 or below required (3.12+ not tested), found: {version_str}"

    return True, f"Python {version_str} OK"


def check_core_dependencies() -> List[Tuple[str, bool, str]]:
    """Check core required dependencies"""
    core_deps = [
        ("ccxt", "ccxt"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("yaml", "pyyaml"),
        ("dotenv", "python-dotenv"),
        ("colorama", "colorama"),
        ("requests", "requests"),
        ("cryptography", "cryptography"),
    ]

    results = []
    for module_name, package_name in core_deps:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            results.append((package_name, True, f"v{version}"))
        except ImportError:
            results.append((package_name, False, "NOT INSTALLED"))

    return results


def check_ml_dependencies() -> List[Tuple[str, bool, str]]:
    """Check ML dependencies (optional)"""
    ml_deps = [
        ("sklearn", "scikit-learn"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("optuna", "optuna"),
    ]

    results = []
    for module_name, package_name in ml_deps:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            results.append((package_name, True, f"v{version}"))
        except ImportError:
            results.append((package_name, False, "NOT INSTALLED (optional)"))

    return results


def check_visualization_dependencies() -> List[Tuple[str, bool, str]]:
    """Check visualization dependencies (optional)"""
    viz_deps = [
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("plotly", "plotly"),
    ]

    results = []
    for module_name, package_name in viz_deps:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            results.append((package_name, True, f"v{version}"))
        except ImportError:
            results.append((package_name, False, "NOT INSTALLED (optional)"))

    return results


def check_database_dependencies() -> List[Tuple[str, bool, str]]:
    """Check database dependencies"""
    db_deps = [
        ("sqlalchemy", "sqlalchemy"),
        ("alembic", "alembic"),
    ]

    results = []
    for module_name, package_name in db_deps:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            results.append((package_name, True, f"v{version}"))
        except ImportError:
            results.append((package_name, False, "NOT INSTALLED"))

    return results


def check_ta_lib() -> Tuple[bool, str]:
    """Check TA-Lib installation (requires system library)"""
    try:
        import talib
        return True, f"v{talib.__version__}"
    except ImportError as e:
        if "libta_lib" in str(e) or "cannot open shared object" in str(e):
            return False, "System TA-Lib library not installed"
        return False, "NOT INSTALLED"


def check_tensorflow() -> Tuple[bool, str]:
    """Check TensorFlow (optional, heavy dependency)"""
    try:
        import tensorflow as tf
        return True, f"v{tf.__version__}"
    except ImportError:
        return False, "NOT INSTALLED (optional)"


def print_section(title: str, results: List[Tuple[str, bool, str]]):
    """Print a section of dependency check results"""
    print(f"\n{title}")
    print("=" * 50)

    for name, ok, info in results:
        status = "✅" if ok else "❌"
        print(f"  {status} {name:<25} {info}")


def main():
    """Main dependency check routine"""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║              QUANTUM TRADER PRO - Dependency Check                 ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

    all_ok = True
    critical_missing = []

    # Python version
    py_ok, py_msg = check_python_version()
    print(f"\n{'✅' if py_ok else '❌'} Python Version: {py_msg}")
    if not py_ok:
        all_ok = False
        critical_missing.append("Python 3.9-3.11")

    # Core dependencies
    core_results = check_core_dependencies()
    print_section("CORE DEPENDENCIES (Required)", core_results)
    for name, ok, _ in core_results:
        if not ok:
            all_ok = False
            critical_missing.append(name)

    # Database
    db_results = check_database_dependencies()
    print_section("DATABASE", db_results)
    for name, ok, _ in db_results:
        if not ok:
            all_ok = False
            critical_missing.append(name)

    # TA-Lib (special case)
    ta_ok, ta_msg = check_ta_lib()
    print(f"\n{'✅' if ta_ok else '⚠️ '} TA-Lib: {ta_msg}")
    if not ta_ok:
        print("    ⚠️  TA-Lib requires system library installation:")
        print("       Ubuntu/Debian: sudo apt-get install ta-lib")
        print("       macOS: brew install ta-lib")
        print("       Then: pip install ta-lib")

    # ML dependencies (optional)
    ml_results = check_ml_dependencies()
    print_section("MACHINE LEARNING (Optional)", ml_results)

    # Visualization (optional)
    viz_results = check_visualization_dependencies()
    print_section("VISUALIZATION (Optional)", viz_results)

    # TensorFlow (optional, heavy)
    tf_ok, tf_msg = check_tensorflow()
    print(f"\n{'✅' if tf_ok else 'ℹ️ '} TensorFlow: {tf_msg}")

    # Summary
    print("\n" + "=" * 70)
    if all_ok:
        print("✅ All critical dependencies are installed!")
        print("\nYou can now run: python main.py")
    else:
        print("❌ Missing critical dependencies:")
        for dep in critical_missing:
            print(f"   - {dep}")
        print("\nInstall missing dependencies with:")
        print("   pip install -r requirements.txt")
    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
