"""
Pytest Configuration File
Phase 2 Module 5: Correlation Module

This file is loaded by pytest before running tests.
It adds src/ to Python's module search path.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# ============================================================================
# CRITICAL FIX: Add src/ to Python path
# ============================================================================
# This single line fixes the ModuleNotFoundError
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ============================================================================
# PYTEST FIXTURES: Test Data Generation
# ============================================================================

@pytest.fixture
def sample_kpi_data():
    """Generate random KPI data for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'KPI_A': np.random.randn(1000),
        'KPI_B': np.random.randn(1000),
        'KPI_C': np.random.randn(1000),
        'KPI_D': np.random.randn(1000),
    })


@pytest.fixture
def perfectly_correlated_data():
    """Generate perfectly correlated KPI data"""
    base = np.random.randn(1000)
    return pd.DataFrame({
        'KPI_PERFECT_POS': base,
        'KPI_PERFECT_NEG': -base,
        'KPI_PERFECT_DUP': base + 1e-10 * np.random.randn(1000),
    })


@pytest.fixture
def uncorrelated_data():
    """Generate uncorrelated KPI data"""
    np.random.seed(42)
    return pd.DataFrame({
        'KPI_INDEP_1': np.random.randn(1000),
        'KPI_INDEP_2': np.random.randn(1000),
        'KPI_INDEP_3': np.random.randn(1000),
    })


@pytest.fixture
def data_with_correlations():
    """Generate data with known strong correlations"""
    np.random.seed(42)
    base = np.random.randn(1000)
    noise = lambda: 0.05 * np.random.randn(1000)
    
    return pd.DataFrame({
        'KPI_X': base,
        'KPI_Y': base + noise(),
        'KPI_Z': -base + noise(),
        'KPI_W': np.random.randn(1000),
    })
