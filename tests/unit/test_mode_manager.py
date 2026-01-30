"""Tests for mode manager state machine."""



def test_mode_manager_initialization():
    """Test ModeManager starts in NORMAL mode."""
    from ddt.supervision.mode_manager import ModeManager, OperationMode

    manager = ModeManager()
    assert manager.mode == OperationMode.NORMAL


def test_transition_normal_to_conservative():
    """Test transition from NORMAL to CONSERVATIVE on high uncertainty."""
    from ddt.supervision.mode_manager import ModeConfig, ModeManager, OperationMode

    config = ModeConfig(uncertainty_normal_to_conservative=1.0)
    manager = ModeManager(config=config)

    # High uncertainty should trigger transition
    mode = manager.update(uncertainty_norm=1.5, is_pe_satisfied=True)
    assert mode == OperationMode.CONSERVATIVE


def test_transition_conservative_to_safe():
    """Test transition from CONSERVATIVE to SAFE_STOP on very high uncertainty."""
    from ddt.supervision.mode_manager import ModeConfig, ModeManager, OperationMode

    config = ModeConfig(
        uncertainty_normal_to_conservative=1.0,
        uncertainty_conservative_to_safe=5.0,
    )
    manager = ModeManager(config=config)

    # First to CONSERVATIVE
    manager.update(uncertainty_norm=1.5, is_pe_satisfied=True)
    assert manager.mode == OperationMode.CONSERVATIVE

    # Then to SAFE_STOP
    mode = manager.update(uncertainty_norm=6.0, is_pe_satisfied=True)
    assert mode == OperationMode.SAFE_STOP


def test_recovery_from_conservative():
    """Test recovery from CONSERVATIVE back to NORMAL."""
    from ddt.supervision.mode_manager import ModeConfig, ModeManager, OperationMode

    config = ModeConfig(
        uncertainty_normal_to_conservative=1.0,
        uncertainty_conservative_to_normal=0.5,
        pe_satisfaction_count_recovery=5,
    )
    manager = ModeManager(config=config)

    # Go to CONSERVATIVE
    manager.update(uncertainty_norm=1.5, is_pe_satisfied=True)
    assert manager.mode == OperationMode.CONSERVATIVE

    # Low uncertainty + PE satisfied for enough steps
    for _ in range(10):
        mode = manager.update(uncertainty_norm=0.3, is_pe_satisfied=True)

    assert mode == OperationMode.NORMAL


def test_pe_violation_triggers_conservative():
    """Test that PE violations can trigger CONSERVATIVE mode."""
    from ddt.supervision.mode_manager import ModeConfig, ModeManager, OperationMode

    config = ModeConfig(pe_violation_count_conservative=5)
    manager = ModeManager(config=config)

    # PE violations with low uncertainty
    for _ in range(10):
        mode = manager.update(uncertainty_norm=0.1, is_pe_satisfied=False)

    assert mode == OperationMode.CONSERVATIVE


def test_safety_margin_factor():
    """Test safety margin factors for different modes."""
    from ddt.supervision.mode_manager import ModeConfig, ModeManager

    config = ModeConfig(
        margin_factor_normal=1.0,
        margin_factor_conservative=2.0,
        margin_factor_safe=5.0,
        uncertainty_normal_to_conservative=1.0,
        uncertainty_conservative_to_safe=5.0,
    )
    manager = ModeManager(config=config)

    # NORMAL
    assert manager.safety_margin_factor == 1.0

    # CONSERVATIVE
    manager.update(uncertainty_norm=2.0, is_pe_satisfied=True)
    assert manager.safety_margin_factor == 2.0

    # SAFE_STOP
    manager.update(uncertainty_norm=6.0, is_pe_satisfied=True)
    assert manager.safety_margin_factor == 5.0


def test_explicit_safe_stop_trigger():
    """Test explicit SAFE_STOP trigger."""
    from ddt.supervision.mode_manager import ModeManager, OperationMode

    manager = ModeManager()
    assert manager.mode == OperationMode.NORMAL

    manager.trigger_safe_stop()
    assert manager.mode == OperationMode.SAFE_STOP


def test_reset():
    """Test mode manager reset."""
    from ddt.supervision.mode_manager import ModeManager, OperationMode

    manager = ModeManager()
    manager.trigger_safe_stop()
    assert manager.mode == OperationMode.SAFE_STOP

    manager.reset()
    assert manager.mode == OperationMode.NORMAL
