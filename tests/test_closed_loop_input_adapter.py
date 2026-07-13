"""Tests for the CARLA closed-loop input adapter (no CARLA install required)."""
import subprocess
import sys


def test_package_imports_without_carla():
    """The closed-loop package must import in an env with no carla/leaderboard."""
    code = (
        "import sys; sys.modules['carla'] = None; sys.modules['leaderboard'] = None; "
        "import navsim.evaluation.carla_closed_loop.pid_controller; "
        "import navsim.evaluation.carla_closed_loop.route_planner; "
        "print('OK')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
