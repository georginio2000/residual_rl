"""Minimal stand-in for the legacy ``mujoco_py`` package.

robomimic 0.2.0/0.3.0's ``EnvRobosuite`` wrapper unconditionally imports
``mujoco_py`` at module scope, but only ever references
``mujoco_py.builder.MujocoException`` as an exception type to catch during
rollouts. Since robosuite 1.4.1 (installed here) uses the modern ``mujoco``
bindings and never actually calls into real ``mujoco_py``, this stub exists
purely so that import succeeds; the exception type is never raised because
the real mujoco_py simulator is never used.
"""

import types

builder = types.ModuleType("mujoco_py.builder")


class MujocoException(Exception):
    pass


builder.MujocoException = MujocoException
