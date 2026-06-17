from __future__ import annotations

from pathlib import Path

from tools import compile as compile_tool


def test_compiler_include_args_uses_env_include_dirs(monkeypatch, tmp_path):
    include_dir = tmp_path / "include"
    include_dir.mkdir()
    missing_dir = tmp_path / "missing"
    monkeypatch.setenv(
        "EXTRA_C_INCLUDE_DIRS",
        f"{include_dir}{compile_tool.os.pathsep}{missing_dir}",
    )

    args = compile_tool._compiler_include_args("gcc")

    assert "-isystem" in args
    assert str(include_dir) in args
    assert str(missing_dir) not in args


def test_compiler_include_args_uses_riscv_sysroot_and_include(monkeypatch, tmp_path):
    sysroot = tmp_path / "sysroot"
    include_dir = tmp_path / "riscv-include"
    sysroot.mkdir()
    include_dir.mkdir()
    monkeypatch.setenv("RISCV_SYSROOT", str(sysroot))
    monkeypatch.setenv("RISCV_INCLUDE_DIR", str(include_dir))

    args = compile_tool._compiler_include_args("riscv64-unknown-elf-gcc")

    assert args[:2] == ["--sysroot", str(sysroot)]
    assert str(include_dir) in args
