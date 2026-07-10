from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from extractor.cli import app

runner = CliRunner()


def test_inspect_with_missing_config_file_exits_cleanly_not_a_traceback(tmp_path: Path):
    result = runner.invoke(
        app, ["inspect", "--folder", str(tmp_path), "--config", str(tmp_path / "missing.yaml")]
    )
    assert result.exit_code == 1
    assert "Failed to load config" in result.output
    assert "Traceback" not in result.output


def test_inspect_with_malformed_yaml_exits_cleanly(tmp_path: Path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("tiers:\n  - name: x\n  bad indent here\n")

    result = runner.invoke(app, ["inspect", "--folder", str(tmp_path), "--config", str(bad_yaml)])

    assert result.exit_code == 1
    assert "Failed to load config" in result.output
    assert "Traceback" not in result.output


def test_inspect_with_invalid_tier_config_exits_cleanly(tmp_path: Path):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(
        "tiers:\n"
        "  - name: bad\n"
        "    model_id: m\n"
        "    context_window_tokens: 100\n"
        "    reserved_output_tokens: 200\n"
        "judge:\n"
        "  name: judge\n"
        "  model_id: m\n"
        "  context_window_tokens: 100\n"
        "  reserved_output_tokens: 50\n"
    )

    result = runner.invoke(app, ["inspect", "--folder", str(tmp_path), "--config", str(bad_config)])

    assert result.exit_code == 1
    assert "Failed to load config" in result.output
    assert "Traceback" not in result.output


def test_inspect_with_valid_config_and_empty_folder_succeeds(tmp_path: Path):
    result = runner.invoke(app, ["inspect", "--folder", str(tmp_path)])
    assert result.exit_code == 0
    assert "0 document(s) found" in result.output


def test_version_command_prints_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip()
