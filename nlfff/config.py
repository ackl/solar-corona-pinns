from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

T = TypeVar("T")


class ConfigError(ValueError):
    """oops"""


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigError(f"{path}: expected YAML file")
    try:
        import yaml

        with path.open("rb") as stream:
            value = yaml.safe_load(stream)
    except Exception as error:
        raise ConfigError(f"{path}: could not load configuration: {error}") from error
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: configuration must be a mapping")
    return value


def _convert(value: Any, annotation: Any, source: str) -> Any:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, UnionType):
        if value is None and type(None) in args:
            return None
        candidates = [item for item in args if item is not type(None)]
        if len(candidates) == 1:
            return _convert(value, candidates[0], source)
    if origin is list:
        if isinstance(value, str):
            if source.startswith("configuration key"):
                raise ConfigError(f"{source}: expected a list")
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise ConfigError(f"{source}: expected a JSON list") from error
        if not isinstance(value, list):
            raise ConfigError(f"{source}: expected a list")
        return [_convert(item, args[0] if args else Any, source) for item in value]
    if annotation is Any:
        return value
    if annotation is Path:
        if not isinstance(value, (str, os.PathLike)):
            raise ConfigError(f"{source}: expected a path")
        return Path(value)
    if annotation is bool:
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        raise ConfigError(f"{source}: expected true or false")
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{source}: expected str")
        return value
    if annotation in {int, float}:
        if isinstance(value, bool):
            raise ConfigError(f"{source}: expected {annotation.__name__}")
        try:
            return annotation(value)
        except (TypeError, ValueError) as error:
            raise ConfigError(f"{source}: expected {annotation.__name__}") from error
    return value


def load_config(
    config_type: type[T],
    *,
    defaults: T | None = None,
    config_path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    env_prefix: str | None = None,
    env_fields: set[str] | frozenset[str] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> T:
    """Resolve defaults < file < allowlisted namespaced environment < CLI."""
    if not is_dataclass(config_type):
        raise TypeError("config_type must be a dataclass type")
    definitions = {item.name: item for item in fields(config_type)}
    if defaults is not None and not isinstance(defaults, config_type):
        raise TypeError("defaults must be an instance of config_type")
    annotations = get_type_hints(config_type)
    values = asdict(defaults if defaults is not None else config_type())
    file_values = _load_mapping(Path(config_path)) if config_path is not None else {}
    unknown = set(file_values) - set(definitions)
    if unknown:
        raise ConfigError("unknown configuration keys: " + ", ".join(sorted(unknown)))
    for name, value in file_values.items():
        values[name] = _convert(value, annotations[name], f"configuration key {name}")
    environment = os.environ if env is None else env
    allowed_environment = set(definitions) if env_fields is None else set(env_fields)
    unknown = allowed_environment - set(definitions)
    if unknown:
        raise ConfigError(
            "unknown environment override keys: " + ", ".join(sorted(unknown))
        )
    if env_prefix:
        for name in allowed_environment:
            variable = env_prefix + name.upper()
            if variable in environment and environment[variable] != "":
                values[name] = _convert(
                    environment[variable], annotations[name], variable
                )
    overrides = dict(cli_overrides or {})
    unknown = set(overrides) - set(definitions)
    if unknown:
        raise ConfigError("unknown CLI override keys: " + ", ".join(sorted(unknown)))
    for name, value in overrides.items():
        if value is not None:
            values[name] = _convert(value, annotations[name], f"CLI argument {name}")
    try:
        return config_type(**values)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid resolved configuration: {error}") from error


def public_config(config: Any) -> dict[str, Any]:
    """JSON shaped config"""
    if not is_dataclass(config):
        raise TypeError("config must be a dataclass instance")
    result = {}
    for definition in fields(config):
        value = getattr(config, definition.name)
        result[definition.name] = str(value) if isinstance(value, Path) else value
    return result
