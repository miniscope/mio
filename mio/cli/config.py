"""
CLI commands for configuration
"""

import ast
import re
from pathlib import Path

import click
import yaml
from pydantic import TypeAdapter, ValidationError
from rich.console import Console
from rich.table import Table

from mio.const import CONFIG_DIR
from mio.models import config as _config
from mio.models.config import set_user_dir
from mio.models.mixins import ConfigYAMLMixin
from mio.types import ConfigID


@click.group(invoke_without_command=True)
@click.pass_context
def config(ctx: click.Context) -> None:
    """
    Command group for config

    When run without arguments, displays current config from all sources
    """
    if ctx.invoked_subcommand is None:
        config_str = _config.Config().to_yaml()
        click.echo(f"mio configuration:\n-----\n{config_str}")


@config.group("global", invoke_without_command=True)
@click.pass_context
def global_(ctx: click.Context) -> None:
    """
    Command group for global configuration directory

    When run without arguments, displays contents of current global config
    """
    if ctx.invoked_subcommand is None:

        with open(_config._global_config_path) as f:
            config_str = f.read()

        click.echo(f"Global configuration: {str(_config._global_config_path)}\n-----\n{config_str}")


@global_.command("path")
def global_path() -> None:
    """Location of the global mio config"""
    click.echo(str(_config._global_config_path))


@config.group(invoke_without_command=True)
@click.pass_context
def user(ctx: click.Context) -> None:
    """
    Command group for the user config directory

    When invoked without arguments, displays the contents of the current user directory
    """
    if ctx.invoked_subcommand is None:
        config = _config.Config()
        config_file = list(config.user_dir.glob("mio_config.*"))
        if len(config_file) == 0:
            click.echo(
                f"User directory specified as {str(config.user_dir)} "
                "but no mio_config.yaml file found"
            )
            return
        else:
            config_file = config_file[0]

        with open(config_file) as f:
            config_str = f.read()

        click.echo(f"User configuration: {str(config_file)}\n-----\n{config_str}")


@user.command("create")
@click.argument("user_dir", type=click.Path(), required=False)
@click.option(
    "--force/--no-force",
    default=False,
    help="Overwrite existing config file if it exists",
)
@click.option(
    "--clean/--dirty",
    default=False,
    help="Create a fresh mio_config.yaml file containing only the user_dir. "
    "Otherwise, by default (--dirty), any other settings from .env, pyproject.toml, etc."
    "are included in the created user config file.",
)
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    help="Show the config that would be written and where it would go without doing anything",
)
def user_create(
    user_dir: Path = None, force: bool = False, clean: bool = False, dry_run: bool = False
) -> None:
    """
    Create a user directory,
    setting it as the default in the global config

    Args:
        user_dir (Path): Path to the directory to create
        force (bool): Overwrite existing config file if it exists
    """
    if user_dir is None:
        user_dir = _config._default_userdir

    try:
        user_dir = Path(user_dir).expanduser().resolve()
    except RuntimeError:
        user_dir = Path(user_dir).resolve()

    if user_dir.is_file and user_dir.suffix in (".yaml", ".yml"):
        config_file = user_dir
        user_dir = user_dir.parent
    else:
        config_file = user_dir / "mio_config.yaml"

    if config_file.exists() and not force and not dry_run:
        click.echo(f"Config file already exists at {str(config_file)}, use --force to overwrite")
        return

    if clean:
        config = {"user_dir": str(user_dir)}

        if not dry_run:
            with open(config_file, "w") as f:
                yaml.safe_dump(config, f)

        config_str = yaml.safe_dump(config)
    else:
        config = _config.Config(user_dir=user_dir)
        config_str = config.to_yaml() if dry_run else config.to_yaml(config_file)

    # update global config pointer
    if not dry_run:
        set_user_dir(user_dir)

    prefix = "DRY RUN - No files changed\n-----\nWould have created" if dry_run else "Created"

    click.echo(f"{prefix} user config at {str(config_file)}:\n-----\n{config_str}")


@user.command("path")
def user_path() -> None:
    """Location of the current user config"""
    path = list(_config.Config().user_dir.glob("mio_config.*"))[0]
    click.echo(str(path))


@config.command("list")
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Display more verbose output\n" "-v: Display full model identifiers and config paths",
)
def _list(verbose: int) -> None:
    """
    Display the available configs in the user config directory and provided by mio.

    By default, results are truncated for narrow consoles -
    models within `mio.models` are displayed with only a leading `.`,
    and paths of builtin configs are shown relative to the package config directory
    """
    config_headers = [cfg for cfg in ConfigYAMLMixin.iter_configs()]
    config_headers = sorted(config_headers, key=lambda cfg: (cfg["mio_model"], cfg["id"]))

    table = Table(title="mio configs")
    table.add_column("id", style="yellow", no_wrap=True)
    table.add_column("mio_model")
    table.add_column("path")

    for header in config_headers:
        table.add_row(
            header["id"],
            re.sub(r"^mio.models", "", header["mio_model"]) if verbose < 1 else header["mio_model"],
            (
                str(header["path"].relative_to(CONFIG_DIR))
                if verbose < 1 and CONFIG_DIR in header["path"].parents
                else str(header["path"])
            ),
        )

    console = Console()
    console.print(table)


@config.command("create")
@click.argument(
    "model",
    required=False,
)
@click.argument(
    "config_id",
    metavar="id",
    required=False,
)
@click.option("-f", "--force", help="Overwrite existing config file", is_flag=True)
@click.option(
    "-v", "--value", help="Pass key/value pairs to the model like key=value", multiple=True
)
@click.option(
    "--output",
    required=False,
    default=None,
    type=click.Path(),
    help="Provide an explicit output path for the yaml config",
)
@click.option("--list", "show_list", is_flag=True, help="List available models")
def create(
    model: str | None = None,
    config_id: ConfigID | None = None,
    force: bool = False,
    value: tuple = (),
    output: Path | None = None,
    show_list: bool = False,
) -> None:
    """
    Create a new default config for a model in the user config directory,
    given the name of a model and the name of a config id.

    ```
    mio config create MyModel some-config-id
    ```

    Use `--list` to see the available model names.

    Pass required key/value pairs with `--value key=value`, e.g.

    mio config create MyModel some-config --value device=abc

    By default, created configs are saved to the user config directory with an escaped file name
    based on the passed ``id`` ,
    but an explicit output path can be passed.
    Note that creating configs in locations
    that are not in the user config directory will prevent mio from finding them.
    (see ``mio config user`` )
    """
    if show_list:
        table = Table(title="mio config models")
        table.add_column("name", style="yellow", no_wrap=True)
        table.add_column("module path")
        for model_name, model in ConfigYAMLMixin.config_models().items():
            table.add_row(model_name, f"{model.__module__}.{model.__name__}")

        console = Console()
        console.print(table)
        return

    assert model is not None, "Model name must be provided, use --list to see available models"
    assert config_id is not None, "config id must be provided"
    try:
        config_id = TypeAdapter(ConfigID).validate_python(config_id)
    except ValidationError as err:
        raise ValueError("Config ID must follow config id pattern") from err

    output = Path(output) if output else _config.Config().config_dir / (config_id + ".yaml")

    if not force and output.exists():
        click.echo(f"{output} already exists. use --force to overwrite")
        return

    models = ConfigYAMLMixin.config_models()
    if model not in models:
        raise ValueError(f"model {model} not found. available models: {models.keys()}")

    try:
        kwargs = _parse_kwargs(value)
    except Exception as e:
        raise ValueError(
            "Error parsing value kwargs, must be specified like --value key=value"
        ) from e

    instance = models[model](id=config_id, **kwargs)
    yaml_str = instance.to_yaml(output)
    click.echo(f"Wrote {model} config to {output}\n{yaml_str}")


@config.command("path")
@click.argument("config_id", metavar="id")
def config_path(config_id: str) -> None:
    """
    Print the path for a config

    e.g. to open a config for editing

    ```
    open $(mio config path my-config)
    ```
    """
    for cfg in ConfigYAMLMixin.iter_configs():
        if cfg["id"] == config_id:
            click.echo(cfg["path"])
            return
    raise KeyError(f"No config {config_id} found")


def _parse_kwargs(value: tuple[str]) -> dict:
    kwargs = {}
    for v in value:
        key, val = v.split("=")
        kwargs[key] = ast.literal_eval(val)
    return kwargs
