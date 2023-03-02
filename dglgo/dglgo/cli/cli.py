import typer
from ..pipeline import *
from ..model import *
from .apply_cli import apply
from .config_apply_cli import config_apply_app
from .config_cli import config_app
from .export_cli import export
from .recipe_cli import recipe_app
from .train_cli import train

no_args_is_help = False
app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(config_app, name="configure", no_args_is_help=no_args_is_help)
app.add_typer(recipe_app, name="recipe", no_args_is_help=True)
app.command(help="Launch training", no_args_is_help=no_args_is_help)(train)
app.command(
    help="Export a runnable python script", no_args_is_help=no_args_is_help
)(export)
app.add_typer(
    config_apply_app, name="configure-apply", no_args_is_help=no_args_is_help
)
app.command(help="Launch inference", no_args_is_help=no_args_is_help)(apply)


def main():
    app()


if __name__ == "__main__":
    app()
