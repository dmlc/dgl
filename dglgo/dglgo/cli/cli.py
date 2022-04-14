import typer
from ..pipeline import *
from ..model import *
from .config_cli import config_app
from .train_cli import train
from .export_cli import export
from .recipe_cli import recipe_app

no_args_is_help = False
app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(config_app, name="configure", no_args_is_help=no_args_is_help)
app.add_typer(recipe_app, name="recipe", no_args_is_help=True)
app.command(help="Launch training", no_args_is_help=no_args_is_help)(train)
app.command(help="Export a runnable python script", no_args_is_help=no_args_is_help)(export)

def main():
    app()

if __name__ == "__main__":
    app()
