import typer
from ..pipeline import *
from ..model import *
from .config_cli import config_app
from .train_cli import train
from .export_cli import export

no_args_is_help = False
app = typer.Typer(no_args_is_help=no_args_is_help, add_completion=False)
app.add_typer(config_app, name="config", no_args_is_help=no_args_is_help)
app.command(help="Train the model", no_args_is_help=no_args_is_help)(train)
app.command(help="Export the python file from config", no_args_is_help=no_args_is_help)(export)

def main():
    app()

if __name__ == "__main__":
    app()