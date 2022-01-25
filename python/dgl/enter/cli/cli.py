import typer
# import contextlib
# import io
# output = io.StringIO()
# with contextlib.redirect_stdout(output):
from ..pipeline import *
from ..model import *
from .config_cli import config_app
from .train_cli import train
from .export_cli import export

app = typer.Typer(no_args_is_help=True)
app.add_typer(config_app, name="config", no_args_is_help=True)
app.command(help="Train the model", no_args_is_help=True)(train)
app.command(help="Export the python file from config", no_args_is_help=True)(export)

def main():
    app()

if __name__ == "__main__":
    app()