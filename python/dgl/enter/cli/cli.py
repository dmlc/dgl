import typer
# import contextlib
# import io
# output = io.StringIO()
# with contextlib.redirect_stdout(output):
from ..pipeline import *
from ..model import *
from .config_cli import config_app
from .train_cli import train

app = typer.Typer()
app.add_typer(config_app, name="config")
app.command(help="Train the model")(train)

def main():
    app()

if __name__ == "__main__":
    app()