import typer
from ..pipeline import *
from ..model import *
from .config_cli import config_app
from .train_cli import train

app = typer.Typer()
app.add_typer(config_app, name="config")
app.command()(train)

if __name__ == "__main__":
    app()