import typer
from .config_cli import config
from .train_cli import train
from .export_cli import export

app = typer.Typer()
app.command()(config)
app.command()(train)
# app.command()(export)

if __name__ == "__main__":
    app()