from pathlib import Path
from typing import Optional
import typer
import os
import shutil
import yaml

def list_recipes():
    file_current_dir = Path(__file__).resolve().parent
    recipe_dir = file_current_dir.parent.parent / "recipes"
    file_list = list(recipe_dir.glob("*.yaml"))
    header = "| {:<30} |  {:<18} | {:<20} |".format("Filename", "Pipeline", "Dataset")
    typer.echo("="*len(header))
    typer.echo(header)
    typer.echo("="*len(header))
    for file in file_list:
        cfg = yaml.safe_load(Path(file).open("r"))
        typer.echo("| {:<30} |  {:<18} | {:<20} |".format(file.name, cfg["pipeline_name"], cfg["data"]["name"]))
    typer.echo("="*len(header))

def copy_recipes(dir: str = typer.Option("dglgo_example_recipes", help="directory name for recipes")):
    file_current_dir = Path(__file__).resolve().parent
    recipe_dir = file_current_dir.parent.parent / "recipes"
    current_dir = Path(os.getcwd())
    new_dir = current_dir / dir
    new_dir.mkdir(parents=True, exist_ok=True)
    for file in recipe_dir.glob("*.yaml"):
        shutil.copy(file, new_dir)
    print("Example recipes are copied to {}".format(new_dir.absolute()))

def get_recipe(recipe_name: Optional[str] = typer.Argument(None, help="The recipe filename to get, e.q. nodepred_citeseer_gcn.yaml")):
    if recipe_name is None:
        typer.echo("Usage: dgl recipe get [RECIPE_NAME] \n")
        typer.echo(" Copy the recipe to current directory \n")
        typer.echo(" Arguments:")
        typer.echo("  [RECIPE_NAME]  The recipe filename to get, e.q. nodepred_citeseer_gcn.yaml\n")
        typer.echo("Here are all avaliable recipe filename")
        list_recipes()
    else:
        file_current_dir = Path(__file__).resolve().parent
        recipe_dir = file_current_dir.parent.parent / "recipes"
        current_dir = Path(os.getcwd())
        recipe_path = recipe_dir / recipe_name
        shutil.copy(recipe_path, current_dir)
        print("Recipe {} is copied to {}".format(recipe_path.absolute(), current_dir.absolute()))


recipe_app = typer.Typer(help="Get example recipes")
recipe_app.command(name="list", help="List all available example recipes")(list_recipes)
recipe_app.command(name="copy", help="Copy all available example recipes to current directory")(copy_recipes)
recipe_app.command(name="get", help="Copy the recipe to current directory")(get_recipe)

if __name__ == "__main__":
    recipe_app()
