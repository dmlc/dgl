from pathlib import Path
import typer
import os
import shutil
import yaml

def list_receipes():
    file_current_dir = Path(__file__).resolve().parent
    recipe_dir = file_current_dir.parent.parent / "recipes"
    file_list = list(recipe_dir.glob("*.yaml"))
    header = "| {:<30} |  {:<18} | {:<20} |".format("Filename", "Pipeline", "Dataset")
    print("="*len(header))
    print(header)
    print("="*len(header))
    for file in file_list:
        cfg = yaml.safe_load(Path(file).open("r"))
        print("| {:<30} |  {:<18} | {:<20} |".format(file.name, cfg["pipeline_name"], cfg["data"]["name"]))
    print("="*len(header))

def copy_receipes(dir: str = typer.Option("dglenter_templates", help="directory name for recipes")):
    file_current_dir = Path(__file__).resolve().parent
    recipe_dir = file_current_dir.parent.parent / "recipes"
    current_dir = Path(os.getcwd())
    new_dir = current_dir / dir
    new_dir.mkdir(parents=True, exist_ok=True)
    for file in recipe_dir.glob("*.yaml"):
        shutil.copy(file, new_dir)
    print("Template recipes are copied to {}".format(new_dir.absolute()))


recipe_app = typer.Typer(help="Get the template recipes")
recipe_app.command(name="list", help="List all available template recipes")(list_receipes)
recipe_app.command(name="copy", help="Copy all available template recipes to current directory")(copy_receipes)

if __name__ == "__main__":
    recipe_app()