import json
from pathlib import Path


def main():
    result_dir = Path(__file__).parent / ".." / Path("results/")
    for per_machine_dir in result_dir.iterdir():
        if per_machine_dir.is_dir():
            try:
                machine_json = json.loads(
                    (per_machine_dir / "machine.json").read_text()
                )
                ram = machine_json["ram"]
                for f in per_machine_dir.glob("*.json"):
                    if f.stem != "machine":
                        result = json.loads(f.read_text())
                        result_ram = result["params"]["ram"]
                        if result_ram != ram:
                            result["params"]["ram"] = ram
                            print(f"Fix ram in {f}")
                            f.write_text(json.dumps(result))
                        else:
                            print(f"Skip {f}")
            except Exception as e:
                print(e)


main()
