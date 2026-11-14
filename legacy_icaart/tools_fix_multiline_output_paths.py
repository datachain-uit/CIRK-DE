import json
import re
from pathlib import Path


BASE = Path(r"D:\feature-selection-experiments\notebooks")

FILE_LINE_RE = re.compile(
    r'^(?P<indent>\s*)(?P<quote>["\'])(?P<name>[^"\']+\.(?:csv|xlsx|json|txt|pt|pth|pkl))(?P=quote)(?P<tail>\s*,?.*)$'
)


def should_wrap(previous_lines: list[str], line: str) -> bool:
    if "output_path(" in line:
        return False
    context = "\n".join(previous_lines[-4:])
    if "Path(" in context or "os.path.join" in context:
        return False
    return any(token in context for token in [".to_csv(", ".to_excel(", ".to_json(", "pd.read_csv(", "open("])


def main():
    for path in sorted(BASE.glob("*_with_metrics.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        changed = False

        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            lines = "".join(cell.get("source", [])).splitlines(keepends=True)
            new_lines = []
            for line in lines:
                stripped_newline = "\n" if line.endswith("\n") else ""
                body = line[:-1] if stripped_newline else line
                match = FILE_LINE_RE.match(body)
                if match and should_wrap(new_lines, body):
                    body = (
                        f'{match.group("indent")}output_path('
                        f'{match.group("quote")}{match.group("name")}{match.group("quote")})'
                        f'{match.group("tail")}'
                    )
                    changed = True
                new_lines.append(body + stripped_newline)
            cell["source"] = new_lines

        if changed:
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"fixed {path.name}")


if __name__ == "__main__":
    main()
