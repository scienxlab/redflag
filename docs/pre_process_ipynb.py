
import sys
import glob
import re
import json


def change_kernel(notebook):
    """
    Vanillafy the kernelspec.
    """
    new_kernelspec = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    notebook['metadata']['kernelspec'].update(new_kernelspec)
    return notebook


def main(path):
    """
    Process the IPYNB files in path, save in place (side-effect).
    """
    fnames = glob.glob(path.strip('/') + '/[!_]*.ipynb')  # Not files with underscore.
    for fname in fnames:
        with open(fname) as f:
            notebook = json.loads(f)

        new_nb = change_kernel(notebook)

        with open(fname, 'w') as f:
            _ = f.write(json.dumps(new_nb))

    return


if __name__ == '__main__':
    _ = main(sys.argv[1])
