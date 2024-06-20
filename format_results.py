from typing import Literal
from os import listdir, rename, remove


def format_results(path: str, prefix: Literal["Internal", "External"]) -> None:
    for fn in listdir(path):
        if fn.endswith("png"):
            rename(f"{path}/{fn}", f"{path}/{fn.replace('case', f'{prefix}_img')[:-4]}_label.png")
        else:
            remove(f"{path}/{fn}")


if __name__ == '__main__':
    format_results("seg-Internal-UNet", "Internal")
    format_results("seg-External-UNet", "External")
    format_results("seg-Internal-Trans", "Internal")
    format_results("seg-External-Trans", "External")
