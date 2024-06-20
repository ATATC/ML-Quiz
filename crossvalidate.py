from medpy import metric
from skimage import io


def crossvalidate(dir_a: str, dir_b: str, num_cases: int) -> tuple[float, float, float, float, float, float]:
    wcdc, ndc = 0, 0
    min_wcdc = None
    max_wcdc = None
    min_ndc = None
    max_ndc = None
    for i in range(num_cases):
        label_seg = io.imread(f"{dir_a}/case_{(n := str(i).zfill(3))}.png")
        label_gt = io.imread(f"{dir_b}/case_{n}.png")
        whole_cell_dc = metric.binary.dc(label_seg > 0, label_gt > 0)
        wcdc += whole_cell_dc
        if min_wcdc is None or whole_cell_dc < min_wcdc:
            min_wcdc = whole_cell_dc
        if max_wcdc is None or whole_cell_dc > max_wcdc:
            max_wcdc = whole_cell_dc
        nucleus_dc = metric.binary.dc(label_seg == 2, label_gt == 2)
        ndc += nucleus_dc
        if min_ndc is None or nucleus_dc < min_ndc:
            min_ndc = nucleus_dc
        if max_ndc is None or nucleus_dc > max_ndc:
            max_ndc = nucleus_dc
    wcdc, ndc = wcdc / num_cases, ndc / num_cases
    return wcdc - min_wcdc, wcdc, max_wcdc - wcdc, ndc - min_ndc, ndc, max_ndc - ndc


def print_validation(a: float, b: float, c: float, d: float, e: float, f: float) -> str:
    return (f"Whole Cell DC: {b * 100:.2f} (-{a * 100:.2f}/+{c * 100:.2f})\n"
            f"Nucleus DC: {e * 100:.2f} (-{d * 100:.2f}/+{f * 100:.2f})")


if __name__ == '__main__':
    print(print_validation(*crossvalidate("seg-Internal-UNet", "seg-Internal-Trans", 77)))
    print(print_validation(*crossvalidate("seg-External-UNet", "seg-External-Trans", 60)))
