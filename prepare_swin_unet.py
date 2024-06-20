from PIL import Image
from shutil import copy


def remove_alpha_channel(image: Image) -> Image:
    if image.mode == 'RGBA':
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    elif image.mode == 'LA':
        background = Image.new("L", image.size, 255)
        background.paste(image, mask=image.split()[1])
        return background
    else:
        return image.convert("RGB")


if __name__ == '__main__':
    for i in range(158):
        tiff_image = Image.open(f"data/imagesTr/img_{(n := str(i).zfill(3))}.tiff")
        tiff_image = tiff_image.resize((224, 224))
        tiff_image = remove_alpha_channel(tiff_image)
        tiff_image.save(f"raw/Dataset001_NAME1/imagesTr/case_{n}_0000.png", "PNG")

    for i in range(158, 231):
        jpg_image = Image.open(f"data/imagesTr/img_{(n := str(i).zfill(3))}.jpg")
        jpg_image = jpg_image.resize((224, 224))
        jpg_image = remove_alpha_channel(jpg_image)
        jpg_image.save(f"raw/Dataset001_NAME1/imagesTr/case_{n}_0000.png", "PNG")

    for i in range(231):
        label_image = Image.open(f"data/labelsTr/img_{(n := str(i).zfill(3))}_label.png")
        label_image = label_image.resize((224, 224))
        label_image = remove_alpha_channel(label_image)
        label_image.save(f"raw/Dataset001_NAME1/labelsTr/case_{n}.png", "PNG")

    for i in range(53):
        tiff_image = Image.open(f"data/imagesTs-Internal/Internal_img_{(n := str(i).zfill(3))}.tiff")
        tiff_image = remove_alpha_channel(tiff_image)
        tiff_image.save(f"imagesTs-Internal/case_{n}_0000.png", "PNG")

    for i in range(53, 77):
        jpg_image = Image.open(f"data/imagesTs-Internal/Internal_img_{(n := str(i).zfill(3))}.jpg")
        jpg_image = remove_alpha_channel(jpg_image)
        jpg_image.save(f"imagesTs-Internal/case_{n}_0000.png", "PNG")

    for i in range(32):
        tiff_image = Image.open(f"data/imagesTs-External/External_img_{(n := str(i).zfill(3))}.tiff")
        tiff_image = remove_alpha_channel(tiff_image)
        tiff_image.save(f"imagesTs-External/case_{n}_0000.png", "PNG")

    for i in range(32, 60):
        jpg_image = Image.open(f"data/imagesTs-External/External_img_{(n := str(i).zfill(3))}.jpg")
        jpg_image = remove_alpha_channel(jpg_image)
        jpg_image.save(f"imagesTs-External/case_{n}_0000.png", "PNG")
