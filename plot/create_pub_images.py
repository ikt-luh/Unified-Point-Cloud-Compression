import os
from PIL import Image, ImageDraw

settings = {
    "original": [("original")],
    "Main": {
        "soldier": (1, 0.4, 0.2),
        "bouquet": (1, 0.2, 0.1),
        #"shiva": (2, 0.3, 0.2),
        "House": (4, 0.4, 0.0),
        "CITISUP": (4, 0.4, 0.0),
    },
    "IT-DL-PCC": {
        "soldier": 0,
        "bouquet": 2,
        #"shiva": 2,
        "House": 3,
        "CITISUP": 1,
    },
    "G-PCC": {
        "soldier": 3,
        "bouquet": 2,
        #"shiva": 2,
        "House": 2,
        "CITISUP": 3,
    },
    "V-PCC": {
        "soldier": 3,
        "bouquet": 2,
        #"shiva": 2,
        "House": 1,
        "CITISUP": 2,
    },
}

sequences = {
    "soldier": ((780, 190, 370), (100, 10, 100), (210, 70, 100), 0, "top"),
    "bouquet": ((680, 0, 520), (320, 320, 100), (370, 500, 140), 0, "back"),
    #"EPFL": ((610, -100, 700), (360, 360, 100), (230, 500, 100), 0, "bottom"),
    "CITISUP": ((630, -110, 600), (140, 500, 140), (300, 550, 140), 270, "top"),
    "House": ((600, -200, 700), (310, 310, 100), (300, 680, 100),90, "top"),

    #"longdress": ((780, 190, 350), (125, 30, 80), (150, 190, 80), "top"),
    #"loot": ((780, 190, 350), (160, 5, 100), (190, 215, 100), "top"),
    #"redandblack": ((780, 180, 360), (100, 6, 100), (95, 220, 100), "top"),
    #"Arco": ((750, 160, 400), (150, 130, 80), (200, 250, 80), "top"),
}
path_skeleton = "../results/{}/renders_test/{}/{}_s{}_a{}_g{}_{}.png"
path_skeleton_related = "../results/{}/renders_test/{}/{}_R{}_{}.png"
results_dir = "./images"

def crop_images():
    """
    Run the cropping script over all settings
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for sequence, boxes in sequences.items():
        for method, setting in settings.items():
            bbox = boxes[0]
            cutout1 = boxes[1]
            cutout2 = boxes[2]
            rotation = boxes[3]
            view = boxes[4]

            if method == "original":
                dir, file = os.path.split(path_skeleton)
                print(sequence)
                print(view)
                image_path = os.path.join(dir, "{}_original_{}.png").format("G-PCC", sequence, sequence, view)
                result_path = os.path.join(results_dir, "render_{}_{}.png".format(sequence, "original"))
                print(result_path)
            else:
                set = setting[sequence]
                print(method)
                if method == "Main":
                    print(set)
                    image_path = path_skeleton.format(method, sequence, sequence, set[0], set[1], set[2], view)
                    result_path = os.path.join(results_dir, "render_{}_{}_s{}_a{}_g{}.png".format(sequence, method, set[0], set[1], set[2]))
                else:
                    image_path = path_skeleton_related.format(method, sequence, sequence, set, view)
                    result_path = os.path.join(results_dir, "render_{}_{}_R{}.png".format(sequence, method, set))

            final_image = create_image_composition(image_path, bbox, cutout1, cutout2, rotation)

            # Save image to result_path
            final_image.save(result_path)


def create_image_composition(image_path, bounding_box, cutout1, cutout2, rotation):
    """
    Creates a composition of the original image and zoomed-in cutouts, with colored boxes.
    
    Parameters:
        image_path (str): 
            Path to the input image.
        bounding_box (tuple): 
            Bounding box for cropping (left, upper, right, lower).
        cutout1 (tuple): 
            Cutout specifications (width, height, offset_x, offset_y).
        cutout2 (tuple): 
            Cutout specifications (width, height, offset_x, offset_y).
        
    Returns:
        final_image (PIL.Image): 
            The final composed image.
    """
    image = Image.open(image_path)
    
    # Crop Image
    left, upper, size = bounding_box

    if rotation:
        image = image.rotate(rotation, expand=False)

    cropped_image = safe_crop(image, (left, upper, left+size, upper+size*2))
    width, height = cropped_image.size
    
    # Compose cut-outs
    cutout1 = (cutout1[0], cutout1[1], cutout1[0] + cutout1[2], cutout1[1] + cutout1[2])
    cutout2 = (cutout2[0], cutout2[1], cutout2[0] + cutout2[2], cutout2[1] + cutout2[2])
    zoomed_in_cutout1 = cropped_image.crop(cutout1).resize((width, width))
    zoomed_in_cutout2 = cropped_image.crop(cutout2).resize((width, width))
    
    # Compose Final Image
    final_height = max(height, 2 * width)
    final_image = Image.new("RGB", (2 * width, height), (255,255,255))
    final_image.paste(cropped_image, (0, 0))
    final_image.paste(zoomed_in_cutout1, (width, 0))
    final_image.paste(zoomed_in_cutout2, (width, width))
    
    # Boxes for cut-outs
    linewidth = 8
    draw = ImageDraw.Draw(final_image)
    draw.rectangle(cutout1, outline="red", width=linewidth)
    draw.rectangle(cutout2, outline="blue", width=linewidth)
    draw.rectangle((width, 0, width*2, width), outline="red", width=linewidth*2)
    draw.rectangle((width, width, width*2, width*2), outline="blue", width=linewidth*2)
    
    return final_image

def safe_crop(image, box, fill=(255, 255, 255)):
    """Crops a region from image, filling areas outside the image with a fill color."""
    img_w, img_h = image.size
    left, upper, right, lower = box
    crop_w = right - left
    crop_h = lower - upper

    # Create new white canvas of desired crop size
    new_im = Image.new("RGB", (crop_w, crop_h), fill)

    # Calculate the region inside the image bounds
    int_left = max(left, 0)
    int_upper = max(upper, 0)
    int_right = min(right, img_w)
    int_lower = min(lower, img_h)

    # Only paste if there's an intersection
    if int_left < int_right and int_upper < int_lower:
        region = image.crop((int_left, int_upper, int_right, int_lower))
        paste_x = int_left - left
        paste_y = int_upper - upper
        new_im.paste(region, (paste_x, paste_y))

    return new_im

if __name__ == "__main__":
    crop_images()