
from PIL import Image, ImageDraw
import os


def create_image_composition(image_path, bounding_box, cutout1, cutout2):
    """
    Creates a composition of the original image and zoomed-in cutouts, with colored boxes.
    
    Parameters:
        image_path (str): Path to the input image.
        bounding_box (tuple): Bounding box for cropping (left, upper, right, lower).
        cutout1 (tuple): Cutout specifications (width, height, offset_x, offset_y).
        cutout2 (tuple): Cutout specifications (width, height, offset_x, offset_y).
        
    Returns:
        Image: The final composed image.
    """
    # Load the image
    image = Image.open(image_path)
    
    # Crop the image using the bounding box
    left, upper, size = bounding_box
    cropped_image = image.crop((left, upper, left+size, upper+size*2))
    
    # Get dimensions of the cropped image
    width, height = cropped_image.size
    
    # Create zoomed-in cutouts that exactly fill the designated space
    cutout1 = (cutout1[0], cutout1[1], cutout1[0] + cutout1[2], cutout1[1] + cutout1[2])
    cutout2 = (cutout2[0], cutout2[1], cutout2[0] + cutout2[2], cutout2[1] + cutout2[2])
    zoomed_in_cutout1 = cropped_image.crop(cutout1).resize((width, width))
    zoomed_in_cutout2 = cropped_image.crop(cutout2).resize((width, width))
    
    # Create a new image with double the width to accommodate the cutouts
    final_image = Image.new("RGB", (2 * width, height))
    
    # Draw the original cropped image on the left
    final_image.paste(cropped_image, (0, 0))
    
    # Draw the zoomed-in cutouts on the right
    final_image.paste(zoomed_in_cutout1, (width, 0))
    final_image.paste(zoomed_in_cutout2, (width, width))
    
    # Draw colored boxes around the cutouts and the position in the original image
    draw = ImageDraw.Draw(final_image)
    
    # Draw a red box around the original cropped image position
    linewidth = 8
    draw.rectangle(cutout1, outline="red", width=linewidth)
    draw.rectangle(cutout2, outline="blue", width=linewidth)

    draw.rectangle((width, 0, width*2, width), outline="red", width=linewidth*2)
    draw.rectangle((width, width, width*2, width*2), outline="blue", width=linewidth*2)
    
    return final_image



settings = {
    "original": [("original")],
    "G-PCC": [(34, 0.75), (40, 0.5), (22, 0.875)],
    "IT-DL-PCC": [(0, 0.001), (0, 0.002), (0, 0.004), (0, 0.0005)],
    "CVPR_inverse_scaling": [(0.1, 0.1), (0.2, 0.2), (0.4,0.4), (1.0, 1.0)]
}

# Sequence ((Bounding Box), Cutout1, Cutout2)
# Boxes are defined by (x,y,sizex,sizey)
sequences = {
    "longdress": ((780, 190, 350), (125, 30, 80), (150, 190, 80), "top"),
    "loot": ((780, 190, 350), (160, 5, 100), (190, 215, 100), "top"),
    "soldier": ((780, 190, 370), (100, 10, 100), (210, 70, 100), "top"),
    "redandblack": ((780, 180, 360), (100, 6, 100), (95, 220, 100), "top"),
    "dancer": ((750, 160, 400), (150, 130, 80), (200, 250, 80), "top"),
    "exercise": ((780, 190, 360), (125, 10, 100), (200, 615, 100), "bottom"),
    "model": ((780, 180, 380), (125, 10, 100), (180, 130, 100), "bottom"),
    "basketball_player": ((750, 170, 400), (160, 10, 100), (220, 260, 100), "bottom")
}
path_skeleton = "./results/{}/renders_test/{}_a{}_g{}_{}.png"
results_dir = "./plot/images"

def crop_images():
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for sequence, boxes in sequences.items():
        for method, setting in settings.items():
            for set in setting:
                bbox = boxes[0]
                cutout1 = boxes[1]
                cutout2 = boxes[2]
                view = boxes[3]

                print(set)
                if method == "original":
                    dir, file = os.path.split(path_skeleton)
                    image_path = os.path.join(dir, "{}_original_{}.png").format("G-PCC", sequence, view)
                    result_path = os.path.join(results_dir, "render_{}_{}.png".format(sequence, "original"))
                    print(result_path)
                else:
                    image_path = path_skeleton.format(method, sequence, set[0], set[1], view)
                    result_path = os.path.join(results_dir, "render_{}_{}_a{}_g{}.png".format(sequence, method, set[0], set[1]))

                final_image = create_image_composition(image_path, bbox, cutout1, cutout2)

                # Save image to result_path
                final_image.save(result_path)


crop_images()