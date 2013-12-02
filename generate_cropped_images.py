import Image
import ImageDraw
from collections import defaultdict
import numbers
import os
import numpy as np


# Fix bug with Image/ImageDraw
import math

if not hasattr(ImageDraw.Image, "isNumberType"):
    def isNumberType(obj):
        return isinstance(obj, numbers.Number)
    ImageDraw.Image.isNumberType = isNumberType


def get_bounding_boxes(ground_truth_file_name):
    ground_truths_file = open(ground_truth_file_name)
    indent_amount = "    "
    current_folder = None
    bounding_boxes = defaultdict(list)  # Stores a list of tuples of (minx, miny, maxx, maxy)
    for line in ground_truths_file:
        line = line[:-1]  # get rid of trailing newline
        if not line or line[0] == "#":
            # Ignore blank lines and lines that start with a hash
            continue
        if line.startswith(indent_amount):
            if current_folder is None:
                raise ValueError("ground truths file must have unindented containing"
                                 "folder name before lines with file information")
            parts = line.strip().split(" ")
            if len(parts) != 13:
                raise ValueError("Face information must have 13 parts (a file name and 12 coordinates). "
                                 "Had %s instead. Line was %s" % (len(parts), line))

            full_name = os.path.join(current_folder, parts[0])

            coordinates = map(int, map(float, parts[1:]))  # There are some float values in the ground truth file
            left_eye = (coordinates[0], coordinates[1])
            right_eye = (coordinates[2], coordinates[3])
            #nose = (coordinates[4], coordinates[5])
            #left_corner_mouth = (coordinates[6], coordinates[7])
            center_mouth = (coordinates[8], coordinates[9])
            #right_corner_mouth = (coordinates[10], coordinates[11])



            # Approximate distance from the mouth to the ear line as the distance
            # of the center of the mouth to midpoint between the eyes.
            midpoint = (np.array(left_eye) + np.array(right_eye))/2.0
            height_dir = midpoint - np.array(center_mouth)
            distance = math.sqrt(height_dir.dot(height_dir))
            height_dir /= distance
            ratio = 1.61803398874989484820458683  # Approximately (phi + 1)/phi = phi
            half_height = distance * ratio

            width_dir = np.array(left_eye) - np.array(right_eye)
            eye_distance = math.sqrt(width_dir.dot(width_dir))
            width_dir = width_dir / eye_distance
            half_width = (5 * eye_distance) / 4.0

            # Expand by 20%
            half_height *= 1.2
            half_width *= 1.2

            rotated_top_left = tuple(midpoint + half_height * height_dir + half_width * width_dir)
            rotated_top_right = tuple(midpoint + half_height * height_dir - half_width * width_dir)
            rotated_bottom_left = tuple(midpoint - half_height * height_dir + half_width * width_dir)
            rotated_bottom_right = tuple(midpoint - half_height * height_dir - half_width * width_dir)

            # Make axis aligned bounding box
            x_vals = (rotated_top_left[0], rotated_top_right[0], rotated_bottom_left[0], rotated_bottom_right[0])
            y_vals = (rotated_top_left[1], rotated_top_right[1], rotated_bottom_left[1], rotated_bottom_right[1])

            min_x = min(x_vals)
            max_x = max(x_vals)
            min_y = min(y_vals)
            max_y = max(y_vals)

            bounding_boxes[full_name].append(map(int, (min_x, min_y, max_x, max_y)))

            #top_right = (max_x, min_y)
            #bottom_left = (min_x, max_y)
            #full_input_name = os.path.join(src_folder, current_folder, parts[0])
            #times_seen_image = image_counts[full_input_name]
            #image_counts[full_input_name] += 1
            #output_name = parts[0].replace(".gif", "_%s.bmp" % times_seen_image)
            #full_output_name = os.path.join(dst_folder, current_folder, output_name)
            #orig_image = Image.open(full_input_name)
            # Draw on points, to see where they're placed
            #orig_image = orig_image.convert("RGB")
            #draw = ImageDraw.Draw(orig_image)
            #red = "#f00"
            #draw.line(left_eye + right_eye, fill=red)
            #draw.line(left_corner_mouth + center_mouth, fill=red)
            #draw.line(center_mouth + right_corner_mouth, fill=red)
            #draw.point(nose, fill=red)
            #orig_image.show()
            #draw.line(top_left + top_right, fill=red)
            #draw.line(top_right + bottom_right, fill=red)
            #draw.line(bottom_right + bottom_left, fill=red)
            #draw.line(bottom_left + top_left, fill=red)
            #orig_image.crop(map(int, top_left + bottom_right)).save(full_output_name)
            #orig_image.show()

        else:
            current_folder = line
    return bounding_boxes


def extract_faces(src_folder, dst_folder, bounding_boxes, target_aspect_ratio, target_height, target_width):
    # For each bounding box, crop image and save to corresponding location in output dir
    for file_name, boxes in bounding_boxes.iteritems():
        output_dir = os.path.join(dst_folder, os.path.dirname(file_name))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        full_input_name = os.path.join(src_folder, file_name)
        without_extension = os.path.extsep.join(file_name.split(os.path.extsep)[:-1])
        output_name_template = os.path.join(dst_folder, without_extension + "_%s" + os.path.extsep + "bmp")
        orig_image = Image.open(full_input_name)
        for i, (left, top, right, bottom) in enumerate(boxes):
            full_output_name = output_name_template % i
            cropped = orig_image.crop((left, top, right, bottom)).copy()
            height = bottom - top
            width = right - left
            #other_width = height / target_aspect_ratio
            other_height = width * target_aspect_ratio
            scale_to_height = height < other_height
            if scale_to_height:
                scale_factor = target_height / float(height)
                new_width = int(width * scale_factor)
                cropped.thumbnail((new_width, target_height), Image.ANTIALIAS)  # Modifies image

                # Crop to correct aspect ratio
                half_width_diff = int((new_width - target_width) / 2)
                new_left = half_width_diff
                new_right = new_width - half_width_diff
                if new_right - new_left > target_width:
                    if new_right - new_left - target_width != 1:
                        print "More than one left over width pixel!!!"  # This should never happen
                    # To account for a fractional (eg .5) half_width_diff
                    new_right -= new_right - new_left - target_width
                final = cropped.crop((new_left, 0, new_right, target_height))
            else:
                # Change width
                scale_factor = target_width / float(width)
                new_height = int(height * scale_factor)
                cropped.thumbnail((target_width, new_height), Image.ANTIALIAS)  # Modifies image

                # Crop to correct aspect ratio
                half_height_diff = int((new_height - target_height) / 2)
                new_top = half_height_diff
                new_botom = new_height - half_height_diff
                if new_botom - new_top > target_height:
                    if new_botom - new_top - target_height != 1:
                        print "More than one left over height pixel!!!"  # This should never happen
                    # To account for a fractional (eg .5) half_height_diff
                    new_top += new_botom - new_top - target_height
                final = cropped.crop((0, new_top, target_width, new_botom))
            final.save(full_output_name)


def extract_negative_patches(src_dir, dst_dir, bounding_boxes, patch_height, patch_width):
    current_patch = 0
    patch_file_template = os.path.join(dst_dir, "negative_%s.bmp")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for file_name, boxes in bounding_boxes.iteritems():
        full_input_name = os.path.join(src_dir, file_name)
        orig_image = Image.open(full_input_name)
        width, height = orig_image.size
        num_patches_wide = width / patch_width
        num_patches_tall = height / patch_height
        for x in range(num_patches_wide):
            for y in range(num_patches_tall):
                patch_left = x*patch_width
                patch_right = patch_left + patch_width
                patch_top = y*patch_height
                patch_bottom = patch_top + patch_height
                can_use_patch = True
                for left, top, right, bottom in boxes:
                    if not ((patch_right <= left or patch_left >= right) and
                                (patch_top >= bottom or patch_bottom <= top)):
                        # Overlaps with a face's bounding box
                        can_use_patch = False
                if not can_use_patch:
                    continue
                full_output_file_name = patch_file_template % current_patch
                orig_image.crop((patch_left, patch_top, patch_right, patch_bottom)).save(full_output_file_name)
                current_patch += 1


def show_histogram(values, num_buckets=50):
    avg = sum(values) / len(values)
    values = sorted(values)
    median = values[len(values) / 2]
    print "mean value", avg, "median value", median
    min_value = min(values)
    max_value = max(values)
    bucket_size = (max_value - min_value) / float(num_buckets)
    buckets = [0 for i in range(num_buckets)]
    for value in values:
        bucket = int((value - min_value) / bucket_size)  # The int function floors the value
        if value == max_value:
            bucket -= 1  # Special case for highest value
        buckets[bucket] += 1
    for bucket, num in enumerate(buckets):
        low_range = min_value + bucket_size * bucket
        high_range = min_value + bucket_size * (bucket + 1)
        graph_bar = "*" * num
        print "%s - %s: %s" % (low_range, high_range, graph_bar)


def show_aspect_ratio_stats(bounding_boxes):
    ratios = []
    for boxes in bounding_boxes.itervalues():
        for left, top, right, bottom in boxes:
            height = bottom - top
            width = right - left
            ratios.append(height / float(width))
    show_histogram(ratios)


def show_size_stats(bounding_boxes, target_aspect_ratio):
    scales = []
    for boxes in bounding_boxes.itervalues():
        for left, top, right, bottom in boxes:
            height = bottom - top
            width = right - left
            other_width = height / target_aspect_ratio
            other_height = width * target_aspect_ratio
            # One should be bigger than actual, one should be smaller
            # We want the smaller pair
            height = min(height, other_height)
            width = min(width, other_width)
            # We can ignore width when measuring scale
            scales.append(height)
    show_histogram(scales)
    print sorted(scales)
    show_histogram([s for s in scales if s < 160])


def main():
    src_folder = "uncropped_images"
    faces_dst_folder = "cropped_images"
    negatives_dst_folder = "negative_examples"
    bounding_boxes = get_bounding_boxes("ground_truth.txt")
    #show_aspect_ratio_stats(bounding_boxes)
    #show_size_stats(bounding_boxes, 1.25)
    cropped_height = 40
    cropped_width = 32
    #extract_faces(src_folder, faces_dst_folder, bounding_boxes, 1.25, cropped_height, cropped_width)
    extract_negative_patches(src_folder, negatives_dst_folder, bounding_boxes, cropped_height, cropped_width)


if __name__ == "__main__":
    main()