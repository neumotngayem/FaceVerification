ROOT_FACEKNOWLEDGE_DATABASE = 'knowledge_database'
CASE_REGISTER = 'r'
CASE_VERIFICATION = 'v'
DETECTION_OUTPUT_FILENAME = 'detection_output.jpg'

def crop_image(yolo_result, image_face):
    # bounding boxes coordinate (left, top, right, bottom)
    boxes = yolo_result.boxes.xyxy
    # only crop the largest face area
    final_box = boxes[0]
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        area_box = (x2 - x1) * (y2 - y1)
        area_final_box = (final_box[2] - final_box[0]) * (final_box[3] - final_box[1])
        # compare the bounding box area
        if area_box > area_final_box:
            final_box = box
    # crop the image with padding
    image_face_crop = image_face[int(final_box[1] - 50):int(final_box[3] + 25), int(final_box[0] - 50):int(final_box[2] + 50)]
    return image_face_crop