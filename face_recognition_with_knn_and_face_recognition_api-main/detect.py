import cv2
import numpy as np

prototxt_path = "face_detector/deploy.prototxt"
# Net weight
caffemodel_path = "face_detector/weights.caffemodel"

CONFIDENCE = 0.6
detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)  # load model

def return_box(image):

    (h, w) = image.shape[:2]
    # prepares image for entrance on the model
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # put image on model
    detector.setInput(blob)
    # propagates image through model
    detections = detector.forward()
    # check confidance of 200 predictions
    list_box = []
    for i in range(0, detections.shape[2]):
        # box --> array[x0,y0,x1,y1]
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        # confidence range --> [0-1]
        confidence = detections[0, 0, i, 2]

        if confidence >= CONFIDENCE:
            if list_box == []:
                list_box = np.expand_dims(box, axis=0)
            else:
                list_box = np.vstack((list_box, box))

    return list_box #trả về list các dự đoán
def return_box_faces(image):
    list_box = return_box(image)
    try:
        box_faces = [(box[1], box[2], box[3], box[0]) for box in list_box.astype("int")]
    except:
        box_faces = []
    #trả về danh sách các khuôn mặt detect đc dạng list tuple điểm, 1 tuple là 1 khuon mat
    #tuple dạng (y0,x1,y1,x0)
    return box_faces
def show_image_after_detect(path_image):
    image = cv2.imread(path_image)
    box_faces = return_box_faces(image)
    for (y0,x1,y1,x0) in box_faces:
        w = x1 - x0
        h = y1 - y0
        cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h), (255, 0, 0), 2)
    cv2.imshow("detect face",image)
    if cv2.waitKey(0) == 27:
        exit(0)
def show_frame_real_time(frame):
    box_faces = return_box_faces(frame)
    for (y0, x1, y1, x0) in box_faces:
        w = x1 - x0
        h = y1 - y0
        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (255, 0, 0), 2)
        cv2.imshow(winname="Face", mat=frame)
    return frame
if __name__ == '__main__':

    #  show_image_after_detect("khautrang.jpg")

    vd = cv2.VideoCapture(0)
   
    # show_frame_real_time(vd)
    while True:
        _, frame = vd.read()
        # Convert image into grayscale
        # gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

        # # Use detector to find landmarks
        # faces = detector(gray)

        # for face in faces:
        #     x1 = face.left()  # left point
        #     y1 = face.top()  # top point
        #     x2 = face.right()  # right point
        #     y2 = face.bottom()  # bottom point
        #     # Draw a rectangle
        #     cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(
        #         x2, y2), color=(0, 255, 0), thickness=4)

        # # show the image
        # cv2.imshow(winname="Face", mat=frame)

        # # Exit when escape is pressed
        # if cv2.waitKey(delay=1) == 27:
        #     
        show_frame_real_time(frame)
        if cv2.waitKey(delay=1) == 27:
            break
    vd.release()

    # Close all windows
    cv2.destroyAllWindows()
        

