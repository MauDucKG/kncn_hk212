import pickle
import face_recognition
import cv2
from detect import return_box_faces
def predict(img_path, model_path, distance_threshold=0.5):
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)
    # tải ảnh và tìm kiếm khuôn mặt
    X_img = face_recognition.load_image_file(img_path)
    X_face_locations = return_box_faces(X_img)

    # nếu k có khuôn mặt trả về mảng rỗng
    if len(X_face_locations) == 0:
        return []

    # chuyển khuôn mặt về dữ liệu mã hóa
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # sử dụng KNN để dự đoán và trả về kết quả
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    #Dự đoán tên người và xóa những người có kết quả dưới ngưỡng
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
def show_prediction_labels_on_image(img_path, predictions):
    #prediction là mảng những người đã dự đoán được bằng hàm predict
    #load hình ảnh từ file ảnh
    image = face_recognition.load_image_file(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #vẽ khung và tên lên trên ảnh
    for name, (top, right, bottom, left) in predictions:
        top_right = (right, top)
        bottom_left = (left, bottom + 22)
        bottom_right = (right, bottom)
        a = left
        b = bottom - top
        top_left = (top, left)
        cv2.rectangle(image, top_right, bottom_left, (255, 0, 0), 3)
        cv2.putText(image, str(name), (left, bottom), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 0, 0), 1, cv2.FILLED)
    #show hình ảnh
    cv2.imshow(img_path, image)
    cv2.waitKey(0)
    cv2.destroyWindow(img_path)
if __name__ == '__main__':
    prediction = predict(img_path="test.jpg",model_path="model.clf")
    show_prediction_labels_on_image(img_path="test.jpg",predictions=prediction)