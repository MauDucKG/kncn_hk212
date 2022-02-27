import pickle
import face_recognition
import cv2
from detect import return_box_faces
def predict(frame, model_path, distance_threshold=0.5):
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)
    #tìm kiếm khuôn mặt từ frame hình
    X_face_locations = return_box_faces(frame)

    # nếu k có khuôn mặt trả về mảng rỗng
    if len(X_face_locations) == 0:
        return []

    # chuyển khuôn mặt về dữ liệu mã hóa
    faces_encodings = face_recognition.face_encodings(frame, known_face_locations=X_face_locations)

    # sử dụng KNN để dự đoán và trả về kết quả
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    #Dự đoán tên người và xóa những người có kết quả dưới ngưỡng
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def predict_from_camera():
    cap = cv2.VideoCapture(0)
    while True:
        res,frame = cap.read()
        #nếu không có camera thoát chương trình
        if not res:
            exit(0)
        #tiến hành dự đoán
        predictions = predict(frame, model_path="model.clf")
        #show kết quả trong cmd
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))
        #vẽ lên frame và show frame
        for name, (top, right, bottom, left) in predictions:
            top_right = (right, top)
            bottom_left = (left, bottom + 22)
            bottom_right = (right, bottom)
            a = left
            b = bottom - top
            top_left = (top, left)
            cv2.rectangle(frame, top_right, bottom_left, (255, 0, 0), 3)
            cv2.putText(frame, str(name), (left, bottom), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 0, 0), 1, cv2.FILLED)
        cv2.imshow("detect", frame)
        if cv2.waitKey(24) == 27: #ấn esc để thoát
            exit(0)

if __name__ == '__main__':
    predict_from_camera()