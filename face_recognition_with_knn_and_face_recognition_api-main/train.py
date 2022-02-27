from detect import return_box_faces #sử dụng lại bài hôm trước

from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import math

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    Y = []

    # lặp lại từng người trong dữ liệu ( tập train)
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # duyệt qua từng ảnh của người hiện tại
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):

            image = face_recognition.load_image_file(img_path)
            # sử dụng phần 1 nhận diện khuôn mặt hàm return_box_faces trả về khuôn mặt
            face_bounding_boxes = return_box_faces(image=image)

            if len(face_bounding_boxes) != 1:
                #nếu có nhiều hơn 1 khuôn mặt thì bỏ qua ảnh đó
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # thêm mã hóa của khuông mặt vào mảng của dữ liệu để chuẩn bị huấn luyện
                # X là dữ liệu mã hóa , Y là tên người ( label)
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                Y.append(class_dir)

    #xác định số cụm knn hay số người cần nhận diện
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # tạo và train
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, Y)

    # lưu model KNN
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return knn_clf

if __name__ == '__main__':
    print("Bắt đầu trainning.........")
    train(train_dir="train",model_save_path="model.clf")
    print("Train kết thúc")