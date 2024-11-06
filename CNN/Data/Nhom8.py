import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image

# Khởi tạo mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes for A, B, C
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Chuẩn bị dữ liệu
train_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
    'C:\\Users\\LG\\Documents\\Machine_Learning\\CNN\\Data', 
    target_size=(64, 64), 
    batch_size=32, 
    class_mode='categorical'
)

# Huấn luyện mô hình
model.fit(training_set, epochs=25)

# Dự đoán
def predict_letter(img_path):
    test_image = image.load_img(img_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    prediction_index = np.argmax(result[0])  # Lấy chỉ số lớp có xác suất cao nhất

    # Chuyển đổi chỉ số thành chữ cái
    if prediction_index == 0:
        prediction = 'A'
    elif prediction_index == 1:
        prediction = 'B'
    else:
        prediction = 'C'
    return prediction

# Sử dụng mô hình để dự đoán chữ cái từ một hình ảnh
print(predict_letter('C:\\Users\\LG\\Documents\\Machine_Learning\\CNN\\Data\\A\\KyTu_A.png'))
