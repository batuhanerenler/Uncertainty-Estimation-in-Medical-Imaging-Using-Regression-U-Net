from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

def unet_model(input_size=(256, 256, 1), num_classes=1, dropout_rate=0.5):
    inputs = Input(input_size)
    
    # Convolutional block function
    def conv_block(input_tensor, num_filters):
        x = Conv2D(num_filters, 3, padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = Conv2D(num_filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        return x

    # Downsampling
    conv1 = conv_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(dropout_rate)(pool1)

    conv2 = conv_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(dropout_rate)(pool2)

    # Bottleneck
    conv3 = conv_block(pool2, 256)

    # Upsampling
    up1 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv3)
    up1 = concatenate([up1, conv2])
    up1 = Dropout(dropout_rate)(up1)
    conv4 = conv_block(up1, 128)

    up2 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv4)
    up2 = concatenate([up2, conv1])
    up2 = Dropout(dropout_rate)(up2)
    conv5 = conv_block(up2, 64)

    # Regresyon çıktısı (tedavi dozu tahmini)
    output1 = Conv2D(num_classes, 1, activation='linear', padding='same')(conv5)

    # Giriş yeniden yapılandırma çıktısı
    output2 = Conv2D(input_size[2], 1, activation='sigmoid', padding='same')(conv5)

    model = Model(inputs=inputs, outputs=[output1, output2])
    return model

# Modeli oluştur
model = unet_model()

# Modeli derle
model.compile(optimizer=Adam(learning_rate=1e-4), 
              loss={'conv2d_22': 'mean_squared_error', 'conv2d_23': 'binary_crossentropy'},
              metrics={'conv2d_22': 'mae', 'conv2d_23': 'accuracy'})

# Modelin özeti
model.summary()

# Örnek veri seti ve eğitim
import numpy as np

# Kurgusal veri seti oluşturulması
num_samples = 100
input_size = (256, 256, 1)
X_train = np.random.random((num_samples, *input_size))
Y_train_dose = np.random.random((num_samples, *input_size))  # Tedavi dozu
Y_train_recon = np.copy(X_train)  # Yeniden yapılandırma için giriş verisi

# Modelin eğitilmesi
model.fit(X_train, {'conv2d_22': Y_train_dose, 'conv2d_23': Y_train_recon}, epochs=10, batch_size=2)
