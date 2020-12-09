from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from util import path, data, misc, generator as gen
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras import models
from dip import dip
import setting.constant as const
import importlib
import sys
import cv2
import numpy as np

class NeuralNetwork():
    def __init__(self): 
        self.arch = importlib.import_module("%s.%s.%s" % (const.dn_NN, const.dn_ARCH, const.MODEL))

        self.fn_logger = path.fn_logger()
        self.fn_checkpoint = path.fn_checkpoint()

        self.dn_image = path.dn_train(const.dn_IMAGE)
        self.dn_aug_image = path.dn_aug(const.dn_IMAGE, mkdir=False)

        self.dn_label = path.dn_train(const.dn_LABEL)
        self.dn_aug_label = path.dn_aug(const.dn_LABEL, mkdir=False)

        self.dn_test = path.dn_test()
        self.dn_test_out = path.dn_test(out_dir=True, mkdir=False)

        try:
            self.model = self.arch.model(self.has_checkpoint())
            if (self.has_checkpoint()):
                print("Loaded: %s\n" % self.fn_checkpoint)
        except Exception as e:
            sys.exit("\nError loading: %s\n%s\n" % (self.fn_checkpoint, str(e)))

    def has_checkpoint(self):
        return self.fn_checkpoint if path.exist(self.fn_checkpoint) else None

    def prepare_data(self, images, labels=None):
        if (labels is None):
            for (i, image) in enumerate(images):
                number = ("%0.3d" % (i+1))
                path_save = path.join(self.dn_test_out, mkdir=True)
                image, _ = dip.preprocessor(image, None)
                original_name = (const.fn_PREPROCESSING % (number))
                data.imwrite(path.join(path_save, original_name), image)

                yield self.arch.prepare_input(image)
        else:
            for (image, label) in zip(images, labels):
                (image, label) = dip.preprocessor(image, label)
                yield self.arch.prepare_input(image), self.arch.prepare_input(label)

    def save_predict(self, original, image):
        path_save = path.join(self.dn_test_out, mkdir=True)

        with open(path.join(path_save, (const.fn_SEGMENTATION)), 'w+') as f:
            for (i, image) in enumerate(image):
                number = ("%0.3d" % (i+1))
                image_name = (const.fn_PREDICT % (number))
                img = image
                image = dip.posprocessor(original[i], self.arch.prepare_output(image))
                data.imwrite(path.join(path_save, image_name), image)
                image_pp,_ = dip.preprocessor(original[i], None)
                
                n_white_pix = np.sum(image_pp == 255)
                n_black_pix = np.sum(image_pp == 0)
                f.write(("Image, %s, ground truth area,%s, \n" % (number, (n_white_pix/n_black_pix)) ))
                
                seg = (image == 255).sum()
                f.write(("Image, %s was approximately,%s, segmented (,%s ,pixels),\n" % (number, (seg/image.size), seg)))

                original_name = (const.fn_ORIGINAL % (number))
                data.imwrite(path.join(path_save, original_name), original[i])

                img_m, _, width = dip.measure(image, None)
                data.imwrite(path.join(path_save, image_name), img_m)

                n_white_pix = np.sum(img_m == 255) 
                n_black_pix = np.sum(img_m == 0)
                f.write(("Image, %s, predict area ,%s, \n" % (number, (n_white_pix/n_black_pix))))
                f.write(("Image, %s, predict width ,%s, \n" % (number, str(width))))

                f1score = f1_score(np.argmax(image_pp, axis=1), np.argmax(img, axis=1), average= 'weighted', labels=np.unique(np.argmax(img, axis=1)))
                precisionscore = precision_score(np.argmax(image_pp, axis=1), np.argmax(img, axis=1), average= 'weighted', labels=np.unique(np.argmax(img, axis=1)))
                recallscore = recall_score(np.argmax(image_pp, axis=1), np.argmax(img, axis=1), average='weighted', labels=np.unique(np.argmax(img, axis=1)))

                f.write(("Image, %s, F1 ,%s, \n" % (number, str(f1score))))
                f.write(("Image, %s, Recall ,%s, \n" % (number, str(precisionscore))))
                f.write(("Image, %s, Precision ,%s, \n" % (number, str(recallscore))))

                overlay_name = (const.fn_OVERLAY % (number))
                overlay = dip.overlay(img_m, original[i])
                data.imwrite(path.join(path_save, overlay_name), overlay)

        f.close()

def train():
    nn = NeuralNetwork()

    total = data.length_from_path(nn.dn_image, nn.dn_aug_image)
    q = misc.round_up(total, 100) - total

    if (q > 0):
        print("Dataset augmentation (%s increase) is necessary (only once)\n" % q)
        gen.augmentation(q)

    images, labels = data.fetch_from_paths([nn.dn_image, nn.dn_aug_image], [nn.dn_label, nn.dn_aug_label])
    images, labels, v_images, v_labels = misc.random_split_dataset(images, labels, const.p_VALIDATION)
    
    epochs, steps_per_epoch, validation_steps = misc.epochs_and_steps(len(images), len(v_images))

    print("Train size:\t\t%s |\tSteps per epoch: \t%s\nValidation size:\t%s |\tValidation steps:\t%s\n" 
        % misc.str_center(len(images), steps_per_epoch, len(v_images), validation_steps))

    patience, patience_early = const.PATIENCE, int(epochs*0.25)
    loop, past_monitor = 0, float('inf')

    checkpoint = ModelCheckpoint(nn.fn_checkpoint, monitor=const.MONITOR, save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor=const.MONITOR, min_delta=const.MIN_DELTA, patience=patience_early, restore_best_weights=True, verbose=1)
    logger = CSVLogger(nn.fn_logger, append=True)

    while True:
        loop += 1
        h = nn.model.fit_generator(
            shuffle=True,
            generator=nn.prepare_data(images, labels),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_steps=validation_steps,
            validation_data=nn.prepare_data(v_images, v_labels),
            use_multiprocessing=False,
            callbacks=[checkpoint, early_stopping, logger])

        val_monitor = h.history[const.MONITOR]
        
        if ("loss" in const.MONITOR):
            val_monitor = min(val_monitor)
            improve = (past_monitor - val_monitor)
        else:
            val_monitor = max(val_monitor)
            improve = (val_monitor - past_monitor)

        print("\n##################")
        print("Finished epoch (%s) with %s: %f" % (loop, const.MONITOR, val_monitor))

        if (abs(improve) == float("inf") or improve > const.MIN_DELTA):
            print("Improved from %f to %f" % (past_monitor, val_monitor))
            past_monitor = val_monitor
            patience = const.PATIENCE
            test(nn)
        elif (patience > 0):
            print("Did not improve from %f" % (past_monitor))
            print("Current patience: %s" % (patience))
            patience -= 1
        else:
            break
        print("##################\n")

def test(nn=None):
    if nn is None:
        nn = NeuralNetwork()

    if (nn.has_checkpoint()):
        images = data.fetch_from_path(nn.dn_test)
        generator = nn.prepare_data(images)
        results = nn.model.predict_generator(generator, len(images), verbose=1)
        nn.save_predict(images, results)

    else:
        print(">> Model not found (%s)\n" % nn.fn_checkpoint)

def convlayer(nn=None):
    if nn is None:
        nn = NeuralNetwork( )

    if (nn.has_checkpoint()):
        images = data.fetch_from_path(nn.dn_test)
        generator = nn.prepare_data(images)
        
        layer_outputs = [layer.output for layer in nn.model.layers[:]]

        results = models.Model(inputs=nn.model.inputs, outputs=layer_outputs).predict_generator(generator, verbose=1)

        layer_names = []
        for layer in nn.model.layers[:]:
            layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
            
        for layer_name, layer_activation in zip(layer_names, results): # Displays the feature maps
            plt.title(layer_name)
            plt.imshow(layer_activation[0,:,:,0], aspect='auto', cmap='viridis')
            plt.savefig('C:/Users/isaac/Downloads/Crackdetection/out/conv' + layer_name + '.png')   
