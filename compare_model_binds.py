import classifier, detector, detector_v2
import os
from detector_train_v2 import parse_xml
from classifier_train import get_img_array



a = classifier.get_model()
a.load_weights("classifier.h5"),
b = classifier.get_model()
b.load_weights("classifier_save2.h5"),
c = classifier.get_model()
c.load_weights("my_model.h5")
classifiers = [a,b,c]
a = detector.get_model()
a.load_weights("detector.h5"),
b = detector_v2.get_model()
b.load_weights("detector_v2.h5"),
c = detector_v2.get_model()
c.load_weights("detector_v2_podgorelo.h5")
detectors = [a,b,c]



CLASSIFIER_DIM = (150,150) 
DETECTOR_DIM = (224,224)
path = "./train_dataset_dataset/dataset/tests2/"
labels = [parse_xml(f'{path}{i}') for i in os.listdir(path) if i.endswith(".xml")]
images_paths = [f'{path}{i}' for i in os.listdir(path) if i.endswith(".jpg")]
input_data = get_img_array(images_paths, DETECTOR_DIM)

detector_outs = []
for d in detectors:
    res = d.predict(input_data)
    detector_outs.append(res)
print(detector_outs)

for c in classifiers:
    pass
        


