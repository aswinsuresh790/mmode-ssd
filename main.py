import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


import tensorflow as tf
#from collections import defaultdict
#from io import StringIO

cap = cv2.VideoCapture(1)
#from utils import label_map_util
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util
#from utils import visualization_utils as vis_util







def detect_objects(our_image):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    col1, col2 = st.beta_columns(2)

    col1.subheader("Original Image")
    st.text("")
    plt.figure(figsize = (15,15))
    plt.imshow(our_image)
    col1.pyplot(use_column_width=True)

    # SSD ALGORITHM
    
    import urllib.request

    url1 = 'https://github.com/AlanRSET/mSSD/releases/download/SSD/frozen_inference_graph.pb'
    hf1 = url1.split('/')[-1]

    urllib.request.urlretrieve(url1, hf1)
    PATH_TO_CKPT = hf1

    # List of the strings that is used to add correct label for each box.
    
    url2 = 'https://github.com/AlanRSET/mSSD/releases/download/SSD/label_map.pbtxt'
    hf2 = url2.split('/')[-1]

    urllib.request.urlretrieve(url2, hf2)
    
    PATH_TO_LABELS = hf2

    NUM_CLASSES = 6

    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.compat.v1.GraphDef()#tf.GraphDef()
      with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid: #tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    def load_image_into_numpy_array(image):
      (im_width, im_height) = image.size
      return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)

    with detection_graph.as_default():
      with tf.compat.v1.Session(graph=detection_graph) as sess:
          image = our_image
          image = image.convert('RGB')
          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
          image_np = load_image_into_numpy_array(image)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how  .
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)

          img=cv2.resize(image_np, (800,600))#edited

    st.text("")
    col2.subheader("Object-Detected Image")
    st.text("")
    plt.figure(figsize = (15,15))
    plt.imshow(img)
    col2.pyplot(use_column_width=True)

    a=[]
    for i in range(len(scores[0])):
      if(scores[0][i]>0.5):
        a.append(i)

    final_class=[]
    for i in a:
      final_class.append(classes[0][i].astype(np.int32))
    final_class.sort()
    final_class=list(set(final_class))

    #st.write("Identified features:")

    k="Identified features: "
    l=0
    for i in final_class:
      if(i==1):
        k=k+("A-LINE")
        l=1
      elif(i==2):
        k=k+("CONTINUOUS PLEURAL LINE")
        l=1
      elif(i==3):
        k=k+("FLUID")
        l=1
      elif(i==4):
        k=k+("IRREGULAR PLEURAL LINE")
        l=1
      elif(i==5):
        k=k+("SEASHORE PATTERN")
        l=1
      elif(i==6):
        k=k+("SINUSOIDAL PLEURAL LINE")
      else:
        k=k+("ERROR")
      if((l==1) & (i!=final_class[-1])):
        k=k+", "

    st.write(k)
    #print("**************************")
    #st.write("Pathology identified:")
    if(final_class[0]==1 and final_class[1]==2 and final_class[2]==5):
      st.success("Pathology identified: NORMAL LUNG")
    elif(final_class[0]==3 and final_class[1]==4):
      st.success("Pathology identified: CONSOLIDATION")
      st.write("Lung consolidation occurs when the air that usually fills the small airways in your lungs is replaced with something else.")
    elif(final_class[0]==3 and final_class[1]==6):
      st.success("Pathology identified: PLEURAL EFFUSION")
      st.write("Pleural Effusion is an excessive buildup of fluid in the space between your lungs and chest cavity.")
    else:
      st.success("Pathology identified: ERROR")



    #st.success("The scan is of {}".format(prediction(arr)))


def object_main():
    """SSD"""

    st.title("M-mode SSD")
    st.write("SSD takes only one shot to detect multiple objects present in an image using multibox. It is significantly faster in speed and high-accuracy object detection algorithm. SSD has a base VGG-16 network followed by multibox conv layers")

    #choice = st.radio("", ("Show Demo", "Browse an Image"))
    #st.write()

    st.set_option('deprecation.showfileUploaderEncoding', False)
    image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

    if image_file is not None:
        our_image = Image.open(image_file)
        detect_objects(our_image)



if __name__ == '__main__':
    object_main()
