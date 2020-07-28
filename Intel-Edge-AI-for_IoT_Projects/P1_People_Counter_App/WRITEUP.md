# Project Write-Up

## Explaining Custom Layers
Custom layers are layers that are not included in the list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

The process behind converting custom layers involves...

To actually add custom layers, there are a few differences depending on the original model framework. In both TensorFlow and Caffe, the first option is to register the custom layers as extensions to the Model Optimizer.

* For Caffe, the second option is to register the layers as Custom, then use Caffe to calculate the output shape of the layer. You’ll need Caffe on your system to do this option.
* For TensorFlow, its second option is to actually replace the unsupported subgraph with a different subgraph. The final TensorFlow option is to actually offload the computation of the subgraph back to TensorFlow during inference.

Some of the potential reasons for handling custom layers are...

The Inference Engine loads the layers from the input model IR files into the specified device plugin, which will search a list of known layer implementations for the device. If your topology contains layers that are not in the list of known layers for the device, the Inference Engine considers the layer to be unsupported and reports an error. To see the layers that are supported by each device plugin for the Inference Engine, refer to the Supported Devices documentation.
Note: If a device doesn't support a particular layer, an alternative to creating a new custom layer is to target an additional device using the HETERO plugin. The Heterogeneous Plugin may be used to run an inference model on multiple devices allowing the unsupported layers on one device to "fallback" to run on another device (e.g., CPU) that does support those layers.

## Comparing Model Performance

My method(s) to compare model ssd_mobilenet_v2_coco_2018_03_29 before and after conversion to Intermediate Representations
were...
* The initail (test) models were downloaded from the Tensorflow Model Zoo and were significantly larger in size compared to the Intel Pretrained Model that was selected.
* Size of model before conversion was 67MB (frozen_inference_graph.pb) and size of model after conversion was 65.11MB (bin+xml).
* Loading + inference time before conversion was (3 min + 10 sec)  and after conversion loading + inference time was (2min + 8 sec) 
* Accuracy was low for ssd_mobilenet_v2_coco_2018_03_29 in comparison to person-detection-retail-0013 model.
* The person-detection-retail-0013 model was the most accurate in detection of persons in the
frames.

compare the differences in network needs and costs of using cloud services as opposed to deploying at the edge...

* Edge models latency is low as compare to cloud,netwrking with cloud for inference takes lot of time.
* cost of the renting server at cloud is so high. where edge model can run on minimal cpu with local network connection.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

1. Security and safety: Monitorng number of people entering in sensitive or red zone area 
2. In retail sector: To monitor queue and direct people to less congested counters
3. Transporation: Monitoring queues and directing travellers for ticket collection or to boarding gates 
4. Crowd management: Counting no of people in gathering or meeting to manage crowd or collect insights about programmes
5. In manufacturing industries: To count people coming and going inside production unit to keep track of stength and monitoring purpose

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

* Lighting: Excessive light focus or dark images will create hindrance in image processing and affect prediction.
* Model accuracy: Models at edge need higher accuracy as in some cases such as security it is case sensitive. Higher accuracy means satisfactory and close to ground truth results.
* Camera focal length: Focal length affects amount of information captured from image.If focal length is shorter it can cover wider area and captures less information from image.If focal length is longer then it covers shorter area and great amount of information from image.
* Image size: Image size is fixed for models, higher resolution images take more memory space and requires higer inference time.If tradeoff of inference time is possible in less time critical applications then high resolution is good option to choose.

## Model Research


In investigating potential people counter models, I tried each of the following three models:

- Model 1: ssd_mobilenet_v2_coco
  - I converted the model to an Intermediate Representation with the following arguments
  -  $ wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
     $ tar -xvpf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
     $ cd ssd_mobilenet_v2_coco_2018_03_29/
     $ python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
 - The model was good for the app.
 - I tried to improve the model for the app by change prob_threshold to 0.3 and the model works well though missing draw box around people in some particular time.
  
- Model 2:  ssd_inception_v2_coco
  - I converted the model to an Intermediate Representation with the following arguments...
     $ wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
     $ tar -xvpf ssd_inception_v2_coco_2018_01_28.tar.gz
     $ cd ssd_inception_v2_coco_2018_01_28/
     $ python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
 - The model was insufficient for the app because it does not work correctly and it can't count correctly the people in the frame.
 - I tried to improve the model for the app by adust prob_threshold but still nothing improved.


- Model 3: faster_rcnn_inception_v2_coco
  - I converted the model to an Intermediate Representation with the following arguments...
     $ wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
     $ tar -xvpf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
     $ cd faster_rcnn_inception_v2_coco_2018_01_28/
     $ python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
  - The model was insufficient for the app because working better than previos one.
  - I tried to improve the model for the app by change prob_threshold to 0.3 but not obviously improved.

### Final Model Selection
After run inference on above models, the suitable accurate model was the one provided by Intel® [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) with the already existing in Intermediate Representations.
Model 4: [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)

Use this commad to dowload the model

python3  /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name person-detection-retail-0013 -o models/

# Run the app

python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm