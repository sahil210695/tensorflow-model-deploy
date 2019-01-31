# TensorFlow Serving for model deployment in production

We have build a model and now we want that model to be accessible from web. There are many ways to do that but we are going to do that using **Tensorflow Serving**.

### Steps
1. Train a dumb model and save that model [Train](train.py)
2. Restore saved model to check everything is working [Restore](restore.py)
3. Export saved model to format that can be used by tensorflow-serving [Export](export.py)
4. Load exported model to check everything is working [Load exported](load_exported.py)
5. Deploy exported model in tensorflow-serving using docker image

#### Train a dumb model and save that model
We will generate random datasaet and create a very basic model that will learn relationship 
between our randomly generated dataset

_Relation between our dataset_
![relation_dataset](images/X%20and%20y%20plot.png)

_Plot for Loss history_
![loss_history](images/epcoh%20loss%20plot.png)

when training gets completed it will save model in [Save Directory](saved_model) using `tf.train.Saver()` and it will also save summary in [Summary](summary) to visualize the graph in tensorboard.

_Tensorflow graph_
![tensorflow_graph](images/tensorboard%20graph.png)

#### Restore saved model to check everything is working
Here will load the saved model and make prediction using that by some input

#### Export saved model to format that can be used by tensorflow-serving
Here comes the interesting part where will take the saved model and export it to [SavedModel](https://www.tensorflow.org/api_docs/python/tf/saved_model) format that tensorflow-serving will use to serve. To do that first we need load meta graph file and fetch tensors from graph which are required for predictions using their names then build tensor info from them that will be used to create signature definition that will be passed to the SavedModelBuilder instance. We can finally build model signature that identifies what serving is going to expect from the client.

Now let's take a look at the export directory.
```
$ ls serve/linear/
1/
```
A sub-directory will be created for exporting each version of the model.
```
$ ls serve/linear/1/
saved_model.pb  variables/
```
We'll use the command line utility saved_model_cli to look at the MetaGraphDefs (the models) and SignatureDefs (the methods you can call) in our SavedModel.
```
$ saved_model_cli show --dir serve/linear/1/ --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['x'] tensor_info:
        dtype: DT_FLOAT
        shape: (1)
        name: data_pipeline/IteratorGetNext:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['y'] tensor_info:
        dtype: DT_FLOAT
        shape: (1)
        name: prediction/add:0
  Method name is: tensorflow/serving/predict
```

#### Load exported model to check everything is working
Here will load the exported **SavedModel** and make prediction using that by some random input

#### Deploy exported model in tensorflow-serving using docker image
One of the easiest ways to get started using TensorFlow Serving is with Docker.

General installation instructions are on the Docker site, but we give some quick links here:
- [Docker for macOS](https://docs.docker.com/docker-for-mac/install/)
- [Docker for Windows](https://docs.docker.com/docker-for-windows/install/) for Windows 10 Pro or later
- [Docker Toolbox](https://docs.docker.com/toolbox/) for much older versions of macOS, or versions of Windows before Windows 10 Pro

Lets serve our model
1. you can pull the latest TensorFlow Serving docker image by running <br>
  `docker pull tensorflow/serving`

2. Run docker image that will serve your model <br>
  `sudo docker run --name tf_serving -p 8501:8501 --mount type=bind,source=$(pwd)/linear/,target=/models/linear -e MODEL_NAME=linear -t tensorflow/serving`

  _if all goes well you'll see logs like_ 😎
  ```
  2019-01-31 07:55:30.133911: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:259] SavedModel load for tags { serve }; Status: success. Took 58911 microseconds.
  2019-01-31 07:55:30.133937: I tensorflow_serving/servables/tensorflow/saved_model_warmup.cc:83] No warmup data file found at /models/linear/1/assets.extra/tf_serving_warmup_requests
  2019-01-31 07:55:30.134095: I tensorflow_serving/core/loader_harness.cc:86] Successfully loaded servable version {name: linear version: 1}
  2019-01-31 07:55:30.137262: I tensorflow_serving/model_servers/server.cc:286] Running gRPC ModelServer at 0.0.0.0:8500 ...
  [warn] getaddrinfo: address family for nodename not supported
  2019-01-31 07:55:30.143359: I tensorflow_serving/model_servers/server.cc:302] Exporting HTTP/REST API at:localhost:8501 ...
  ```

3. In addition to gRPC APIs TensorFlow ModelServer also supports RESTful APIs.
- Model status API: It returns the status of a model in the ModelServer <br>
  `GET http://host:port/v1/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]`
  
  Go to your browser type `http://localhost:8501/v1/models/linear/versions/1` and you'll get status <br>
  ```
  {
     "model_version_status":[
        {
           "version":"1",
           "state":"AVAILABLE",
           "status":{
              "error_code":"OK",
              "error_message":""
           }
        }
     ]
  }
  ```
  
- Model Metadata API: It returns the metadata of a model in the ModelServer <br>
  `GET http://host:port/v1/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/metadata`
  
  Go to your browser type `http://localhost:8501/v1/models/linear/versions/1/metadata` and you'll get metadata <br>
  ```
  {
     "model_spec":{
        "name":"linear",
        "signature_name":"",
        "version":"1"
     },
     "metadata":{
        "signature_def":{
           "signature_def":{
              "serving_default":{
                 "inputs":{
                    "x":{
                       "dtype":"DT_FLOAT",
                       "tensor_shape":{
                          "dim":[
                             {
                                "size":"1",
                                "name":""
                             }
                          ],
                          "unknown_rank":false
                       },
                       "name":"data_pipeline/IteratorGetNext:0"
                    }
                 },
                 "outputs":{
                    "y":{
                       "dtype":"DT_FLOAT",
                       "tensor_shape":{
                          "dim":[
                             {
                                "size":"1",
                                "name":""
                             }
                          ],
                          "unknown_rank":false
                       },
                       "name":"prediction/add:0"
                    }
                 },
                 "method_name":"tensorflow/serving/predict"
              }
           }
        }
     }
  }
  ```
  
- Predict API: It returns the predictions from you model based on the inputs that you'll send <br>
  `POST http://host:port/v1/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]:predict`
  
  Let's make a curl request and check we are getting predictions or not <br>
  `curl -d '{"signature_name":"serving_default", "instances":[10, 20, 30]}' -X POST http://localhost:8501/v1/models/linear:predict` <br> <br>
  **output**
  ```
  {
     "predictions":[
        38.4256,
        68.9495,
        99.4733
     ]
  }
  ```

Now you can use anything to make **POST** request to the model server and even use Flask, Django or any other framework of you choice to integrate this **REST API** and make application on top it 😎👍✌️
