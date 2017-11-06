## Save docker container for further deployment
Below I use Docker image and container for CPU build, becuase GPU instances at AWS are expensive and you need additional approval from Amazon for their use.

### Create image from container
Docker has two main terms: image and container. You can think about them as image is a class and container is an object. All chages we made corresponf to a container, so if we want use it later for creation of other containers, it is not possible. We need create an image from running container. Docker allows us to do that with ```commit``` command.
```
docker commit 35027f92f2f8 $USER/tensorflow-serving-gan:v1.0
```

### Test new image
To test a created image, just create a new container from it:
```
docker run --name=tf_container_cpu_2 -it $USER/tensorflow-serving-gan:v1.0
```
and then start the server with our model:
```
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=gan --model_base_path=gan-export &> gan_log &
```
Check the log:
```
cat gan_log
```
Last should be something like this:
```
I tensorflow_serving/model_servers/main.cc:298] Running ModelServer at 0.0.0.0:9000 ...
```

### Drawbacks
The original image provided by TensorFlow Serving team is about 1.1 GB. Since we compile the Serving and downloads a couple of thing, our image is about 5.1 GB. We could optimize this size, but it is not a point of this article.

## Kubernetes in the Cloud
Kubernetes is all about automated container deployment, scailing and management. We already created a Docker container, now it is time to deploy it into the Cloud.
Kubernetes deployment 

## Kubernetes on Azure
### Setup
- Get Azure free trial account: [https://azure.microsoft.com/free/](https://azure.microsoft.com/free/). 
- Go to Azure portal: [https://portal.azure.com/](https://portal.azure.com/) to check that you have an access to it and it was setup correclty.
- Install [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli):
```
echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ wheezy main" | \
     sudo tee /etc/apt/sources.list.d/azure-cli.list
sudo apt-key adv --keyserver packages.microsoft.com --recv-keys 417A0893
sudo apt-get install apt-transport-https
sudo apt-get update && sudo apt-get install azure-cli
```
Check installed version
```
az --version
```
It equal or should be greater than 2.0.x

### Create Kubernetes cluster
- Login to Azure:
```
az login
```
Follow instructions in the Terminal and in the browser. If everything worked then you should see something like this in the Terminal:
```
[
  {
    "cloudName": "AzureCloud",
    "id": "baf24db5-8f0e-462c-8749-bfbaae59fc8d",
    "isDefault": true,
    "name": "Free Trial",
    "state": "Enabled",
    "tenantId": "048f92c7-dedc-4820-8af4-a45d8a9ab90b",
    "user": {
      "name": "first_name.last_name@mail.com",
      "type": "user"
    }
  }
]
```
- Create resource group. It is a logical group where you deploy your resources and manage them
```
az group create --name ganRG --location eastus
```
You should get a response something like
```
{
  "id": "/subscriptions/baf24db5-8f0e-462c-8749-bfbaae59fc8d/resourceGroups/ganRG",
  "location": "eastus",
  "managedBy": null,
  "name": "ganRG",
  "properties": {
    "provisioningState": "Succeeded"
  },
  "tags": null
}
```
You can double-check in Azure Portal under "Resource Group" that it was created
- Create Kubernetes cluster in Azure Container Service. We create 1 master node and 1 Linux agent nodes. Free trial limit the size of Cores to 4, i.e. wir have 2 for Master, 2 for Agent.
```
az acs create --orchestrator-type=kubernetes \
    --resource-group ganRG \
    --name=ganK8sCluster \
    --agent-count=1 \
    --generate-ssh-keys 
```
You will get generated SSH keys in the default location if the don't already exist.
Wait a couple of minutes... If everything went OK, then you should see a response like that:
```
{
  "id": "/subscriptions/baf24db5-8f0e-462c-8749-bfbaae59fc8d/resourceGroups/ganRG/providers/Microsoft.Resources/deployments/azurecli1498767761.777487522811",
  "name": "azurecli1498767761.777487522811",
  "properties": {
    "correlationId": "8f874429-7cf7-4adb-a970-c41e9d3c0cc4",
    "debugSetting": null,
    "dependencies": [],
    "mode": "Incremental",
    "outputs": null,
    "parameters": {
      "clientSecret": {
        "type": "SecureString"
      }
    },
    "parametersLink": null,
    "providers": [
      {
        "id": null,
        "namespace": "Microsoft.ContainerService",
        "registrationState": null,
        "resourceTypes": [
          {
            "aliases": null,
            "apiVersions": null,
            "locations": [
              "eastus"
            ],
            "properties": null,
            "resourceType": "containerServices"
          }
        ]
      }
    ],
    "provisioningState": "Succeeded",
    "template": null,
    "templateLink": null,
    "timestamp": "2017-06-29T20:31:17.359107+00:00"
  },
  "resourceGroup": "ganRG"
}
```
You can also check in Azure Portal that for the resource group ganRG we have 2 virtual machines - k8s-master-... and k8s-agent-...


### kubectl
```
sudo az acs kubernetes install-cli 
```

### Connect to Azure with kubectl
- Get credentials:
```
az acs kubernetes get-credentials --resource-group=ganRG --name=ganK8sCluster
```
- Verify connection:
```
kubectl get nodes
```
You should see something like this:
```
NAME                    STATUS                     AGE       VERSION
k8s-agent-bb8987c3-0    Ready                      7m        v1.6.6
k8s-master-bb8987c3-0   Ready,SchedulingDisabled   7m        v1.6.6
```
**CAUTION**  
Azure create two virtual machines. Be aware to stop or deallocate virtual machines if you do not use them to avoid extra costs:
```
az vm [stop|deallocate] --resource-group=ganRG --name=k8s-agent-...
az vm [stop|deallocate] --resource-group=ganRG --name=k8s-master-...
```
you can start them again with:
```
az vm start --resource-group=ganRG --name=k8s-agent-...
az vm start --resource-group=ganRG --name=k8s-master-...
```

## Deploy Docker container into Kubernetes
Now as we setup our Kubernetes cluster it is time to deploy into it!

### Azure container registry
First we need Container Registry to push our Docker image and get it later for deployment to Kibernetes. We create it in Azure:
```
az acr create --name=ganEcr --resource-group=ganRG --sku=Basic
```
You should get something like this:
```
{
  "adminUserEnabled": false,
  "creationDate": "2017-06-29T20:36:21.082996+00:00",
  "id": "/subscriptions/baf24db5-8f0e-462c-8749-bfbaae59fc8d/resourceGroups/ganRG/providers/Microsoft.ContainerRegistry/registries/ganEcr",
  "location": "eastus",
  "loginServer": "ganecr.azurecr.io",
  "name": "ganEcr",
  "provisioningState": "Succeeded",
  "resourceGroup": "ganRG",
  "sku": {
    "name": "Basic",
    "tier": "Basic"
  },
  "storageAccount": {
    "name": "ganecr203541"
  },
  "tags": {},
  "type": "Microsoft.ContainerRegistry/registries"
}
```
Enable Admin user on the registry:
```
az acr update -n ganEcr --admin-enabled true
```

### Upload Docker image
Now we need to upload our docker image.  
Obtain admin user credentials:
```
az acr credential show --name=ganEcr
```
You should get something like this:
```
{
  "passwords": [
    {
      "name": "password",
      "value": "=bh5wXWOUSrJtKPHReTAgi/bijQCkjsq"
    },
    {
      "name": "password2",
      "value": "OV0Va1QXv=GPL+sGm9ZossmvgIoYBdif"
    }
  ],
  "username": "ganEcr"
}
```
Login into Azure Registry Containner:
```
docker login ganecr.azurecr.io -u=ganEcr -p=<password value from credentials>
```
Tag docker image in such way:
```
docker tag $USER/tensorflow-serving-gan:v1.0 ganecr.azurecr.io/tensorflow-serving-gan
```

Now push it into Azure Registry Container:
```
docker push ganecr.azurecr.io/tensorflow-serving-gan
```
It will take a while, so be patient :-)

### Kubernetes. Key concepts
Kubernetes cluster consists of physical **nodes**. 

#### Node
Node is a worker machine in the Kubernetes cluster (Virtual Machine or bare metal). It is managed by the master component and has all services to run **pods*** (see below). Those services include, for example, Docker, which is important for us.

#### Pod
Pod is a group of one or more containers (one Docker container in our case), the shared storage for those containers, and options about how to run the containers. So pod is a logical host for tightly coupled containers.  
In our pod we have only one Docker container, namely, our TensorFlow GAN Serving container.  
In our configuration we specified replicas - acually the number of pods that we want to create. Those pods, in our configuration, run on the same node, since we have only one worker node (the master node do not run user containers).

#### Service
Pods are mortal. They are born and die. ReplicationControllers in particular create and destroy Pods dynamically. While each Pod gets its own IP address, even those IP addresses cannot be relied upon to be stable over time. So if we have pods that need talk to each other (e.g. frontend to backend) or we want to access some pods externally (in our case), then we have a problem.  
Kubernetes **services** solve it. This is an abstraction, which defines a logical set of pods and policy to access them. In our case we have one service thta abstacts two pods.

### Deployment configuration
We store deployment

### Kubectl - Kubernetes command line tool
For all operations we use [kubectl](https://kubernetes.io/docs/user-guide/kubectl-overview/) - command line interface for running commands against Kubernetes clusters. We will use following commands:
- _kubectl create_ to create pods and service specified in the deployment configuration
- _kubectl get_ to get the infomation about pods and service

### Where to find all information
All information regarrding our Kubernetes cluster is store in ~/.kube/config.  
You can find here the information about Kubernetes cluster, users, certificates.  

### Deployment
For deployment we use kubectl - command line interface for running commands against Kubernetes clusters.

- Create Kubernetes deployment
```
cd <path to GAN project>
kubectl create -f gan_k8s.yaml
```
You should get:
```
deployment "gan-deployment" created
service "gan-service" created
```
Check that pods and service are created successfully:
```
kubectl get pods
```
should return something like:
```
NAME                              READY     STATUS    RESTARTS   AGE
gan-deployment-3500298660-3gmkj   1/1       Running   0          24m
gan-deployment-3500298660-h9g3q   1/1       Running   0          24m
```
It takes a time before the STATUS goes to Running. 
```
kubectl get services
```
should return something like:
```
NAME          CLUSTER-IP     EXTERNAL-IP    PORT(S)          AGE
gan-service   10.0.134.234   40.87.62.198   9000:30694/TCP   24m
kubernetes    10.0.0.1       <none>         443/TCP          7h
```
It takes some time before our gan-service get an external IP address. Only after that you can issue requests.  
To get the details information about the service execute
```
kubectl describe services gan-service
```
You should see details about it:
```
Name:			gan-service
Namespace:		default
Labels:			run=gan-service
Annotations:		<none>
Selector:		app=gan-server
Type:			LoadBalancer
IP:			10.0.134.234
LoadBalancer Ingress:	40.87.62.198
Port:			<unset>	9000/TCP
NodePort:		<unset>	30694/TCP
Endpoints:		10.244.0.10:9000,10.244.0.11:9000
Session Affinity:	None
Events:
  FirstSeen	LastSeen	Count	From			SubObjectPath	Type		Reason			Message
  ---------	--------	-----	----			-------------	--------	------			-------
  22m		22m		1	service-controller			Normal		CreatingLoadBalancer	Creating load balancer
  20m		20m		1	service-controller			Normal		CreatedLoadBalancer	Created load balancer
```
Here you notice thta we have a load balancer that forwards requests to out pods (you see that we have 2 endpoints - one for each node). IP here is an IP of our Kubernetes service (aka cluster IP) and it is not accessible externally.

### Check functioning
Now we should issue the same command as we did to check functioning of the serving in a Docker container locally. As a server address we have to take _LoadBalancer Ingress_. It is an IP address that is visible externally.
```
cd <path to GAN project>
python svnh_semi_supervised_client.py --server=40.87.62.198:9000 --image=./svnh_test_images/image_3.jpg
```
You should get a known result:
```
outputs {
  key: "scores"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 10
      }
    }
    float_val: 8.630897802584857e-17
    float_val: 1.219293777054986e-09
    float_val: 6.613714575998131e-10
    float_val: 1.5203355241411032e-09
    float_val: 0.9999998807907104
    float_val: 9.070973139291283e-12
    float_val: 1.5690838628401593e-09
    float_val: 9.12262028080068e-17
    float_val: 1.0587883991775016e-07
    float_val: 1.0302327879685436e-08
  }
}
```
Congratulations! You deployed the GAN model in the Cloud in a Kubernetes cluster. Kubernetes scales, load balances and manages our GAN model in reliable way.
