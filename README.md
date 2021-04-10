# ResnetAgePrediciton
Pytorch framework age Prediction by brain scans: cortical thickness, cortical curvature and cortical myelination. 

Using custom built resnet as describes in the article:
"Identity Mappings in Deep Residual Networks"
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
https://arxiv.org/abs/1603.05027

Elastic deformation and gaussian noise transforms added to imporve validation loss and generalizability. 

At Epoch 50 Final 
at 506 sample size

![training error](https://user-images.githubusercontent.com/43177212/114280841-21711200-9a33-11eb-9599-d8fbbf0e443b.png)
![validation error](https://user-images.githubusercontent.com/43177212/114280844-22a23f00-9a33-11eb-9b61-a10022336b8e.png)

Final validation MSELoss = 0.047

Note: Data Unavailable for upload to do confidentiality. 
