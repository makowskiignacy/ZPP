import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, samples
from attacks.foolboxattacks.projected_gradient_descent import ProjectedGradientDescentInf

if __name__ == '__main__':
    model = models.resnet18(pretrained=True).eval()
    fmodel = PyTorchModel(model, bounds=(0, 1))
    data = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))

    result = ProjectedGradientDescentInf().conduct(model, data)

    print(result)