from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import math
import resnet as models
from dataset_test3D import dataset3D
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


cuda = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
source1_name = "GE"
source2_name = 'Philips'
target_name = "Siemens"
dataset = "ADNI2"

IMG_PATH = '../Longitudinal_ADNI2'
results_dir = './Results'
img_size = 224


db_test = dataset3D(base_dir=IMG_PATH, list_dir='./Dataset', split="test_ADNI2_Siemens_MNI")
target_test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
print("The length of target test set is: {}".format(len(db_test)))


def test(model, loader):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    total = 0
    
    with torch.no_grad():
        for data_3D, target in loader:
            temp_correct = 0
            temp_correct1 = 0
            temp_correct2 = 0
            if cuda:
                data_3D, target = data_3D.cuda(), target.cuda()
            for slice_number in range(95, 125):    #this slice range in the 2D coronal plane used to train the model            
                temp_x = data_3D[:, :, :, slice_number, :]
                temp_x = temp_x.repeat(1, 3, 1, 1)
                data, target = Variable(temp_x), Variable(target)
                pred1, pred2 = model(data, mark = 0)
                pred1 = torch.nn.functional.softmax(pred1, dim=1)
                pred2 = torch.nn.functional.softmax(pred2, dim=1)
                pred = (pred1 + pred2) / 2
                test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
                pred = pred.data.max(1)[1]
                temp_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                pred = pred1.data.max(1)[1]
                temp_correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
                pred = pred2.data.max(1)[1]
                temp_correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()
                
            if temp_correct >= 15:
                correct = correct + 1
            if temp_correct1 >= 15:
                correct1 = correct1 + 1
            if temp_correct2 >= 15:
                correct2 = correct2 + 1
            total = total +1
                
        print("Total samples: ", total)                                   
        test_loss /= len(loader.dataset)
        print(target_name, ' Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(loader.dataset),
            100. * correct / len(loader.dataset)))
        print('source1 correct: {}, source2 correct: {}, Average correct: {}'.format(correct1, correct2, correct))
    return correct

if __name__ == '__main__':
    model = models.DAMS(num_classes=2)
    if cuda:
        model.cuda()
    model.load_state_dict(torch.load(results_dir+'/'+dataset + '_' + source1_name + '_' + source2_name + '_to_' + target_name + '_max_accuracy.pth'))
    test_correct = test(model, target_test_loader)
    
    print("Testset performance: ", dataset,  source1_name, source2_name, "to", target_name, "Testset correct:", test_correct, 'Accuracy: {}/{} ({:.0f}%)'.format(
    test_correct, len(target_test_loader.dataset), 100. * test_correct / len(target_test_loader.dataset)), "\n")

