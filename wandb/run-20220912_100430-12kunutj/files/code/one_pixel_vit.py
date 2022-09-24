from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.utils as vutils
from timm.models import create_model
import torchvision.transforms as transforms
import torch
from OnePatch import OnePatchAttack
from cPixelAttack import cOnePixel
import matplotlib.pyplot as plt
import  numpy as np
import torchvision.datasets as dataset
import wandb
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import torch.nn as nn


#uncomment to log with wandb
wandb.init(project= "one_pixel_attack_deit_small_thesis")
artifact = wandb.Artifact("deit_small_1px_samples_" + str(wandb.run.id) , type="prediction")
columns = ["id", "scr_img", "adv_image", "ground_truth", "pred b4 attack", "confidence", "gradcam_scr", "gradcam_adv", "pred after attack", "adv confidence"]
log_table = wandb.Table(columns=columns)

def show_img(im):
    k = im.view(3, 224, 224)
    #k = im.squeeze(0).permute(1, 2, 0).detach().cpu()
    k = k.permute(1, 2, 0).detach().cpu()
    plt.imshow(k)
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = create_model('deit_small_patch16_224', pretrained=True)
#net =  create_model("small_patch16_224_hierarchical", pretrained=True)
net.to(device)
net.eval()

WIDTH = 224
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
normalize = transforms.Normalize( mean=mean, std=std)
data_transform = transforms.Compose([
    transforms.Resize(WIDTH),
    transforms.CenterCrop(WIDTH),
    transforms.ToTensor(),
    #normalize
    ])
data_to_pil = transforms.ToPILImage()

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

target_layers = [net.blocks[-1].norm1]
cam1 = GradCAM(model=net, target_layers=target_layers, use_cuda=True, reshape_transform=reshape_transform)
#cam2 = GradCAM(model=net, target_layers=target_layers, use_cuda=True, reshape_transform=reshape_transform)

path_to_resources = '/home/abass.abdulsalam/Documents/imgnet/processed_val/'
batch_size = 1
test_dataset = dataset.ImageFolder(root=path_to_resources, transform=data_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             num_workers=2, shuffle=True)


#scr = OnePatchAttack(net)
scr = cOnePixel(net)
with tqdm(test_loader, unit = 'batch', position = 0, leave=True) as p_bar:
    success = 0
    fool_rate = 0
    total = 0

    for batch_idx, (data, labels) in enumerate(p_bar):
        data = data.to(device)
        gt_label = labels.to(device)

        output =  net(data)
        #output = torch.vstack(output)
        maxval, pred_before_attack = output.data.max(1, keepdim=True)
        total += 1

        attacked_img = scr.forward(data, labels)
        adv_output = net(attacked_img)
        # adv_output = torch.vstack(adv_output)
        maxval_adv, pred_after_attack = adv_output.data.max(1, keepdim=True)

        if pred_after_attack != gt_label:
            success += 1

        if pred_after_attack != pred_before_attack:
            fool_rate += 1
        
        


        targets_cam = [ClassifierOutputTarget(gt_label.item())]

        grayscale_cam_scr = cam1(input_tensor=data, targets=targets_cam, aug_smooth=True)
        grayscale_cam_scr = grayscale_cam_scr[0, :]
        rgb_img_scr = np.float32(data.squeeze(0).permute(1, 2, 0).cpu())
        gradcam_img_scr = show_cam_on_image(rgb_img_scr, grayscale_cam_scr)

        grayscale_cam_adv = cam1(input_tensor=attacked_img, targets=targets_cam, aug_smooth=True)
        grayscale_cam_adv = grayscale_cam_adv[0, :]
        rgb_img_adv = np.float32(attacked_img.squeeze(0).permute(1, 2, 0).cpu())
        gradcam_img_adv = show_cam_on_image(rgb_img_adv, grayscale_cam_adv)
        
            # if batch_idx < 20:
            #     vutils.save_image(data.data, "./%s/%d_%d_original.png" % (
            #     "logs/one_patch/three", batch_idx, labels),
            #                         normalize=True)
            #     vutils.save_image(attacked_img.data, "./%s/%d_%d_adversarial_%d.png" % (
            #         "logs/one_patch/three", batch_idx, labels, label_adv),
            #                         normalize=True)

        # p_bar.set_description("Attack Success: {:.2f}, FR:{:.2f}, Queries: {}".format((success * 100 / total), \
        #                         (fool_rate * 100 /total), scr.required_iterations))

        wandb.log({"Attack Success" : (success * 100 / total),
                   "Fool rate" : (fool_rate * 100 /total),
                   #"queries" : scr.required_iterations[0],
                   "number of pixels":4})

        log_table.add_data(batch_idx, wandb.Image(data), wandb.Image(attacked_img), gt_label.item(), pred_before_attack.item(), maxval.item(),\
                           wandb.Image(gradcam_img_scr), wandb.Image(gradcam_img_adv), pred_after_attack.item(), maxval_adv.item())
    artifact.add(log_table, "predictions")
    wandb.run.log_artifact(artifact)
        #wandb.log({"Deit_tiny" : log_table})

#print(scr.required_iterations[-1])
