from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.utils as vutils
from timm.models import create_model
import torchvision.transforms as transforms
import torch
from OnePatch import OnePatchAttack
import matplotlib.pyplot as plt
import  numpy as np
import torchvision.datasets as dataset
import wandb
wandb.init(project= "one_patch_attack_deit_small_thesis")
artifact = wandb.Artifact("deit_small1_samples_" + str(wandb.run.id) , type="prediction")
columns = ["id", "image", "ground_truth", "pred b4 attack", "confidence", "adv image", "pred after attack", "adv confidence"]
log_table = wandb.Table(columns=columns)

def show_img(im):
    k = im.view(3, 224, 224)
    #k = im.squeeze(0).permute(1, 2, 0).detach().cpu()
    k = k.permute(1, 2, 0).detach().cpu()
    plt.imshow(k)
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = create_model('deit_small_patch16_224', pretrained=True)
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

path_to_resources = '/home/abass.abdulsalam/Documents/imgnet/processed_val/'
batch_size = 1
test_dataset = dataset.ImageFolder(root=path_to_resources, transform=data_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             num_workers=2, shuffle=True)
scr = OnePatchAttack(net)

with tqdm(test_loader, unit = 'batch', position = 0, leave=True) as p_bar:
    success = 0
    fool_rate = 0
    total = 0

    for batch_idx, (data, labels) in enumerate(p_bar):
        data = data.to(device)
        gt_label = labels.to(device)

        output =  net(data)
        maxval, pred_before_attack = output.data.max(1, keepdim= True)

        total += 1

        attacked_img = scr.forward(data, labels)
        adv_output = net(attacked_img)
        maxval_adv, pred_after_attack = adv_output.data.max(1, keepdim=True)

        if pred_after_attack != gt_label:
            success += 1

        if pred_after_attack != pred_before_attack:
            fool_rate += 1

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
                   "number of patches" : 1})

        log_table.add_data(batch_idx, wandb.Image(data), gt_label, pred_before_attack, maxval,wandb.Image(attacked_img),\
                           pred_after_attack, maxval_adv)
    artifact.add(log_table, "predictions")
    wandb.run.log_artifact(artifact)
        #wandb.log({"Deit_tiny" : log_table})

#print(scr.required_iterations[-1])