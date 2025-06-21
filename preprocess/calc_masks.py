
import cv2 as cv
import os 
from torchvision import transforms
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import glob
import sys
import pathlib
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import argparse


def postprocess_mask(tensor):
    image = np.array(tensor) * 255.
    image = np.maximum(np.minimum(image, 255), 0)
    return image.astype(np.uint8)

def obtain_modnet_mask(im: torch.tensor, modnet: nn.Module,
                       ref_size = 512,):
    transes = [ transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ]
    im_transform = transforms.Compose( transes)
    im = im_transform(im)
    im = im[None, :, :, :]

    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')
    
    _, _, matte = modnet(im, True)
    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    return matte[None]


def valid(model, valloader, input_size, image_size, num_samples, gpus):
    model.eval()

    parsing_preds = np.zeros((num_samples, image_size[0], image_size[1]),
                             dtype=np.uint8)

    hpreds_lst = []
    wpreds_lst = []

    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    eval_scale=[0.66, 0.80, 1.0]
    # eval_scale=[1.0]
    flipped_idx = (15, 14, 17, 16, 19, 18)
    with torch.no_grad():
        for index, image in enumerate(valloader):
            # num_images = image.size(0)
            # print( image.size() )
            # image = image.squeeze()
            if index % 10 == 0:
                print('%d  processd' % (index * 1))
            #====================================================================================            
            mul_outputs = []
            for scale in eval_scale:                
                interp_img = torch.nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
                scaled_img = interp_img( image )   
                # print( scaled_img.size() )             
                outputs = model( scaled_img.cuda() )
                prediction = outputs[0][-1]
                #==========================================================
                hPreds = outputs[2][0]
                wPreds = outputs[2][1]
                hpreds_lst.append( hPreds[0].data.cpu().numpy() )
                wpreds_lst.append( wPreds[0].data.cpu().numpy() )
                #==========================================================
                single_output = prediction[0]
                flipped_output = prediction[1]
                flipped_output[14:20,:,:]=flipped_output[flipped_idx,:,:]
                single_output += flipped_output.flip(dims=[-1])
                single_output *=0.5
                # print( single_output.size() )
                single_output = interp( single_output.unsqueeze(0) )                 
                mul_outputs.append( single_output[0] )
            fused_prediction = torch.stack( mul_outputs )
            fused_prediction = fused_prediction.mean(0)
            fused_prediction = F.interpolate(fused_prediction[None], size=image_size, mode='bicubic')[0]
            fused_prediction = fused_prediction.permute(1, 2, 0)  # HWC
            fused_prediction = torch.argmax(fused_prediction, dim=2)
            fused_prediction = fused_prediction.data.cpu().numpy()
            parsing_preds[idx, :, :] = np.asarray(fused_prediction, dtype=np.uint8)
            #==================================================================================== 
            idx += 1

    parsing_preds = parsing_preds[:num_samples, :, :]
    return parsing_preds, hpreds_lst, wpreds_lst


    
def main(args):
    print("Start calculating masks!")

    # data_dir should be something like 'datasets/MeRSemble/031'
    os.makedirs(f'{data_dir}/NeuralHaircut_masks/hair', exist_ok=True)
    os.makedirs(f'{data_dir}/NeuralHaircut_masks/body', exist_ok=True)
    os.makedirs(f'{data_dir}/NeuralHaircut_masks/face', exist_ok=True)

    # under the images, there are folders for each frame, we need to create the same folder structure for NeuralHaircut_masks
    frames_names = os.listdir(f'{data_dir}/images') 
    for frame in frames_names:
        mask_folder = os.path.join(f'{data_dir}/NeuralHaircut_masks/hair', frame) 
        os.makedirs(mask_folder, exist_ok=True)
        mask_folder = os.path.join(f'{data_dir}/NeuralHaircut_masks/body', frame)
        os.makedirs(mask_folder, exist_ok=True)
        mask_folder = os.path.join(f'{data_dir}/NeuralHaircut_masks/face', frame)
        os.makedirs(mask_folder, exist_ok=True)
        mask_folder = os.path.join(f'{data_dir}/NeuralHaircut_masks/head', frame)
        os.makedirs(mask_folder, exist_ok=True)


    # there may be legacy mask images in image folder, so we need to filter out those by using image_[0-9]*.
    # filepaths = glob.glob(os.path.join(data_dir, 'images', '*', f'image_[0-9]*.{args.image_format}'))
    images = sorted(glob.glob(os.path.join(data_dir, 'images', '*', f'image_*.{args.image_format}')))
    n_images = len(images)
    
    # tens_list = []
    # for i in range(n_images):
    #     tens_list.append(T.ToTensor()(Image.open(images[i])))

#     load MODNET model for silhouette masks
    modnet = nn.DataParallel(MODNet(backbone_pretrained=False))
    modnet.load_state_dict(torch.load(args.MODNET_ckpt))
    device = torch.device('cuda')
    modnet.eval().to(device)
    
    # Create silh masks
    silh_list = []
    for i in tqdm(range(len(images))):
        data = T.ToTensor()(Image.open(images[i]))
        silh_mask = obtain_modnet_mask(data, modnet, 512)
        silh_list.append(silh_mask)
        cv2.imwrite(images[i].replace('images', 'NeuralHaircut_masks/body'), postprocess_mask(silh_mask)[0].astype(np.uint8))
        # cv2.imwrite(images[i].replace('image_', 'mask_'), postprocess_mask(silh_mask)[0].astype(np.uint8))
    
    print("Start calculating hair masks!")
#     load CDGNet for hair masks
    model = Res_Deeplab(num_classes=20)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    state_dict = model.state_dict().copy()
    state_dict_old = torch.load(args.CDGNET_ckpt, map_location='cpu').copy()

    for key in state_dict.keys():
        if key in state_dict_old.keys():
            state_dict[key] = deepcopy(state_dict_old[key])
        elif 'module.' + key in state_dict_old.keys():
            nkey = 'module.' + key
            state_dict[key] = deepcopy(state_dict_old[nkey])


    # for key, nkey in zip(state_dict_old.keys(), state_dict.keys()):
    #     if key != nkey:
    #         # remove the 'module.' in the 'key'
    #         state_dict[key[7:]] = deepcopy(state_dict_old[key])
    #     else:
    #         state_dict[key] = deepcopy(state_dict_old[key])

    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()

    # basenames = sorted([s.split('.')[0] for s in os.listdir(os.path.join(args.scene_path, 'image'))])
    input_size = (1024, 1024)
    filepaths = sorted(glob.glob(os.path.join(data_dir, 'images', '*', f'image_*.{args.image_format}')))
    filepaths_blocks = [ filepaths[i:i+5000] for i in range(0, len(filepaths), 5000) ]

    for filepaths in filepaths_blocks:
        raw_images = []
        images = []
        masks = []
        for image in tqdm(filepaths):
            img = Image.open(image)
            raw_images.append(np.asarray(img))
            img = transform(img.resize(input_size))[None]
            img = torch.cat([img, torch.flip(img, dims=[-1])], dim=0)
            mask = np.asarray(Image.open(image.replace('images', 'NeuralHaircut_masks/body')))
            mask = cv2.resize(mask, input_size)
            images.append(img)
            masks.append(mask)

        # image_size = (mask.shape[1], mask.shape[0])
        image_size = input_size
        parsing_preds, hpredLst, wpredLst = valid(model, images, input_size, image_size, len(images), gpus=1)

        for i in range(len(images)):
            hair_mask = np.asarray(Image.fromarray((parsing_preds[i] == 2)).resize(image_size, Image.BICUBIC))
            hair_mask = hair_mask * masks[i]

            face_mask = np.asarray(Image.fromarray((parsing_preds[i] == 13)).resize(image_size, Image.BICUBIC))
            face_mask = face_mask * masks[i]

            head_mask = face_mask | hair_mask
            Image.fromarray(head_mask).save(filepaths[i].replace('images', 'NeuralHaircut_masks/head'))

            Image.fromarray(hair_mask).save(filepaths[i].replace('images', 'NeuralHaircut_masks/hair'))
   
    print('Results saved in folder: ', os.path.join(data_dir, 'NeuralHaircut_masks'))
        
model_dir = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'ext', 'Matte-Anything')

if __name__ == '__main__':

#     --MODNET_ckpt $PROJECT_DIR/assets/MODNet/modnet_photographic_portrait_matting.ckpt \
#     --CDGNET_ckpt $PROJECT_DIR/assets/CDGNet/LIP_epoch_149.pth \
#     --ext_dir $PROJECT_DIR/ext/
#     # --data_path $DATA_PATH --model_dir $PROJECT_DIR/ext/Matte-Anything --img_size 512 \


    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--image_format', default='jpg', type=str)
    parser.add_argument('--model_dir', default=model_dir, type=str)
    parser.add_argument('--postfix', default='', type=str)
    parser.add_argument('--img_size', default=-1, type=int)
    parser.add_argument('--max_size', default=-1, type=int)
    parser.add_argument('--kernel_size', default=10, type=int)
    parser.add_argument('--MODNET_ckpt', default='../../ext/MODNet/modnet_photographic_portrait_matting.ckpt', type=str)
    parser.add_argument('--CDGNET_ckpt', default='../../ext/CDGNet/LIP_epoch_149.pth', type=str)
    parser.add_argument('--ext_dir', default='../../ext/', type=str)
    parser.add_argument('--model_dir', default=model_dir, type=str)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    img_size = args.img_size
    max_size = args.max_size
    data_dir = args.data_path
    model_dir = args.model_dir

    sys.path.append(args.ext_dir)

    # calc silh masks
    from MODNet.src.models.modnet import MODNet
    from tqdm import tqdm

    sys.path.append(os.path.join(args.ext_dir, 'CDGNet'))
    # calc hair masks
    from CDGNet.networks.CDGNet import Res_Deeplab
    import os
    from copy import deepcopy

    # sys.path.append(os.path.join(args.ext_dir, 'BiRefNet'))
    # # Use codes locally
    # from models.birefnet import BiRefNet

    # # Load weights from Hugging Face Models
    # birefnet = BiRefNet.from_pretrained('ZhengPeng7/BiRefNet')
    # torch.set_float32_matmul_precision(['high', 'highest'][0])
    # birefnet.to('cuda')
    # birefnet.eval()
    # birefnet.half()

    # def extract_object(birefnet, imagepath):
    #     # Data settings
    #     image_size = (1024, 1024)
    #     transform_image = transforms.Compose([
    #         transforms.Resize(image_size),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ])

    #     image = Image.open(imagepath)
    #     input_images = transform_image(image).unsqueeze(0).to('cuda').half()

    #     # Prediction
    #     with torch.no_grad():
    #         preds = birefnet(input_images)[-1].sigmoid().cpu()
    #     pred = preds[0].squeeze()
    #     pred_pil = transforms.ToPILImage()(pred)
    #     mask = pred_pil.resize(image.size)
    #     image.putalpha(mask)
    #     breakpoint()
    #     # save the mask
    #     mask = np.array(mask)
    #     mask = cv.resize(mask, image_size, interpolation=cv.INTER_NEAREST)
        
    #     Image.fromarray(mask).save('test.png')
    #     return image, mask
    
    # # Visualization
    # image, mask = extract_object(birefnet, imagepath='/local/home/haonchen/Gaussian-Head-Avatar/datasets/mini_demo_dataset/100/images/0003/image_222200036.jpg')[0]


    main(args)
    sys.exit(0)


    # models = {
    #     'vit_h': f'{model_dir}/pretrained/sam_vit_h_4b8939.pth',
    #     'vit_b': f'{model_dir}/pretrained/sam_vit_b_01ec64.pth'
    # }

    # vitmatte_models = {
    #     'vit_b': f'{model_dir}/pretrained/ViTMatte_B_DIS.pth',
    # }

    # vitmatte_config = {
    #     'vit_b': f'{model_dir}/configs/matte_anything.py',
    # }

    # grounding_dino = {
    #     'config': f'{model_dir}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
    #     'weight': f'{model_dir}/pretrained/groundingdino_swint_ogc.pth'
    # }

    # device = 'cuda'
    # # data_dir should be something like 'datasets/MeRSemble/031'
    # os.makedirs(f'{data_dir}/masks{args.postfix}/hair', exist_ok=True)
    # os.makedirs(f'{data_dir}/masks{args.postfix}/body', exist_ok=True)
    # os.makedirs(f'{data_dir}/masks{args.postfix}/face', exist_ok=True)

    # sam_model = 'vit_h'
    # vitmatte_model = 'vit_b'

    # predictor = init_segment_anything(sam_model)
    # vitmatte = init_vitmatte(vitmatte_model)
    # grounding_dino = dino_load_model(grounding_dino['config'], grounding_dino['weight'])

    # # under the images, there are folders for each frame, we need to create the same folder structure for masks
    # frames_names = os.listdir(f'{data_dir}/images') 
    # for frame in frames_names:
    #     mask_folder = os.path.join(f'{data_dir}/masks{args.postfix}/hair', frame) 
    #     os.makedirs(mask_folder, exist_ok=True)
    #     mask_folder = os.path.join(f'{data_dir}/masks{args.postfix}/body', frame)
    #     os.makedirs(mask_folder, exist_ok=True)
    #     mask_folder = os.path.join(f'{data_dir}/masks{args.postfix}/face', frame)
    #     os.makedirs(mask_folder, exist_ok=True)


    # # there may be legacy mask images in image folder, so we need to filter out those by using image_[0-9]*.
    # # filepaths = glob.glob(os.path.join(data_dir, 'images', '*', f'image_[0-9]*.{args.image_format}'))
    # filepaths = glob.glob(os.path.join(data_dir, 'images', '*', f'image_*.{args.image_format}'))
    # for filename in tqdm.tqdm(sorted(filepaths)):
    #     with torch.no_grad():
    #         img = Image.open(filename)
    #         orig_img_size = img.size
    #         if img_size != -1 or max_size != -1:
    #             img_size = max_size - 1 if img_size == -1 else img_size
    #             max_size = max_size if max_size != -1 else None
    #             img = Resize(img_size, InterpolationMode.BICUBIC, max_size)(img)

    #         # TODO:images like "image_lowres_220700191.jpg" don't need to be processed for hair mask
    #         _, mask_hair, _, _ = run_inference(
    #             np.asarray(img), [], 
    #             erode_kernel_size=args.kernel_size, 
    #             dilate_kernel_size=args.kernel_size, 
    #             fg_box_threshold=0.25, 
    #             fg_text_threshold=0.25, 
    #             fg_caption="only hair", 
    #             tr_box_threshold=0.5, 
    #             tr_text_threshold=0.25,
    #             tr_caption="glass.lens.crystal.diamond.bubble.bulb.web.grid")
                
    #         _, mask_face, _, _ = run_inference(
    #             np.asarray(img), [], 
    #             erode_kernel_size=args.kernel_size, 
    #             dilate_kernel_size=args.kernel_size, 
    #             fg_box_threshold=0.5, # higher threshold to reduce false positive
    #             fg_text_threshold=0.25, 
    #             fg_caption="face", 
    #             tr_box_threshold=0.5, 
    #             tr_text_threshold=0.25,
    #             tr_caption="glass.lens.crystal.diamond.bubble.bulb.web.grid")

    #         _, mask_body, _, _ = run_inference(
    #             np.asarray(img), [], 
    #             erode_kernel_size=args.kernel_size, 
    #             dilate_kernel_size=args.kernel_size, 
    #             fg_box_threshold=0.25, 
    #             fg_text_threshold=0.25, 
    #             fg_caption="human", 
    #             tr_box_threshold=0.5, 
    #             tr_text_threshold=0.25,
    #             tr_caption="glass.lens.crystal.diamond.bubble.bulb.web.grid")
            
    #         mask_hair = Image.fromarray((mask_hair * 255).astype('uint8'))
    #         mask_face = Image.fromarray((mask_face * 255).astype('uint8'))
    #         mask_body = Image.fromarray((mask_body * 255).astype('uint8'))

    #         if img_size != -1:
    #             mask_hair = mask_hair.resize(orig_img_size, Image.BICUBIC)
    #             mask_face = mask_face.resize(orig_img_size, Image.BICUBIC)
    #             mask_body = mask_body.resize(orig_img_size, Image.BICUBIC)

    #         mask_hair.save(filename.replace(f'images', f'masks{args.postfix}/hair').replace(args.image_format, 'jpg'))
    #         mask_face.save(filename.replace(f'images', f'masks{args.postfix}/face').replace(args.image_format, 'jpg'))
    #         mask_body.save(filename.replace(f'images', f'masks{args.postfix}/body').replace(args.image_format, 'jpg'))
            
    #         # HAVATAR way of data storage
    #         mask_body.save(filename.replace(f'image_', f'mask_'))
