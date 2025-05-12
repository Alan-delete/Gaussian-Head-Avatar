# import sys
# # sys.path.append('../../ext/Matte-Anything')
# sys.path.append('../ext/Matte-Anything')
# from matte_anything import generate_trimap, generate_checkerboard_image, convert_pixels
# from PIL import Image
# import numpy as np
# import torch
# import glob
# from matplotlib import pyplot as plt
# import os
# import cv2
# import torch
# import numpy as np
# import gradio as gr
# from PIL import Image
# from torchvision.ops import box_convert
# from detectron2.config import LazyConfig, instantiate
# from detectron2.checkpoint import DetectionCheckpointer
# from segment_anything import sam_model_registry, SamPredictor
# import groundingdino.datasets.transforms as T
# from groundingdino.util.inference import load_model as dino_load_model, predict as dino_predict, annotate as dino_annotate
# import argparse
# import pathlib
# import pickle as pkl
# from torchvision.transforms import Resize, InterpolationMode
# import tqdm
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning) 
# warnings.filterwarnings("ignore", category=UserWarning) 


# def init_segment_anything(model_type):
#     """
#     Initialize the segmenting anything with model_type in ['vit_b', 'vit_l', 'vit_h']
#     """
    
#     sam = sam_model_registry[model_type](checkpoint=models[model_type]).to(device)
#     predictor = SamPredictor(sam)

#     return predictor

# def init_vitmatte(model_type):
#     """
#     Initialize the vitmatte with model_type in ['vit_s', 'vit_b']
#     """
#     cfg = LazyConfig.load(vitmatte_config[model_type])
#     vitmatte = instantiate(cfg.model)
#     vitmatte.to(device)
#     vitmatte.eval()
#     DetectionCheckpointer(vitmatte).load(vitmatte_models[model_type])

#     return vitmatte

# def run_inference(input_x, selected_points, erode_kernel_size, dilate_kernel_size, fg_box_threshold, fg_text_threshold, fg_caption, 
#                     tr_box_threshold, tr_text_threshold, tr_caption = "glass, lens, crystal, diamond, bubble, bulb, web, grid"):
    
#     predictor.set_image(input_x)

#     dino_transform = T.Compose(
#     [
#         T.RandomResize([800], max_size=1333),
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ])
#     image_transformed, _ = dino_transform(Image.fromarray(input_x), None)
    
#     if len(selected_points) != 0:
#         points = torch.Tensor([p for p, _ in selected_points]).to(device).unsqueeze(1)
#         labels = torch.Tensor([int(l) for _, l in selected_points]).to(device).unsqueeze(1)
#         transformed_points = predictor.transform.apply_coords_torch(points, input_x.shape[:2])
#         print(points.size(), transformed_points.size(), labels.size(), input_x.shape, points)
#         point_coords=transformed_points.permute(1, 0, 2)
#         point_labels=labels.permute(1, 0)
#     else:
#         transformed_points, labels = None, None
#         point_coords, point_labels = None, None
    
#     if fg_caption is not None and fg_caption != "": # This section has benefited from the contributions of neuromorph,thanks! 
#         fg_boxes, logits, phrases = dino_predict(
#             model=grounding_dino,
#             image=image_transformed,
#             caption=fg_caption,
#             box_threshold=fg_box_threshold,
#             text_threshold=fg_text_threshold,
#             device=device)
#         if fg_boxes.shape[0] == 0:
#             # no fg object detected
#             transformed_boxes = None
#         else:
#             h, w, _ = input_x.shape
#             fg_boxes = torch.Tensor(fg_boxes).to(device)
#             fg_boxes = fg_boxes * torch.Tensor([w, h, w, h]).to(device)
#             fg_boxes = box_convert(boxes=fg_boxes, in_fmt="cxcywh", out_fmt="xyxy")
#             transformed_boxes = predictor.transform.apply_boxes_torch(fg_boxes, input_x.shape[:2])
#     else:
#         transformed_boxes = None
                
#     # predict segmentation according to the boxes
#     masks, scores, logits = predictor.predict_torch(
#         point_coords = point_coords,
#         point_labels = point_labels,
#         boxes = transformed_boxes,
#         multimask_output = False,
#     )
#     masks = masks.cpu().detach().numpy()
#     mask_all = np.ones((input_x.shape[0], input_x.shape[1], 3))
#     for ann in masks:
#         color_mask = np.random.random((1, 3)).tolist()[0]
#         for i in range(3):
#             mask_all[ann[0] == True, i] = color_mask[i]
#     img = input_x / 255 * 0.3 + mask_all * 0.7
    
#     # generate alpha matte
#     torch.cuda.empty_cache()
#     mask = (masks[0][0] * 255).astype(np.uint8)
#     trimap = generate_trimap(mask, erode_kernel_size, dilate_kernel_size).astype(np.float32)
#     trimap[trimap==128] = 0.5
#     trimap[trimap==255] = 1
    
#     boxes, logits, phrases = dino_predict(
#         model=grounding_dino,
#         image=image_transformed,
#         caption= tr_caption,
#         box_threshold=tr_box_threshold,
#         text_threshold=tr_text_threshold,
#         device=device)
#     annotated_frame = dino_annotate(image_source=input_x, boxes=boxes, logits=logits, phrases=phrases)
    
#     annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

#     if boxes.shape[0] == 0:
#         # no transparent object detected
#         pass
#     else:
#         h, w, _ = input_x.shape
#         boxes = boxes * torch.Tensor([w, h, w, h])
#         xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
#         trimap = convert_pixels(trimap, xyxy)

#     input = {
#         "image": torch.from_numpy(input_x).permute(2, 0, 1).unsqueeze(0)/255,
#         "trimap": torch.from_numpy(trimap).unsqueeze(0).unsqueeze(0),
#     }

#     torch.cuda.empty_cache()
#     alpha = vitmatte(input)['phas'].flatten(0,2)
#     alpha = alpha.detach().cpu().numpy()
    
#     # get a green background
#     background = generate_checkerboard_image(input_x.shape[0], input_x.shape[1], 8)

#     # calculate foreground with alpha blending
#     foreground_alpha = input_x * np.expand_dims(alpha, axis=2).repeat(3,2)/255 + background * (1 - np.expand_dims(alpha, axis=2).repeat(3,2))/255

#     # calculate foreground with mask
#     foreground_mask = input_x * np.expand_dims(mask/255, axis=2).repeat(3,2)/255 + background * (1 - np.expand_dims(mask/255, axis=2).repeat(3,2))/255

#     foreground_alpha[foreground_alpha>1] = 1
#     foreground_mask[foreground_mask>1] = 1

#     # return img, mask_all
#     trimap[trimap==1] == 0.999

#     return  mask, alpha, foreground_mask, foreground_alpha



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
import matplotlib.pyplot as plt
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
