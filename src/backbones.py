import cv2
import torch
import torchvision.models as models
# import clip
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import numpy as np
from transformers import AutoImageProcessor, AutoModel


# Base Wrapper Class
class VisionTransformerWrapper:
    def __init__(self, model_name, device, smaller_edge_size=224, half_precision=False, weights_path=None):
        self.device = device
        self.smaller_edge_size = smaller_edge_size
        self.half_precision = half_precision
        self.model_name = model_name
        self.weights_path = weights_path
        self.patch_size = None
        self.resize_transform = None
        self.model = self.load_model()

    def load_model(self):
        raise NotImplementedError("This method should be overridden in a subclass")
    
    def extract_features(self, img_tensor):
        raise NotImplementedError("This method should be overridden in a subclass")

    def extract_multilayer_features(self, img_tensor, layer_indices=None):
        raise NotImplementedError("This method should be overridden in a subclass")


# ViT-B/16 Wrapper
class ViTWrapper(VisionTransformerWrapper):
    def load_model(self):
        if self.model_name == "vit_b_16":
            model = models.vit_b_16(weights = models.ViT_B_16_Weights.DEFAULT)
            self.transform = models.ViT_B_16_Weights.DEFAULT.transforms()
            self.grid_size = (14,14)
        elif self.model_name == "vit_b_32":
            model = models.vit_b_32(weights = models.ViT_B_32_Weights.DEFAULT)
            self.transform = models.ViT_B_32_Weights.DEFAULT.transforms()
            self.grid_size = (7,7)
        elif self.model_name == "vit_l_16":
            model = models.vit_l_16(weights = models.ViT_L_16_Weights.DEFAULT)
            self.transform = models.ViT_L_16_Weights.DEFAULT.transforms()
            self.grid_size = (14,14)
        elif self.model_name == "vit_l_32":
            model = models.vit_l_32(weights = models.ViT_L_32_Weights.DEFAULT)
            self.transform = models.ViT_L_32_Weights.DEFAULT.transforms()
            self.grid_size = (7,7)
        else:
            raise ValueError(f"Unknown ViT model name: {self.model_name}")
        
        model.eval()
        # print(self.transform)
        if self.grid_size[0] > 0:
            self.patch_size = 16 if self.grid_size[0] == 14 else 32

        return model.to(self.device)
    
    def prepare_image(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img_tensor = self.transform(img).unsqueeze(0)
        return img_tensor, self.grid_size

    def extract_features(self, img_tensor):
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            patches = self.model._process_input(img_tensor)
            class_token = self.model.class_token.expand(patches.size(0), -1, -1)
            patches = torch.cat((class_token, patches), dim=1)
            patch_features = self.model.encoder(patches)
            return patch_features[:, 1:, :].squeeze().cpu().numpy()  # Exclude the class token

    def get_embedding_visualization(self, tokens, grid_size = (14,14), resized_mask=None, normalize=True):
        pca = PCA(n_components=3, svd_solver='randomized')
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*self.grid_size, -1))
        if normalize:
            normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
            return normalized_tokens
        else:
            return reduced_tokens

    def compute_background_mask(self, img_features, grid_size, threshold = 10, masking_type = False):
        # No masking for ViT supported at the moment... (Only DINOv2)
        return np.ones(img_features.shape[0], dtype=bool)
    

# DINOv2 Wrapper
class DINOv2Wrapper(VisionTransformerWrapper):
    def load_model(self):
        model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        model.eval()

        # print(f"Loaded model: {self.model_name}")
        # print("Resizing images to", self.smaller_edge_size)

        # Set transform for DINOv2
        self.resize_transform = transforms.Resize(
            size=self.smaller_edge_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.transform = transforms.Compose([
            self.resize_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
            ])
        self.patch_size = int(model.patch_size)
        
        return model.to(self.device)
    
    def prepare_image(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        image_tensor = self.transform(img)
        # Crop image to dimensions that are a multiple of the patch size
        height, width = image_tensor.shape[1:] # C x H x W
        cropped_width, cropped_height = width - width % self.patch_size, height - height % self.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.patch_size, cropped_width // self.patch_size)
        return image_tensor, grid_size
    

    def extract_features(self, image_tensor):
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.unsqueeze(0).half().to(self.device)
            else:
                image_batch = image_tensor.unsqueeze(0).to(self.device)

            tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
        return tokens.cpu().numpy()


    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None, normalize=True):
        pca = PCA(n_components=3, svd_solver='randomized')
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        if normalize:
            normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
            return normalized_tokens
        else:
            return reduced_tokens


    def compute_background_mask_from_image(self, image, threshold = 10, masking_type = None):
        image_tensor, grid_size = self.prepare_image(image)
        tokens = self.extract_features(image_tensor)
        return self.compute_background_mask(tokens, grid_size, threshold, masking_type)


    def compute_background_mask(self, img_features, grid_size, threshold = 10, masking_type = False, kernel_size = 3, border = 0.2):
        # Kernel size for morphological operations should be odd
        pca = PCA(n_components=1, svd_solver='randomized')
        first_pc = pca.fit_transform(img_features.astype(np.float32))
        if masking_type == True:
            mask = first_pc > threshold
            # test whether the center crop of the images is kept (adaptive masking), adapt if your objects of interest are not centered!
            m = mask.reshape(grid_size)[int(grid_size[0] * border):int(grid_size[0] * (1-border)), int(grid_size[1] * border):int(grid_size[1] * (1-border))]
            if m.sum() <=  m.size * 0.35:
                mask = - first_pc > threshold
            # postprocess mask, fill small holes in the mask, enlarge slightly
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
        elif masking_type == False:
            mask = np.ones_like(first_pc, dtype=bool)
        return mask.squeeze()


DINOv3_MODEL_MAP = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vitsplus16": "facebook/dinov3-vitsplus16-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_vithplus16": "facebook/dinov3-vithplus16-pretrain-lvd1689m",
}


class DINOv3Wrapper(VisionTransformerWrapper):
    def load_model(self):
        model_source = self.weights_path if self.weights_path not in (None, "") else DINOv3_MODEL_MAP.get(self.model_name, self.model_name)
        local_only_preferred = self.weights_path in (None, "")
        try:
            processor = AutoImageProcessor.from_pretrained(model_source, local_files_only=local_only_preferred)
            model = AutoModel.from_pretrained(model_source, local_files_only=local_only_preferred)
        except OSError:
            if local_only_preferred:
                processor = AutoImageProcessor.from_pretrained(model_source, local_files_only=False)
                model = AutoModel.from_pretrained(model_source, local_files_only=False)
            else:
                raise
        model.eval()

        self.resize_transform = transforms.Resize(
            size=self.smaller_edge_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.transform = transforms.Compose([
            self.resize_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(processor.image_mean), std=tuple(processor.image_std)),
        ])

        self.patch_size = int(model.config.patch_size)
        self.num_prefix_tokens = int(1 + getattr(model.config, "num_register_tokens", 0))
        return model.to(self.device)

    def prepare_image(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        image_tensor = self.transform(img)
        height, width = image_tensor.shape[1:]
        cropped_width, cropped_height = width - width % self.patch_size, height - height % self.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]
        grid_size = (cropped_height // self.patch_size, cropped_width // self.patch_size)
        return image_tensor, grid_size

    def extract_features(self, image_tensor):
        with torch.inference_mode():
            image_batch = image_tensor.unsqueeze(0)
            if self.half_precision and str(self.device).startswith("cuda"):
                image_batch = image_batch.half()
            image_batch = image_batch.to(self.device)
            outputs = self.model(pixel_values=image_batch, return_dict=True)
            patch_tokens = outputs.last_hidden_state[:, self.num_prefix_tokens :, :]
        return patch_tokens.squeeze(0).cpu().numpy()

    def extract_multilayer_features(self, image_tensor, layer_indices=None):
        with torch.inference_mode():
            image_batch = image_tensor.unsqueeze(0)
            if self.half_precision and str(self.device).startswith("cuda"):
                image_batch = image_batch.half()
            image_batch = image_batch.to(self.device)
            outputs = self.model(pixel_values=image_batch, return_dict=True, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("DINOv3 model did not return hidden_states.")

            num_encoder_layers = len(hidden_states) - 1
            if layer_indices is None:
                resolved_layers = list(range(1, num_encoder_layers + 1))
            else:
                resolved_layers = [int(layer) for layer in layer_indices]
                invalid = [layer for layer in resolved_layers if layer < 1 or layer > num_encoder_layers]
                if invalid:
                    raise ValueError(
                        f"Invalid DINOv3 layer indices {invalid}. Valid range is 1..{num_encoder_layers}."
                    )

            layer_tokens = []
            for layer_idx in resolved_layers:
                tokens = hidden_states[layer_idx][:, self.num_prefix_tokens :, :]
                layer_tokens.append(tokens.squeeze(0).cpu().numpy())

        stacked = np.stack(layer_tokens, axis=1)
        return stacked, resolved_layers

    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None, normalize=True):
        pca = PCA(n_components=3, svd_solver='randomized')
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        if normalize:
            normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
            return normalized_tokens
        else:
            return reduced_tokens

    def compute_background_mask_from_image(self, image, threshold = 10, masking_type = None):
        image_tensor, grid_size = self.prepare_image(image)
        tokens = self.extract_features(image_tensor)
        return self.compute_background_mask(tokens, grid_size, threshold, masking_type)

    def compute_background_mask(self, img_features, grid_size, threshold = 10, masking_type = False, kernel_size = 3, border = 0.2):
        pca = PCA(n_components=1, svd_solver='randomized')
        first_pc = pca.fit_transform(img_features.astype(np.float32))
        if masking_type == True:
            mask = first_pc > threshold
            m = mask.reshape(grid_size)[int(grid_size[0] * border):int(grid_size[0] * (1-border)), int(grid_size[1] * border):int(grid_size[1] * (1-border))]
            if m.sum() <=  m.size * 0.35:
                mask = - first_pc > threshold
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
        elif masking_type == False:
            mask = np.ones_like(first_pc, dtype=bool)
        return mask.squeeze()


def get_model(model_name, device, smaller_edge_size=448, weights_path=None):
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    print(f"Smaller edge size: {smaller_edge_size}")
    if weights_path is not None:
        print(f"Backbone weights: {weights_path}")

    if model_name.startswith("vit"):
        return ViTWrapper(model_name, device, smaller_edge_size, weights_path=weights_path)
    elif model_name.startswith("dinov2"):
        return DINOv2Wrapper(model_name, device, smaller_edge_size, weights_path=weights_path)
    elif model_name.startswith("dinov3"):
        return DINOv3Wrapper(model_name, device, smaller_edge_size, weights_path=weights_path)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
