import cv2
import math
import numpy as np
import os
import queue
import threading
import torch
from basicsr.utils.download_util import load_file_from_url
from torch.nn import functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RealESRGANer():
    """A helper class for upsampling images with RealESRGAN.

    Args:
        scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
        model_path (str): The path to the pretrained model. It can be urls (will first download it automatically).
        model (nn.Module): The defined network. Default: None.
        tile (int): As too large images result in the out of GPU memory issue, so this tile option will first crop
            input images into tiles, and then process each of them. Finally, they will be merged into one image.
            0 denotes for do not use tile. Default: 0.
        tile_pad (int): The pad size for each tile, to remove border artifacts. Default: 10.
        pre_pad (int): Pad the input images to avoid border artifacts. Default: 10.
        half (float): Whether to use half precision during inference. Default: False.
    """

    def __init__(self,
                 scale,
                 model_path,
                 dni_weight=None,
                 model=None,
                 tile=0,
                 tile_pad=10,
                 pre_pad=10,
                 half=False,
                 device=None,
                 gpu_id=None):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half

        # initialize model
        if gpu_id:
            self.device = torch.device(
                f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu') if device is None else device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        if isinstance(model_path, list):
            # dni
            assert len(model_path) == len(dni_weight), 'model_path and dni_weight should have the save length.'
            loadnet = self.dni(model_path[0], model_path[1], dni_weight)
        else:
            # if the model_path starts with https, it will first download models to the folder: weights
            if model_path.startswith('https://'):
                model_path = load_file_from_url(
                    url=model_path, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
            loadnet = torch.load(model_path, map_location=torch.device('cpu'))

        # prefer to use params_ema
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)

        model.eval()
        self.model = model.to(self.device)
        if self.half:
            self.model = self.model.half()

    def dni(self, net_a, net_b, dni_weight, key='params', loc='cpu'):
        """Deep network interpolation.

        ``Paper: Deep Network Interpolation for Continuous Imagery Effect Transition``
        """
        net_a = torch.load(net_a, map_location=torch.device(loc))
        net_b = torch.load(net_b, map_location=torch.device(loc))
        for k, v_a in net_a[key].items():
            net_a[key][k] = dni_weight[0] * v_a + dni_weight[1] * net_b[key][k]
        return net_a

    def pre_process(self, img):
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible
        """
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        if self.half:
            self.img = self.img.half()

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        # mod pad for divisible borders
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        # model inference
        self.output = self.model(self.img)

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    with torch.no_grad():
                        output_tile = self.model(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output

    def _enhance_tensor(self, img_tensor, img_mode, outscale):
        # 直接使用张量进行推理，跳过pre_process中的numpy到tensor转换
        # 确保输入张量类型与模型权重类型匹配
        if self.half and img_tensor.dtype == torch.float32:
            img_tensor = img_tensor.half()
        self.img = img_tensor
        
        # 检查是否需要tile处理
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
        
        output_img = self.post_process()
        # 确保输出在GPU上
        # 如果使用半精度，先转换为float32以避免精度问题
        if self.half:
            output_img = output_img.data.squeeze().float().clamp_(0, 1)
        else:
            output_img = output_img.data.squeeze().clamp_(0, 1)
        
        # 根据需要调整输出比例
        if outscale is not None:
            h, w = output_img.shape[1:]  # 假设输出格式是 (C, H, W)
            target_h = int(h * outscale / self.scale)
            target_w = int(w * outscale / self.scale)
            output_img = output_img.unsqueeze(0)  # 添加批次维度
            output_img = F.interpolate(
                output_img, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # 转换为BGR格式并调整维度顺序
        output_img = output_img[[2, 1, 0], :, :]  # RGB -> BGR
        output_img = output_img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
        
        return output_img, img_mode

    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        # 检查输入类型
        is_tensor_input = isinstance(img, torch.Tensor)
        
        if is_tensor_input:
            # 如果输入已经是PyTorch张量
            h_input, w_input = img.shape[2], img.shape[3]  # 假设输入格式为 (1, C, H, W)
            img_tensor = img
            img_mode = 'RGB'  # 假设输入是RGB格式
            max_range = 255
            output_img, img_mode = self._enhance_tensor(img_tensor, img_mode, outscale)
        else:
            # 传统numpy数组处理路径
            h_input, w_input = img.shape[0:2]
            # img: numpy
            img = img.astype(np.float32)
            if np.max(img) > 256:  # 16-bit image
                max_range = 65535
                print('\tInput is a 16-bit image')
            else:
                max_range = 255
            img = img / max_range
            if len(img.shape) == 2:  # gray image
                img_mode = 'L'
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA image with alpha channel
                img_mode = 'RGBA'
                alpha = img[:, :, 3]
                img = img[:, :, 0:3]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if alpha_upsampler == 'realesrgan':
                    alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
            else:
                img_mode = 'RGB'
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # ------------------- process image (without the alpha channel) ------------------- #
            self.pre_process(img)
            if self.tile_size > 0:
                self.tile_process()
            else:
                self.process()
            output_img = self.post_process()
            # 保持在GPU上进行所有处理，避免转回CPU
            output_img = output_img.data.squeeze().float().clamp_(0, 1)
            # 使用PyTorch操作在GPU上完成颜色通道转换，而不是转回numpy
            output_img = output_img[[2, 1, 0], :, :]  # RGB -> BGR
            output_img = output_img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
        
        if img_mode == 'L':
            # 在GPU上转换为灰度图像
            # 使用标准亮度公式: 0.299*R + 0.587*G + 0.114*B
            output_img = 0.299 * output_img[:, :, 2] + 0.587 * output_img[:, :, 1] + 0.114 * output_img[:, :, 0]
            output_img = output_img.unsqueeze(2)  # 恢复通道维度

        # ------------------- process the alpha channel if necessary ------------------- #
        if img_mode == 'RGBA' and not is_tensor_input:  # 张量输入暂时不处理alpha通道
            if alpha_upsampler == 'realesrgan':
                self.pre_process(alpha)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_alpha = self.post_process()
                # 保持在GPU上处理alpha通道
                output_alpha = output_alpha.data.squeeze().float().clamp_(0, 1)
                output_alpha = output_alpha[[2, 1, 0], :, :]  # RGB -> BGR
                output_alpha = output_alpha.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
                # 在GPU上转换为灰度
                output_alpha = 0.299 * output_alpha[:, :, 2] + 0.587 * output_alpha[:, :, 1] + 0.114 * output_alpha[:, :, 0]
            else:  # 使用CUDA加速的resize而不是cv2
                # 将alpha转换为tensor并移至GPU
                alpha_tensor = torch.from_numpy(alpha).float().to(self.device)
                alpha_tensor = alpha_tensor.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
                # 使用PyTorch的插值函数在GPU上调整大小
                output_alpha = F.interpolate(
                    alpha_tensor, 
                    size=(img.shape[0] * self.scale, img.shape[1] * self.scale), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze()

            # 在GPU上合并alpha通道
            output_img = torch.cat([output_img, output_alpha.unsqueeze(2)], dim=2)

        # ------------------------------ 转换为最终格式并返回 ------------------------------ #
        # 如果是张量输入，且不需要额外的缩放，保持在GPU上
        if is_tensor_input and (outscale is None or outscale == float(self.scale)):
            # 将结果从[0,1]范围转换到[0,255]范围并转换为uint8
            output = (output_img * 255.0).round().byte()
            # 保持在GPU上
            return output, img_mode
        else:
            # 仅在最后一步转回CPU并转换为numpy数组
            output_img = output_img.cpu().numpy()
            
            if max_range == 65535:  # 16-bit image
                output = (output_img * 65535.0).round().astype(np.uint16)
            else:
                output = (output_img * 255.0).round().astype(np.uint8)

            if outscale is not None and outscale != float(self.scale):
                # 如果需要不同的输出比例，使用PyTorch在GPU上完成
                output_tensor = torch.from_numpy(output).to(self.device).float()
                if img_mode == 'RGBA':
                    output_tensor = output_tensor.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
                    output_tensor = F.interpolate(
                        output_tensor, 
                        size=(int(h_input * outscale), int(w_input * outscale)), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0).cpu().numpy()
                    output = (output_tensor).round().astype(np.uint8)
                else:
                    # 灰度图像处理
                    if len(output_tensor.shape) == 2:
                        output_tensor = output_tensor.unsqueeze(0).unsqueeze(0)
                    elif len(output_tensor.shape) == 3:
                        output_tensor = output_tensor.permute(2, 0, 1).unsqueeze(0)
                    output_tensor = F.interpolate(
                        output_tensor, 
                        size=(int(h_input * outscale), int(w_input * outscale)), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    if img_mode == 'L':
                        output = output_tensor.squeeze().cpu().numpy().round().astype(np.uint8)
                    else:
                        output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().round().astype(np.uint8)

        return output, img_mode


class PrefetchReader(threading.Thread):
    """Prefetch images.

    Args:
        img_list (list[str]): A image list of image paths to be read.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, img_list, num_prefetch_queue):
        super().__init__()
        self.que = queue.Queue(num_prefetch_queue)
        self.img_list = img_list

    def run(self):
        for img_path in self.img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            self.que.put(img)

        self.que.put(None)

    def __next__(self):
        next_item = self.que.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class IOConsumer(threading.Thread):

    def __init__(self, opt, que, qid):
        super().__init__()
        self._queue = que
        self.qid = qid
        self.opt = opt

    def run(self):
        while True:
            msg = self._queue.get()
            if isinstance(msg, str) and msg == 'quit':
                break

            output = msg['output']
            save_path = msg['save_path']
            cv2.imwrite(save_path, output)
        print(f'IO worker {self.qid} is done.')
