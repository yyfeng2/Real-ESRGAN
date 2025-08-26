import argparse
import fix_torchvision_import  # 添加这一行来修复 torchvision 导入问题
import cv2
import glob
import mimetypes
import numpy as np
import os
import shutil
import subprocess
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from os import path as osp
from tqdm import tqdm
import imageio

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# 移除ffmpeg-python依赖，使用imageio[ffmpeg]
def get_video_meta_info(video_path):
    """使用imageio获取视频元数据"""
    ret = {}
    try:
        reader = imageio.get_reader(video_path, 'ffmpeg')
        meta = reader.get_meta_data()
        ret['width'] = meta.get('size', (0, 0))[0]
        ret['height'] = meta.get('size', (0, 0))[1]
        ret['fps'] = meta.get('fps', 24)
        ret['nb_frames'] = meta.get('nframes', 0)
        # imageio不直接提供音频获取，我们设置为None
        ret['audio'] = None
        reader.close()
    except Exception as e:
        print(f"获取视频元数据失败: {e}")
    return ret


def get_sub_video(args, num_process, process_idx):
    """使用imageio分割视频"""
    if num_process == 1:
        return args.input

    meta = get_video_meta_info(args.input)
    duration = int(meta['nb_frames'] / meta['fps'])
    part_time = duration // num_process
    start_frame = part_time * process_idx * meta['fps']
    end_frame = start_frame + part_time * meta['fps'] if process_idx != num_process - 1 else meta['nb_frames']

    print(f'duration: {duration}, part_time: {part_time}, start_frame: {start_frame}, end_frame: {end_frame}')
    os.makedirs(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'), exist_ok=True)
    out_path = osp.join(args.output, f'{args.video_name}_inp_tmp_videos', f'{process_idx:03d}.mp4')

    # 使用imageio读取并写入视频片段
    reader = imageio.get_reader(args.input, 'ffmpeg')
    writer = imageio.get_writer(out_path, fps=meta['fps'], codec='libx264', pixelformat='yuv420p')

    for i, frame in enumerate(reader):
        if start_frame <= i < end_frame:
            writer.append_data(frame)
        elif i >= end_frame:
            break

    reader.close()
    writer.close()
    return out_path


class Reader:

    def __init__(self, args, total_workers=1, worker_idx=0):
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        self.input_type = 'folder' if input_type is None else input_type
        self.paths = []  # for image&folder type
        self.audio = None
        self.input_fps = None
        if self.input_type.startswith('video'):
            video_path = get_sub_video(args, total_workers, worker_idx)
            self.reader = imageio.get_reader(video_path, 'ffmpeg')
            meta = self.reader.get_meta_data()
            self.width = meta.get('size', (0, 0))[0]
            self.height = meta.get('size', (0, 0))[1]
            self.input_fps = meta.get('fps', 24)
            # 初始化nb_frames为0
            self.nb_frames = 0
            # 尝试获取nframes，处理可能的无穷大情况
            try:
                nframes = meta.get('nframes', 0)
                # 检查是否是有限数值
                if isinstance(nframes, (int, float)) and nframes > 0 and nframes < float('inf'):
                    self.nb_frames = int(nframes)
            except Exception as e:
                print(f"获取视频帧数失败: {e}")
            
            # 如果nframes不可用或无效，使用duration近似计算
            if self.nb_frames <= 0:
                try:
                    duration = meta.get('duration', 0)
                    if isinstance(duration, (int, float)) and duration > 0 and duration < float('inf'):
                        self.nb_frames = int(duration * self.input_fps)
                    else:
                        # 如果所有方法都失败，设置一个合理的默认值
                        self.nb_frames = 300  # 默认300帧
                except Exception as e:
                    print(f"计算视频帧数失败: {e}")
                    self.nb_frames = 300  # 默认300帧
            
            self.current_frame = 0
            self.video_reader = iter(self.reader)

        else:
            if self.input_type.startswith('image'):
                self.paths = [args.input]
            else:
                paths = sorted(glob.glob(os.path.join(args.input, '*')))
                tot_frames = len(paths)
                num_frame_per_worker = tot_frames // total_workers + (1 if tot_frames % total_workers else 0)
                self.paths = paths[num_frame_per_worker * worker_idx:num_frame_per_worker * (worker_idx + 1)]

            self.nb_frames = len(self.paths)
            assert self.nb_frames > 0, 'empty folder'
            from PIL import Image
            tmp_img = Image.open(self.paths[0])
            self.width, self.height = tmp_img.size
        self.idx = 0

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        # 确保返回整数
        return int(self.nb_frames)

    def get_frame_from_video(self):
        if self.current_frame >= self.nb_frames:
            return None
        try:
            # 使用imageio直接获取下一帧
            frame = next(self.video_reader)
            # 确保是BGR格式（与OpenCV兼容）
            if len(frame.shape) == 3 and frame.shape[2] >= 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            self.current_frame += 1
            return frame_bgr
        except StopIteration:
            return None
        except Exception as e:
            print(f"读取视频帧错误: {e}")
            return None

    def get_frame_from_list(self):
        if self.idx >= self.nb_frames:
            return None
        img = cv2.imread(self.paths[self.idx])
        self.idx += 1
        return img

    def get_frame(self):
        if self.input_type.startswith('video'):
            return self.get_frame_from_video()
        else:
            return self.get_frame_from_list()

    def close(self):
        if hasattr(self, 'reader') and self.reader is not None:
            self.reader.close()


class Writer:

    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)
        if out_height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')

        # 使用imageio创建视频写入器，不使用size参数
        self.writer = imageio.get_writer(
            video_save_path,
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p'
        )
        self.out_width = out_width
        self.out_height = out_height

    def write_frame(self, frame):
        # imageio需要RGB格式，而OpenCV是BGR，所以需要转换
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 确保frame尺寸正确
        if frame_rgb.shape[1] != self.out_width or frame_rgb.shape[0] != self.out_height:
            frame_rgb = cv2.resize(frame_rgb, (self.out_width, self.out_height))
        self.writer.append_data(frame_rgb)

    def close(self):
        self.writer.close()


def inference_video(args, video_save_path, device=None, total_workers=1, worker_idx=0):
    # ---------------------- determine models according to model names ---------------------- #
    args.model_name = args.model_name.split('.pth')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # ---------------------- determine model paths ---------------------- #
    model_path = os.path.join('weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        device=device,
    )

    if 'anime' in args.model_name and args.face_enhance:
        print('face_enhance is not supported in anime models, we turned this option off for you. '
              'if you insist on turning it on, please manually comment the relevant lines of code.')
        args.face_enhance = False

    if args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)  # TODO support custom device
    else:
        face_enhancer = None

    reader = Reader(args, total_workers, worker_idx)
    audio = reader.get_audio()
    height, width = reader.get_resolution()
    fps = reader.get_fps()
    writer = Writer(args, audio, height, width, video_save_path, fps)

    pbar = tqdm(total=len(reader), unit='frame', desc='inference')
    while True:
        img = reader.get_frame()
        if img is None:
            break

        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print('Error', error)

        else:
            writer.write_frame(output)

        torch.cuda.synchronize(device)
        pbar.update(1)

    reader.close()
    writer.close()


def run(args):
    args.video_name = osp.splitext(os.path.basename(args.input))[0]
    video_save_path = osp.join(args.output, f'{args.video_name}_{args.suffix}.mp4')

    if args.extract_frame_first:
        tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
        os.makedirs(tmp_frames_folder, exist_ok=True)
        # 使用imageio读取视频并截取部分
        reader = imageio.get_reader(args.input, 'ffmpeg')
        for i, frame in enumerate(reader):
            frame_path = osp.join(tmp_frames_folder, f'frame{i:08d}.png')
            imageio.imwrite(frame_path, frame)
        reader.close()
        args.input = tmp_frames_folder

    num_gpus = torch.cuda.device_count()
    num_process = num_gpus * args.num_process_per_gpu
    if num_process == 1:
        inference_video(args, video_save_path)
        return

    ctx = torch.multiprocessing.get_context('spawn')
    pool = ctx.Pool(num_process)
    os.makedirs(osp.join(args.output, f'{args.video_name}_out_tmp_videos'), exist_ok=True)
    pbar = tqdm(total=num_process, unit='sub_video', desc='inference')
    for i in range(num_process):
        sub_video_save_path = osp.join(args.output, f'{args.video_name}_out_tmp_videos', f'{i:03d}.mp4')
        pool.apply_async(
            inference_video,
            args=(args, sub_video_save_path, torch.device(i % num_gpus), num_process, i),
            callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()

    # combine sub videos
    # prepare vidlist.txt
    with open(f'{args.output}/{args.video_name}_vidlist.txt', 'w') as f:
        for i in range(num_process):
            f.write(f'file \'{args.video_name}_out_tmp_videos/{i:03d}.mp4\'\n')

    # 使用imageio合并视频
    writer = None
    for i in range(num_process):
        sub_video_path = osp.join(args.output, f'{args.video_name}_out_tmp_videos', f'{i:03d}.mp4')
        reader = imageio.get_reader(sub_video_path, 'ffmpeg')
        meta = reader.get_meta_data()
        fps = meta.get('fps', 24)
        size = meta.get('size', (0, 0))

        if writer is None:
            # 创建写入器
            writer = imageio.get_writer(
                video_save_path,
                fps=fps,
                codec='libx264',
                pixelformat='yuv420p'
            )

        for frame in reader:
            writer.append_data(frame)
        reader.close()

    if writer is not None:
        writer.close()


def main():
    """Inference demo for Real-ESRGAN.
    It mainly for restoring anime videos.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input video, image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='realesr-animevideov3',
        help=('Model names: realesr-animevideov3 | RealESRGAN_x4plus_anime_6B | RealESRGAN_x4plus | RealESRNet_x4plus |'
              ' RealESRGAN_x2plus | realesr-general-x4v3'
              'Default:realesr-animevideov3'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored video')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument('--fps', type=float, default=None, help='FPS of the output video')
    # 移除ffmpeg_bin参数，因为使用imageio[ffmpeg]
    parser.add_argument('--extract_frame_first', action='store_true')
    parser.add_argument('--num_process_per_gpu', type=int, default=1)

    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    args = parser.parse_args()

    args.input = args.input.rstrip('/').rstrip('\\')
    os.makedirs(args.output, exist_ok=True)

    # 改进视频检测逻辑
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    is_video = False
    # 检查文件扩展名
    if os.path.isfile(args.input):
        _, ext = os.path.splitext(args.input.lower())
        if ext in video_extensions:
            is_video = True
    # 或者使用 mimetypes 检测
    elif mimetypes.guess_type(args.input)[0] is not None and mimetypes.guess_type(args.input)[0].startswith('video'):
        is_video = True

    if is_video and args.input.endswith('.flv'):
        mp4_path = args.input.replace('.flv', '.mp4')
        # 使用imageio转换FLV到MP4
        try:
            reader = imageio.get_reader(args.input, 'ffmpeg')
            meta = reader.get_meta_data()
            writer = imageio.get_writer(
                mp4_path,
                fps=meta.get('fps', 24),
                codec='libx264',
                pixelformat='yuv420p'
            )
            for frame in reader:
                writer.append_data(frame)
            reader.close()
            writer.close()
            args.input = mp4_path
        except Exception as e:
            print(f"转换FLV到MP4失败: {e}")

    if args.extract_frame_first and not is_video:
        args.extract_frame_first = False

    run(args)

    if args.extract_frame_first:
        tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
        shutil.rmtree(tmp_frames_folder)


if __name__ == '__main__':
    main()
