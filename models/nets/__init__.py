from .SegFormer.segformer import SegFormer
from .SegNeXt.model import SegNeXt
from .SegNeXt.backbone import MSCANet
from .SegNeXt.decoder import HamDecoder
from .UNet.unet import UNet

def load_model(model_name='segformer', scale='b0', num_classes=1, in_chans=1):
    if model_name == 'segformer':
        return SegFormer(phi=scale, num_classes=num_classes, in_chans=in_chans)
    elif model_name == 'segnext':
        if scale == 't':
            return SegNeXt(num_classes=num_classes, in_channnels=in_chans, embed_dims=[32, 64, 160, 256],
                    ffn_ratios=[8, 8, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
                    dec_outChannels=256, dropout=0.0, drop_path=0.0)
        elif scale == 's':
            return SegNeXt(num_classes=num_classes, in_channnels=in_chans, embed_dims=[64, 128, 320, 512],
                    ffn_ratios=[8, 8, 4, 4], depths=[2, 2, 4, 2], num_stages=4,
                    dec_outChannels=256, dropout=0.0, drop_path=0.0)
        elif scale == 'b':
            return SegNeXt(num_classes=num_classes, in_channnels=in_chans, embed_dims=[64, 128, 320, 512],
                    ffn_ratios=[8, 8, 4, 4], depths=[3, 3, 12, 3], num_stages=4,
                    dec_outChannels=512, dropout=0.0, drop_path=0.0)
        elif scale == 'l':
            return SegNeXt(num_classes=num_classes, in_channnels=in_chans, embed_dims=[64, 128, 320, 512],
                    ffn_ratios=[8, 8, 4, 4], depths=[3, 5, 27, 3], num_stages=4,
                    dec_outChannels=1024, dropout=0.0, drop_path=0.0)
    elif model_name == 'mscan': # debug
        if scale == 't':
            return MSCANet(in_channnels=in_chans, embed_dims=[32, 64, 160, 256],
                    ffn_ratios=[8, 8, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
                    drop_path=0.0)
        elif scale == 's':
            return MSCANet(in_channnels=in_chans, embed_dims=[32, 128, 320, 512],
                    ffn_ratios=[8, 8, 4, 4], depths=[2, 2, 4, 2], num_stages=4,
                    drop_path=0.0)
        elif scale == 'b':
            return MSCANet(in_channnels=in_chans, embed_dims=[32, 128, 320, 512],
                    ffn_ratios=[8, 8, 4, 4], depths=[3, 3, 12, 3], num_stages=4,
                    drop_path=0.0)
        elif scale == 'l':
            return MSCANet(in_channnels=in_chans, embed_dims=[32, 128, 320, 512],
                    ffn_ratios=[8, 8, 4, 4], depths=[3, 5, 27, 3], num_stages=4,
                    drop_path=0.0)
    elif model_name == 'hamdecoder':
        if scale == 't':
            return HamDecoder(outChannels=256, enc_embed_dims=[32, 64, 160, 256])
        elif scale == 's':
            return HamDecoder(outChannels=256, enc_embed_dims=[64, 128, 320, 512])
        elif scale == 'b':
            return HamDecoder(outChannels=512, enc_embed_dims=[64, 128, 320, 512])
        elif scale == 'l':
            return HamDecoder(outChannels=1024, enc_embed_dims=[64, 128, 320, 512])
    elif model_name == 'unet':
        return UNet(n_channels=1, n_classes=1)