from model import GradTTS_NS
import torch
import controlnet_params as params
import json

def test_init():

    from text.symbols import symbols

    add_blank = params.add_blank

    nsymbols = len(symbols) + 1 if add_blank else len(symbols)
    n_enc_channels = params.n_enc_channels
    filter_channels = params.filter_channels
    filter_channels_dp = params.filter_channels_dp
    n_enc_layers = params.n_enc_layers
    enc_kernel = params.enc_kernel
    enc_dropout = params.enc_dropout
    n_heads = params.n_heads
    window_size = params.window_size

    n_feats = params.n_feats

    dec_dim = params.dec_dim
    beta_min = params.beta_min
    beta_max = params.beta_max
    pe_scale = params.pe_scale

    model = GradTTS_NS(nsymbols, 1, None, n_enc_channels, filter_channels, filter_channels_dp,
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
                    n_feats, dec_dim, beta_min, beta_max, pe_scale).cuda()

def test_weightloading():
    import controlnet_params as params
    from text.symbols import symbols

    add_blank = params.add_blank

    nsymbols = len(symbols) + 1 if add_blank else len(symbols)
    n_enc_channels = params.n_enc_channels
    filter_channels = params.filter_channels
    filter_channels_dp = params.filter_channels_dp
    n_enc_layers = params.n_enc_layers
    enc_kernel = params.enc_kernel
    enc_dropout = params.enc_dropout
    n_heads = params.n_heads
    window_size = params.window_size

    n_feats = params.n_feats

    dec_dim = params.dec_dim
    beta_min = params.beta_min
    beta_max = params.beta_max
    pe_scale = params.pe_scale

    model = GradTTS_NS(nsymbols, 1, None, n_enc_channels, filter_channels, filter_channels_dp,
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
                    n_feats, dec_dim, beta_min, beta_max, pe_scale).cuda()

    base_state_dict = torch.load(r"D:\projects\ns_from_tts\checkpts\grad-tts.pt")  #todo use relative path
    model.init_controlnet(base_state_dict)

    ckpt = model.state_dict()
    torch.save(ckpt, f=f"test_state_dict.pt")

    model2 = GradTTS_NS(nsymbols, 1, None, n_enc_channels, filter_channels, filter_channels_dp,
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
                    n_feats, dec_dim, beta_min, beta_max, pe_scale).cuda()

    model2.load_state_dict(torch.load(f"test_state_dict.pt"))
    model2.decoder.estimator.is_initialized = True


    for name, param in model.named_parameters():
        param_cp = model2.state_dict()[name]
        assert torch.allclose(param, param_cp, rtol=1e-3), f"{name} is not equal"
