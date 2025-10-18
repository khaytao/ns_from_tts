from model import GradTTS_NS
import torch
import controlnet_params as params
import json

INFERENCE_PATH = r"D:\projects\ns_from_tts\resources\filelists\clean_lj"

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

    n_feats = params.n_feats
    n_fft = params.n_fft
    sample_rate = params.sample_rate
    hop_length = params.hop_length
    win_length = params.win_length
    f_min = params.f_min
    f_max = params.f_max

    model = GradTTS_NS(nsymbols, 1, 64, n_enc_channels, filter_channels, filter_channels_dp,
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
                    n_feats, dec_dim, beta_min, beta_max, pe_scale).cuda()
    base_state_dict_path = r"D:\projects\ns_from_tts\checkpts\grad-tts.pt"
    base_state_dict = torch.load(base_state_dict_path)  #todo use relative path
    model.load_weights(base_state_dict_path)

    ckpt = model.state_dict()

    for name, param in base_state_dict.items():
        param_ckpt = ckpt[name]
        assert torch.allclose(param, param_ckpt, rtol=1e-3), f"{name} is not equal"

    torch.save(ckpt, f=f"test_state_dict.pt")


    model2 = GradTTS_NS(nsymbols, 1, None, n_enc_channels, filter_channels, filter_channels_dp,
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
                    n_feats, dec_dim, beta_min, beta_max, pe_scale).cuda()

    model2.load_state_dict(torch.load(f"test_state_dict.pt"))
    model2.decoder.estimator.is_initialized = True


    for name, param in model.named_parameters():
        param_cp = model2.state_dict()[name]
        assert torch.allclose(param, param_cp, rtol=1e-3), f"{name} is not equal"



    from data import TextMelDataset
    from text import text_to_sequence, cmudict
    CMU_PATH = '../resources/cmu_dictionary'
    # cmu = cmudict.CMUDict(CMU_PATH)
    add_blank = params.add_blank
    import sys
    sys.path.insert(0, '"../"hifi-gan')

    test_dataset = TextMelDataset(INFERENCE_PATH, CMU_PATH, add_blank,
                                  n_fft, n_feats, sample_rate, hop_length,
                                  win_length, f_min, f_max)

    i = 0
    item = test_dataset[i]
    mel = item['y'].cuda()
    mel = mel[None, :, :]

    print(f'Synthesizing {i} text...', end=' ')
    # x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
    x = item['x'].to(torch.long).unsqueeze(0).cuda()
    x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
    c_lengths = torch.LongTensor([mel.shape[-1]]).cuda()


    y1, _, _ = model(x, x_lengths, mel,c_lengths, n_timesteps=10, temperature=1.5,
                                                   stoc=False, spk=None, length_scale=1, use_mas=False)
    y2, _, _ = model2(x, x_lengths, mel,c_lengths, n_timesteps=10, temperature=1.5,
                                                   stoc=False, spk=None, length_scale=1, use_mas=False)

    assert torch.allclose(y1, y2, rtol=1e-3), "output not equal"