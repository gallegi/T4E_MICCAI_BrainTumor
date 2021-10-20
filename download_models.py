import gdown
import os

os.makedirs('pretrained_models', exist_ok=True)
url = 'https://drive.google.com/uc?id=14dmJ2HEEsdDzcsbJC_qTAQPxOGSSkDY3'
output = 'pretrained_models/eca_nfnet_l0.pth'
gdown.download(url, output, quiet=False)

os.makedirs('pretrained_models', exist_ok=True)
url = 'https://drive.google.com/uc?id=1hV4HECWeiHpFkBTES2nYPpnlYM0dApRJ'
output = 'pretrained_models/unet_pp_densenet121_2channels_out.pth'
gdown.download(url, output, quiet=False)

os.makedirs('models/densenet121_2d_segment', exist_ok=True)
url = 'https://drive.google.com/uc?id=12EVeyHI_kQlryAp6554Au4S1pt1ektnY'
output = 'models/densenet121_2d_segment/Fold0_densenet121_2d_segment.pth'
gdown.download(url, output, quiet=False)

os.makedirs('models/eca_nfnet_l0_2d_classification/T1w', exist_ok=True)
url = 'https://drive.google.com/uc?id=1wFJAurtdm8_G-nV2DVUQlWy7X4h4lBjo'
output = 'models/eca_nfnet_l0_2d_classification/T1w/T1w_Fold0_eca_nfnet_l0_2d_classification.pth'
gdown.download(url, output, quiet=False)

os.makedirs('models/eca_nfnet_l0_2d_classification/T1wCE', exist_ok=True)
url = 'https://drive.google.com/uc?id=1fmMScOnUyDWNDEYewtX7CpDMz6EFr3Vd'
output = 'models/eca_nfnet_l0_2d_classification/T1wCE/T1wCE_Fold0_eca_nfnet_l0_2d_classification.pth'
gdown.download(url, output, quiet=False)

os.makedirs('models/eca_nfnet_l0_2d_classification/T2w', exist_ok=True)
url = 'https://drive.google.com/uc?id=1JPAcR2vCDtvblDqPljx_poZNUTC00OET'
output = 'models/eca_nfnet_l0_2d_classification/T2w/T2w_Fold0_eca_nfnet_l0_2d_classification.pth'
gdown.download(url, output, quiet=False)

os.makedirs('models/eca_nfnet_l0_2d_classification/FLAIR', exist_ok=True)
url = 'https://drive.google.com/uc?id=12UBvr4ewswPWzf-8OyM4GjqYugonlTjY'
output = 'models/eca_nfnet_l0_2d_classification/FLAIR/FLAIR_Fold0_eca_nfnet_l0_2d_classification.pth'
gdown.download(url, output, quiet=False)
