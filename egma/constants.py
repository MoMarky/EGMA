'''drawn from Gloria github: https://github.com/marshuang80/gloria
'''

BERT_TYPE = 'emilyalsentzer/Bio_ClinicalBERT'

VIT_TYPE = 'microsoft/swin-tiny-patch4-window7-224'


# IMG_SIZE = 224
IMG_SIZE = 256
IMG_MEAN = .5862785803043838
IMG_STD = .27950088968644304

CHEXPERT_TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]
CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}

CHEXPERT_RETRIVAL_CLASS_PROMPTS = {
    "No Finding": [
        "The lungs are clear.",
        "No abnormalities are present.",
        "The chest is normal.",
        "No clinically significant radiographic abormalities.",
        "No radiographically visible abnormalities in the chest.",
    ],
    "Cardiomegaly": [
        "The heart is mildly enlarged.",
        "Cardiomegaly is present.",
        "The heart shadow is enlarged.",
        "The cardiac silhouette is enlarged.",
        "Cardiac enlargement is seen.",
    ],
    "Edema": [
        "Mild interstitial pulmonary edema is present.",
        "The presence of hazy opacity suggests interstitial pulmonary edema.",
        "Moderate alveolar edema is present.",
        "Mild diffuse opacity likely represents pulmonary edema.",
        "Cardiogenic edema likely is present.",
    ],
    "Pneumonia": [
        "A consolidation at the base likely represents pneumonia.",
        "Pneumonia is present.",
        "The presence of air bronchograms suggest pneumonia.",
        "A fluffy opacity suggests pneumonia.",
        "A pulmonary opacity with ill defined borders likely represents pneumonia.",
    ],
    "Atelectasis": [
        "Platelike opacity likely represents atelectasis.",
        "Geometric opacity likely represents atelectasis.",
        "Atelectasis is present.",
        "Basilar opacity and volume loss is likely due to atelectasis.",
        "Patchy atelectasis is seen.",
    ],
    "Pneumothorax": [
        "An apical pneumothorax is present.",
        "A basilar pneumothorax is seen.",
        "A medial pneumothorax is present adjacent to the heart.",
        "A lateral pleural line suggests pneumothorax.",
        "Pleural air is present.",
    ],
    "Pleural Effusion": [
        "A pleural effusion is present.",
        "Blunting of the costophrenic angles represents pleural effusions.",
        "Trace pleural fluid is present.",
        "The pleural space is partially filled with fluid.",
        "Layering pleural effusions are present.",
    ],
    "Fracture": [
        "An angulated fracture is present.",
        "An oblique radiolucent line suggests a fracture.",
        "A cortical step off indicates the presence of a fracture.",
        "A communuted displaced fracture is present.",
        "A fracture is present.",
    ],
}

COVID_TASKS = [
    'Normal',
    'COVID',
]
COVID_CLASS_PROMPTS = {
    'COVID': {
        'adjective': ['patchy','confluent'],
        'description': ['ground glass'],
        'subtype': ['opacity', 'consolidation'],
        'location': ['in peripheral', 'in mid', 'in lower'],
    }
}

RSNA_TASKS = [
    'Normal',
    'Pneumonia',
]
RSNA_CLASS_PROMPTS = {
    'Pneumonia': {
        'adjective': ['round', 'early', 'focal', 'multifocal', 'small', ''],
        'subtype': ['bacterial', 'viral', 'mycoplasma', ''],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left middle lobe",
            "at the right middle lobe",
            ""
        ]
    },
}

WEIGHTS_NAME = 'pytorch_model.bin'

SIIMACR_TASKS = [
    "No Finding",
    "Pneumothorax",
]
SIIMACR_CLASS_PROMPTS = {
    "Pneumothorax": [
        "An apical pneumothorax is present.",
        "A basilar pneumothorax is seen.",
        "A medial pneumothorax is present adjacent to the heart.",
        "A lateral pleural line suggests pneumothorax.",
        "Pleural air is present.",
    ],
    "No Finding": [
        "The lungs are clear.",
        "No abnormalities are present.",
        "The chest is normal.",
        "No clinically significant radiographic abormalities.",
        "No radiographically visible abnormalities in the chest.",
    ],
}

# store the URL of pretrained weights, `dev` needs to change to `main` after merging it to main branch.
PRETRAINED_URL_MEDCLIP_RESNET = 'https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_resnet_weight.txt'
PRETRAINED_URL_MEDCLIP_VIT = 'https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_vit_weight.txt'