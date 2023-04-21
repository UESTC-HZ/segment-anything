from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

model = {
    "vit_h": "model/sam_vit_h_4b8939.pth",
    "vit_l": "model/sam_vit_l_0b3195.pth",
    "vit_b": "model/sam_vit_b_01ec64.pth",
}


def load_predictor_model(model_type, device="cuda"):
    sam_checkpoint = model[model_type]
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    print("Loaded Model: " + sam_checkpoint)
    predictor = SamPredictor(sam)
    return predictor


def load_generator_model(model_type, device="cuda"):
    sam_checkpoint = model[model_type]
    device = device
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    generator = SamAutomaticMaskGenerator(sam)
    return generator
