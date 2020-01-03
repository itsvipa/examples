import os
from .neural_style.neural_style import stylize, train


def main():
    image_name = "skott.jpg"
    model_name = "midas5"
    train_image = "midas2.jpg"

    class style_args:
        content_image = "fast_neural_style/images/content-images/{}".format(image_name)
        output_image = "fast_neural_style/output/{}_styled_{}".format(model_name, image_name)
        model = "fast_neural_style/saved_models/{}.pth".format(model_name)
        content_scale = 2
        cuda = 1
        export_onnx = False

    class train_args:
        dataset = "D:/train-images"
        style_image = "fast_neural_style/images/style-images/{}".format(train_image)
        save_model_dir = "fast_neural_style/saved_models"
        cuda = 1
        epochs = 2
        batch_size = 4
        image_size = 256
        seed = 42
        content_weight = 1e5
        style_weight = 2e9
        lr = 1e-3
        log_interval = 100
        checkpoint_interval = 2000
        subset_size = 2000
        style_size = None
        checkpoint_model_dir = "fast_neural_style/saved_models/checkpoints"
        model_name = model_name + ".pth"
    if(not os.path.exists(train_args.save_model_dir + "/" + train_args.model_name)):
        train(train_args)
    if(not os.path.exists(style_args.output_image)):
        stylize(style_args)

