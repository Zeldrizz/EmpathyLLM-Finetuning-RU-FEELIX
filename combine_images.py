from PIL import Image

folder = "llama-3.1-8b-try4"

image1 = Image.open(f"{folder}/train_loss_corrected.png")
image2 = Image.open(f"{folder}/eval_loss_corrected.png")
image3 = Image.open(f"{folder}/grad_norm_corrected.png")
image4 = Image.open(f"{folder}/learning_rate_corrected.png")

image_size = image1.size

result_image = Image.new("RGB", (image_size[0] * 2, image_size[1] * 2))

result_image.paste(image1, (0, 0))
result_image.paste(image2, (image_size[0], 0))
result_image.paste(image3, (0, image_size[1]))
result_image.paste(image4, (image_size[0], image_size[1]))

result_image.save(f"{folder}/total_statistics.png")