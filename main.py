import tkinter as tk
from tkinter import filedialog, font
from PIL import Image, ImageTk
from googletrans import Translator
translator = Translator()
import OCR 


WIDTH = 760
HEIGHT = 640
IMAGE_WIDTH = int(WIDTH*0.3)
IMAGE_HEIGHT = int(HEIGHT*0.3)


def MODEL_TEST(img):
    return "This is a test!"


# def MODEL_MobileNet(img):
#     import tensorflow as tf
#     from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
#     from tensorflow.keras.preprocessing import image
#     import numpy as np
#     from PIL import Image

#     if img.mode != 'RGB':
#         img = img.convert('RGB')

#     img = img.resize((224, 224))
#     model = MobileNetV2(weights='imagenet')

#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)

#     predictions = model.predict(img_array)

#     decoded_predictions = decode_predictions(predictions, top=1)[0][0]
#     return f"This image most likely contains: {decoded_predictions[1]}"


# def MODEL_OURS(img):
#     # put our model here, input is an image, output is a string
#     # --------------- start from here ---------------

#     # --------------- end here ---------------

#     return "this is our model!"


# # select the model you want to use to recognize the image
# def model(img):
#     # when you finish the code in MODEL_OURS, please change the model_name to 'MODEL_OURS'
#     model_name = 'MODEL_MobileNet'
#     result = globals()[model_name](img)
#     result = translator.translate(result, dest='zh-cn').text
#     return result


def upload_image():
    # print("HHHHHHHHHHHHHHHH")
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg")]) #*.jpg;*.jpeg;*.png
    print(filepath)
    if filepath:
        img = Image.open(filepath)
        #img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img_photo = ImageTk.PhotoImage(img)
        image_label.config(image=img_photo)
        image_label.image = img_photo
        text_label.config(text="加载中...")
        
        output = OCR.model(img)
        text_label.config(text=output)
        output_translated = translator.translate(output, dest='en').text
        translater_label.config(text="loading...")
        translater_label.config(text=output_translated)
    else:
        text_label.config(text="图片上传失败！请重新上传！")
        translater_label.config(text="image upload failed! please try again!")


# 1. create a main window
root = tk.Tk()
root.title("cs585 ocr project")
root.geometry(f"{WIDTH}x{HEIGHT}")
root.resizable(width=False, height=False)

# 2. set background
open_image = Image.open("ai.jpg")
if open_image.size[0] != WIDTH or open_image.size[1] != HEIGHT:
    open_image = open_image.resize((WIDTH, HEIGHT))
background_image = ImageTk.PhotoImage(open_image)
background_label = tk.Label(root, image=background_image)
background_label.place(relx=0, rely=0, relwidth=0.3, relheight=1)

# 3. put some introduction text
introduction_text = "Chinese OCR through a CRNN Architecture with Attention"
intro_font = font.Font(family="Helvetica", size=15, weight="bold",)
introduction_label = tk.Label(root, justify='left',text=introduction_text, wraplength=200, font=intro_font, fg='orange')
introduction_label.place(relx=0.3, rely=0, relwidth=0.35, relheight=0.2)

# 4-1. create frames for image and text
image_frame = tk.Frame(root, bg='black', bd=2)
image_frame.place(relx=0.65, rely=0.05, relwidth=0.3, relheight=0.3)

text_frame = tk.Frame(root, bg='black', bd=2)
text_frame.place(relx=0.35, rely=0.45, relwidth=0.6, relheight=0.2)

translater_frame = tk.Frame(root, bg='black', bd=2)
translater_frame.place(relx=0.35, rely=0.78, relwidth=0.6, relheight=0.2)

text1 = tk.Label(root, text="image:", fg='black')
text1.place(relx=0.65, rely=0.0, relwidth=0.3, relheight=0.05)
text2 = tk.Label(root, text="results:", fg='black')
text2.place(relx=0.35, rely=0.4, relwidth=0.6, relheight=0.05)
text3 = tk.Label(root, text="translater:", fg='black')
text3.place(relx=0.35, rely=0.73, relwidth=0.6, relheight=0.05)

# 4-2. create labels for image and text
image_label = tk.Label(image_frame)
image_label.pack(fill='both', expand=True)

text_label = tk.Label(text_frame, text="", wraplength=WIDTH*0.57, justify='left')
text_label.pack(fill='both', expand=True)

translater_label = tk.Label(translater_frame, text="", wraplength=WIDTH*0.57, justify='left')
translater_label.pack(fill='both', expand=True)

# 5. create buttons for upload and exit
upload_button = tk.Button(root, text="upload image", command=upload_image, bg='cyan') #upload_image
upload_button.place(relx=0.42, rely=0.25, anchor='center')

exit_button = tk.Button(root, text="exit", command=root.quit, bg='lightgreen')
exit_button.place(relx=0.55, rely=0.25, anchor='center')

# 6. run
root.mainloop()
